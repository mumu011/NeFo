# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
import traceback
from collections import defaultdict
import contextlib
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Seq2SeqTrainer
from typing_extensions import override
from typing import Literal
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ..callbacks import PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

from transformers import AutoModelForCausalLM
from torch.utils.data import DistributedSampler, Sampler

logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    Extended for Vision-Language Models to handle multimodal inputs during TTL training.
    """

    # Class-level flags/accumulators: shared across all instances within the same process.
    # Prevents repeated logging and accumulates timing stats across reset_trainer() calls.
    _efficiency_stats_logged: bool = False
    _eff_step_times: List[float] = []
    _eff_mem_peaks: List[float] = []
    _eff_mem_befores: List[float] = []
    _eff_mem_afters: List[float] = []
    _eff_batch_sizes: List[int] = []

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], model_args=None, data_args=None, generating_args=None, pretrain_model=None, **kwargs   # 字符串表示类型，增强静态检查的准确性
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.data_args = data_args
        self.generating_args = generating_args
        self.pretrain_model = pretrain_model

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _log_efficiency_stats_once(self, model: "torch.nn.Module") -> None:
        """Log trainable parameter count once at the first training step.

        Outputs to both the logger and <output_dir>/efficiency_stats.txt so that
        the numbers can be compared across TENT / SAR / TLM / NeFo runs.
        """
        if CustomSeq2SeqTrainer._efficiency_stats_logged:
            return
        CustomSeq2SeqTrainer._efficiency_stats_logged = True

        try:
            is_rank0 = getattr(self.accelerator, "process_index", 0) == 0
        except Exception:
            is_rank0 = True
        if not is_rank0:
            return

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        setting = getattr(self.finetuning_args, "setting", "unknown")

        lines = [
            "=" * 60,
            f"[Parameter Efficiency]  setting={setting}",
            f"  Trainable parameters : {trainable_params:>15,}",
            f"  Total parameters     : {total_params:>15,}",
            f"  Trainable ratio      : {100.0 * trainable_params / max(total_params, 1):.4f}%",
            "=" * 60,
        ]
        msg = "\n".join(lines)
        logger.info_rank0(msg)
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, "efficiency_stats.txt"), "w", encoding="utf-8") as f:
            print(msg, file=f)

    def _accumulate_and_log_efficiency(
        self,
        step_time: float,
        mem_before_mb: float,
        mem_after_mb: float,
        mem_peak_mb: float,
        batch_size: int,
        log_every: int = 10,
    ) -> None:
        """Accumulate per-step timing / memory stats and flush every *log_every* steps.

        Metrics logged:
        - Average step wall-clock time (s)
        - Per-sample adaptation overhead (ms)
        - Peak GPU memory allocated (MB)
        - Current GPU memory allocated before / after step (MB)
        """
        CustomSeq2SeqTrainer._eff_step_times.append(step_time)
        CustomSeq2SeqTrainer._eff_mem_peaks.append(mem_peak_mb)
        CustomSeq2SeqTrainer._eff_mem_befores.append(mem_before_mb)
        CustomSeq2SeqTrainer._eff_mem_afters.append(mem_after_mb)
        CustomSeq2SeqTrainer._eff_batch_sizes.append(batch_size)

        try:
            is_rank0 = getattr(self.accelerator, "process_index", 0) == 0
        except Exception:
            is_rank0 = True
        if not is_rank0:
            return

        n = len(CustomSeq2SeqTrainer._eff_step_times)
        if n % log_every != 0:
            return

        recent_times = CustomSeq2SeqTrainer._eff_step_times[-log_every:]
        recent_peaks = CustomSeq2SeqTrainer._eff_mem_peaks[-log_every:]
        recent_befores = CustomSeq2SeqTrainer._eff_mem_befores[-log_every:]
        recent_afters = CustomSeq2SeqTrainer._eff_mem_afters[-log_every:]
        recent_bs = CustomSeq2SeqTrainer._eff_batch_sizes[-log_every:]

        avg_time = sum(recent_times) / len(recent_times)
        avg_bs = sum(recent_bs) / len(recent_bs)
        per_sample_ms = avg_time / max(avg_bs, 1) * 1000
        avg_peak = sum(recent_peaks) / len(recent_peaks)
        avg_before = sum(recent_befores) / len(recent_befores)
        avg_after = sum(recent_afters) / len(recent_afters)
        setting = getattr(self.finetuning_args, "setting", "unknown")

        lines = [
            f"[Efficiency @ step {n}]  setting={setting}",
            f"  Avg step wall time    : {avg_time:.3f} s  (last {log_every} steps)",
            f"  Per-sample overhead   : {per_sample_ms:.2f} ms",
            f"  Avg batch size        : {avg_bs:.1f}",
            f"  GPU mem before step   : {avg_before:.1f} MB",
            f"  GPU mem after step    : {avg_after:.1f} MB",
            f"  GPU mem peak          : {avg_peak:.1f} MB",
        ]
        msg = "\n".join(lines)
        logger.info_rank0(msg)
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, "efficiency_stats.txt"), "a", encoding="utf-8") as f:
            print(msg, file=f)

    @override
    def training_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        num_items_in_batch: Optional[int] = None,
    ) -> "torch.Tensor":
        """Wrap the parent training_step to measure wall-clock time and GPU memory.

        Stats are accumulated and periodically flushed to efficiency_stats.txt so
        that TENT / SAR / TLM / NeFo (online_ttl) can be compared side-by-side.
        """
        # Log trainable parameter count once.
        self._log_efficiency_stats_once(model)

        # Snapshot memory and start timer.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / 1024 ** 2
        else:
            mem_before = 0.0
        t0 = time.perf_counter()

        # Delegate to parent (calls compute_loss internally).
        if num_items_in_batch is not None:
            result = super().training_step(model, inputs, num_items_in_batch)
        else:
            result = super().training_step(model, inputs)

        step_time = time.perf_counter() - t0
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024 ** 2
            mem_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        else:
            mem_after = mem_peak = 0.0

        batch_size = inputs.get("input_ids", inputs.get("input_ids_aug", torch.zeros(1))).shape[0]
        self._accumulate_and_log_efficiency(step_time, mem_before, mem_after, mem_peak, batch_size)

        return result

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"] if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            labels = labels.detach().clone() if labels is not None else None  # backup labels
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        # 🔧 CRITICAL FIX: For GeoChat/RS-LLaVA, rename pixel_values to images
        generation_inputs = {}
        is_custom_llava = False
        try:
            if self.model_args and hasattr(self.model_args, 'model_name_or_path'):
                model_path_str = str(self.model_args.model_name_or_path).lower()
                is_custom_llava = 'geochat' in model_path_str or 'rs-llava' in model_path_str
        except Exception:
            pass

        if is_custom_llava:
            for k, v in inputs.items():
                if k in ["question_id", "input_ids_aug"]:
                    continue
                if k == 'pixel_values':
                    # Expects 'images' parameter instead of 'pixel_values'
                    generation_inputs['images'] = v
                    logger.info_rank0(f"🖼️ Found pixel_values with shape: {v.shape if hasattr(v, 'shape') else 'unknown'}")
                else:
                    generation_inputs[k] = v
        else:
            generation_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}

        # Debug: check if images are present
        if 'images' not in generation_inputs and 'pixel_values' not in generation_inputs:
            logger.warning_rank0("⚠️ No images found in generation inputs!")

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, generation_inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _process_augmented_inputs(self, input_ids_aug, attention_mask):
        """
        处理增强数据中的special token，替换为pad token并更新attention mask

        Args:
            input_ids_aug: 包含special token的增强input_ids
            attention_mask: 原始attention mask

        Returns:
            tuple: (处理后的input_ids_aug, 处理后的attention_mask_aug)
        """
        special_token_id = self.tokenizer.convert_tokens_to_ids('<Neg_Mask>')
        pad_token_id = self.tokenizer.pad_token_id

        # 复制避免修改原始数据
        processed_input_ids = input_ids_aug.clone()
        processed_attention_mask = attention_mask.clone()

        # 找到special token位置并替换
        special_positions = (processed_input_ids == special_token_id)
        processed_input_ids[special_positions] = pad_token_id
        processed_attention_mask[special_positions] = 0

        return processed_input_ids, processed_attention_mask

    def _pad_tensors_to_target_len(self, src_tensor: "torch.Tensor", tgt_tensor: "torch.Tensor") -> "torch.Tensor":
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory

    def _decode_input_ids(self, input_ids: "torch.Tensor") -> List[str]:
        """Decode a batch of input_ids safely for debugging.

        - Filters out negative placeholder tokens (e.g., -200 for image slots) before decoding.
        - Keeps special tokens like <Neg_Mask> by setting skip_special_tokens=False so we can observe them.
        - Works on tensors located on any device.
        """
        if isinstance(input_ids, torch.Tensor):
            ids_list: List[List[int]] = input_ids.detach().to("cpu").tolist()
        else:
            ids_list = input_ids

        cleaned_ids_list: List[List[int]] = []
        for seq in ids_list:
            # 过滤所有负数占位符（如 -200）；保留 >=0 的真实词表token
            cleaned = [t for t in seq if isinstance(t, int) and t >= 0]
            cleaned_ids_list.append(cleaned)

        try:
            decoded: List[str] = self.tokenizer.batch_decode(
                cleaned_ids_list,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            # 回退逐条decode，避免某些tokenizer的batch行为异常
            decoded = []
            for seq in cleaned_ids_list:
                try:
                    decoded.append(self.tokenizer.decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=False))
                except Exception as e:  # 最坏情况下给出占位
                    decoded.append(f"<decode_error: {e}>")
        return decoded

    def _maybe_debug_print_inputs(self, inputs: Dict[str, Any], max_samples: int = 2, max_chars: int = 10000) -> None:
        """Conditionally print decoded inputs for debugging.

        Enable by setting environment variable TTL_DEBUG_INPUTS=1.
        Only prints on rank0 to avoid duplicate logs in DDP.
        """
        if os.environ.get("TTL_DEBUG_INPUTS", "0") != "1":
            return
        # 仅在主进程打印
        try:
            is_rank0 = getattr(self.accelerator, "process_index", 0) == 0
        except Exception:
            is_rank0 = True
        if not is_rank0:
            return

        try:
            if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                decoded_orig = self._decode_input_ids(inputs["input_ids"])[:max_samples]
                for i, s in enumerate(decoded_orig):
                    show = s if len(s) <= max_chars else s[:max_chars] + " ..."
                    logger.info_rank0(f"[TTL Debug] input_ids[{i}]: {show}")

            if "input_ids_aug" in inputs and isinstance(inputs["input_ids_aug"], torch.Tensor):
                decoded_aug = self._decode_input_ids(inputs["input_ids_aug"])[:max_samples]
                for i, s in enumerate(decoded_aug):
                    show = s if len(s) <= max_chars else s[:max_chars] + " ..."
                    logger.info_rank0(f"[TTL Debug] input_ids_aug[{i}]: {show}")
        except Exception as e:
            logger.warning_rank0(f"[TTL Debug] Failed to decode/print inputs: {e}")

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        🔧 Modified to support multi-GPU parallel saving with rank-specific files.
        """
        # 🔧 支持多GPU并行保存：每个GPU独立保存到不同文件
        if hasattr(self.accelerator, 'process_index'):
            rank = self.accelerator.process_index
            output_prediction_file = os.path.join(self.args.output_dir, f"generated_predictions_rank{rank}.jsonl")
        else:
            # 单GPU情况
            output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")

        logger.info(
            f"[GPU {getattr(self.accelerator, 'process_index', 0)}] Saving prediction results to {output_prediction_file}")

        labels = np.where(
            dataset["labels"] != IGNORE_INDEX, dataset["labels"], self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][: pad_len[0]]), axis=-1)

        # 一行代码过滤掉 -200 并解码
        decoded_inputs = self.tokenizer.batch_decode(
            [[token_id for token_id in input_ids if token_id != -200] for input_ids in dataset["input_ids"]],
            skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        with open(output_prediction_file, "a", encoding="utf-8") as writer:
            res: List[str] = []
            for i, (text, label, pred) in enumerate(zip(decoded_inputs, decoded_labels, decoded_preds)):
                question_id = dataset["question_id"][i]
                res.append(
                    json.dumps({"prompt": text, "question_id": question_id, "label": label, "answer": pred},
                               ensure_ascii=False))

            writer.write("\n".join(res) + "\n")

    
    @torch.no_grad()
    def cal_ce(self, logits, labels):
        """
        计算 cross entropy
        """
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits: "torch.Tensor" = logits[..., :-1, :]
        shift_labels: "torch.Tensor" = labels[..., 1:]
        

        loss_mask = shift_labels != IGNORE_INDEX
        flatten_logits = shift_logits.contiguous().view(-1, shift_logits.size(-1))
        flatten_labels = shift_labels.contiguous().view(-1)
        token_logps: "torch.Tensor" = criterion(flatten_logits, flatten_labels) # [bs*seq_len]
        token_logps = token_logps.contiguous().view(shift_logits.size(0), -1)  # [bs, seq_len]
        
        # Handle cases where there are no valid tokens
        valid_tokens = loss_mask.sum(-1)
        sentence_logps_normal = torch.zeros_like(valid_tokens, dtype=logits.dtype)
        for i in range(len(valid_tokens)):
            if valid_tokens[i] > 0:
                sentence_logps_normal[i] = (token_logps[i] * loss_mask[i]).sum() / valid_tokens[i]
        
        return sentence_logps_normal
    

    def cal_kl(self, logits, labels):
        loss_fct = nn.KLDivLoss(reduction='batchmean') 

        shift_logits: "torch.Tensor" = logits[..., :-1, :]
        shift_labels: "torch.Tensor" = labels[..., 1:]

        sentence_kl = torch.zeros(shift_logits.size(0), device=shift_logits.device)  # [bs]
        for i, (shift_logit, shift_label) in enumerate(zip(shift_logits, shift_labels)):
            mask = shift_label != IGNORE_INDEX
            shift_logit, shift_label = shift_logit[mask], shift_label[mask]
            if len(shift_label) == 0:  # Skip if no valid tokens
                continue
            log_probs = shift_logit.log_softmax(dim=-1)
            one_hot_targets = torch.zeros_like(log_probs).scatter_(1, shift_label.unsqueeze(1), 1).to(log_probs.device)
            sentence_kl[i] = loss_fct(log_probs, one_hot_targets) 

        return sentence_kl

    def _fix_multimodal_labels(self, logits: torch.Tensor, inputs: dict) -> dict:
        """
        TTL多模态修复：将inputs["labels"]中的-200替换为正确数量的-100
        
        Args:
            logits: 模型输出的logits，用于确定实际序列长度
            inputs: 输入字典，包含labels等
        
        Returns:
            修正后的inputs字典
        """
        if not torch.any(inputs["labels"] == -200):
            return inputs
            
        actual_seq_len = logits.shape[1]
        original_seq_len = inputs["labels"].shape[1]
        
        if actual_seq_len == original_seq_len:
            return inputs
            
        # 替换labels中的-200为相应数量的-100 (最初的简单版本，修正计算)
        # 每个-200需要被替换为多少个-100：
        # 总扩展量 + 1 (因为-200本身也要被替换)
        image_token_expansion = (actual_seq_len - original_seq_len) + 1  # 1295 + 1 = 1296
        
        new_labels_list = []
        for batch_idx in range(inputs["labels"].shape[0]):
            old_labels = inputs["labels"][batch_idx].tolist()
            new_labels = []
            
            for token in old_labels:
                if token == -200:  # 图像token位置
                    new_labels.extend([-100] * image_token_expansion)  # 替换为expansion个-100
                else:
                    new_labels.append(token)
            
            new_labels_list.append(new_labels)
        
        # 转换为tensor
        inputs["labels"] = torch.tensor(new_labels_list, dtype=inputs["labels"].dtype, device=inputs["labels"].device)
        return inputs

    def log_to_file(self, sentence_ce, sentence_kl, mask, coeff, loss, base_kl_loss=None):
        """
        Log the sentence cross-entropy and KL divergence to a file.
        """
        with open(f"{self.args.output_dir}/logfile.txt", 'a', encoding="utf-8") as f:
            # 🔧 处理online_ttl的None参数情况
            if mask is None or coeff is None:  # online_ttl logging path
                if base_kl_loss is not None:
                    print(f"Online TTL - KL prcp loss: {sentence_ce}, Aug entropy loss: {sentence_kl}, Base KL loss: {base_kl_loss}, Total loss: {loss}", file=f)
                else:
                    print(f"Online TTL - KL prcp loss: {sentence_ce}, Aug entropy loss: {sentence_kl}, Total loss: {loss}", file=f)
            else:
                for ce, kl, m, coef in zip(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff):
                    if m:
                        print(f"This sample is selected. Threshold: {self.finetuning_args.threshold}, Cross-entropy: {ce}, KL divergence: {kl}, Weight coefficient: {coef}, Final loss: {loss}", file=f)
                    else:
                        print(f"This sample is discarded. Threshold: {self.finetuning_args.threshold}, Cross-entropy: {ce}, KL divergence: {kl}", file=f)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        assert self.tokenizer.padding_side == 'right', "Training should be done with right padding."

        if "labels" in inputs:
            labels = inputs["labels"].clone()
            # 🔧 修复：只将-200之前的token设为-100，保留-200供后续处理
            is_neg200 = (labels == -200)
            # 为每个样本找到第一个-200的位置
            batch_size, seq_len = labels.shape
            device = labels.device
            # 找到每行第一个-200的位置（简化版本，假设每行都有-200）
            first_neg200_pos = is_neg200.float().argmax(dim=1)
            
            # 创建mask：只覆盖-200之前的位置（不包括-200本身）
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            mask = positions < first_neg200_pos.unsqueeze(1)
            # 应用mask：只将-200之前的token设为-100
            labels[mask] = -100
            inputs["labels"] = labels

        # For vision-language models, ONLY GeoChat/RS-LLaVA expects 'images' instead of 'pixel_values'
        # Keep Qwen2.5-VL and others using 'pixel_values' unchanged
        try:
            is_custom_llava = False
            if self.model_args and hasattr(self.model_args, 'model_name_or_path'):
                model_path_str = str(self.model_args.model_name_or_path).lower()
                is_custom_llava = 'geochat' in model_path_str or 'rs-llava' in model_path_str
        except Exception:
            is_custom_llava = False

        if is_custom_llava and 'pixel_values' in inputs:
            inputs = inputs.copy()  # Don't modify original inputs
            inputs['images'] = inputs.pop('pixel_values')
             
        # Use is_custom_llava for generation logic check later if needed
        is_geochat = is_custom_llava 
        
        if self.finetuning_args.setting == "offline_ttl":
            model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
            # 1. In offline setting, perform a forward pass using the base model to get logits
            with torch.no_grad(): 
                model.eval()
                with self.accelerator.unwrap_model(model).disable_adapter():
                    pretrain_logits = model(**model_inputs).logits
                    inputs = self._fix_multimodal_labels(pretrain_logits, inputs)
                    sentence_ce = self.cal_ce(pretrain_logits, inputs["labels"])

            # 2. Filter samples based on cross-entropy (equivalent to KL divergence), keep those above threshold, and calculate weighting coefficients
            mask = sentence_ce > self.finetuning_args.threshold 
            coeff = self.finetuning_args.lamb * torch.exp(sentence_ce.clone().detach() - self.finetuning_args.threshold) # [bs,]
            
            model.train() # Resume training mode
            outputs = model(**model_inputs)  # Forward pass to get logits

            # 3. Calculate KL divergence
            sentence_kl = self.cal_kl(outputs.logits, inputs["labels"])
            
            # 4. Compute total loss
            sentence_kl = sentence_kl.mul(coeff).mul(mask)  # [bs,]
            if mask.sum() == 0:
                total_loss = sentence_kl.mean()
            else:
                total_loss = sentence_kl.sum() / mask.sum()

            self.log_to_file(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff, total_loss.item())

        elif self.finetuning_args.setting == "online_ttl_TLM":
            # 1. Perform a forward pass using the model being trained to get logits
            # 过滤掉不需要的字段，避免传递给模型
            model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
            outputs = model(**model_inputs)

            # 🔑 TTL多模态修复：调整labels长度
            inputs = self._fix_multimodal_labels(outputs.logits, inputs)

            # 2. Filter samples based on cross-entropy; in online setting, CE is calculated from current model
            sentence_ce = self.cal_ce(outputs.logits, inputs["labels"])  # [bs,]
            mask = sentence_ce > self.finetuning_args.threshold  # Keep samples above threshold
            # 3. Calculate KL divergence
            sentence_kl = self.cal_kl(outputs.logits, inputs["labels"])  # [bs,]
            coeff = self.finetuning_args.lamb * torch.exp(
                sentence_ce.clone().detach() - self.finetuning_args.threshold)  # [bs,]

            # 4. Compute total loss
            sentence_kl = sentence_kl.mul(coeff).mul(mask)  # [bs,]

            if mask.sum() == 0:
                total_loss = sentence_kl.mean()
            else:
                total_loss = sentence_kl.sum() / mask.sum()

            self.log_to_file(sentence_ce.clone().detach(), sentence_kl.clone().detach(), mask, coeff, total_loss.item())

            if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
                # other model should not scale the loss
                if return_outputs:
                    return (total_loss / self.args.gradient_accumulation_steps, outputs)
                else:
                    return total_loss / self.args.gradient_accumulation_steps

            return (total_loss, outputs) if return_outputs else total_loss

        elif self.finetuning_args.setting == "online_ttl":
            self._maybe_debug_print_inputs(inputs)

            use_sft_loss = getattr(self.finetuning_args, "use_sft_loss", False)
            sft_loss_weight = getattr(self.finetuning_args, "sft_loss_weight", 1.0)

            # 🔑 优化：如果input_ids和input_ids_aug完全相同（无否定词），直接返回零loss
            # 避免浪费计算资源在generate和KL计算上
            is_same = torch.equal(inputs["input_ids"], inputs["input_ids_aug"])
            
            if is_same:
                # 没有否定词，跳过TTL训练
                if use_sft_loss:
                    # 即使无否定词，仍可用ground_truth（已编码在labels中）做SFT监督
                    # inputs["input_ids"]  = 原始否定问句 + 回答（ground_truth 来自 messages 中 assistant 的内容）
                    # inputs["input_ids_aug"] 被排除，不参与SFT
                    # labels = input_ids（自监督），经顶部预处理后 prompt 位置为 -100，答案位置为真实 token
                    # -200 是图像占位符，CE loss 不能接受，必须替换为 -100
                    sft_model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
                    sft_labels = sft_model_inputs["labels"].clone()
                    sft_labels[sft_labels == -200] = -100  # 图像位置不参与文字CE loss
                    sft_model_inputs = {**sft_model_inputs, "labels": sft_labels}
                    sft_outputs = model(**sft_model_inputs)
                    sft_loss = sft_outputs.loss
                    print(f'##########################################sft_loss (is_same): {sft_loss}')
                    return sft_loss_weight * sft_loss
                # 使用模型参数创建一个有梯度的零loss（确保可以backward）
                dummy_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
                return dummy_loss
            
            # 1. 对 Neg Token 进行 Mask，得到 reponse
            # 处理增强数据中的special token，即<Neg_Mask>
            input_ids_aug, attention_mask_aug = self._process_augmented_inputs(
                inputs["input_ids_aug"], inputs["attention_mask"]
            )

            # 🔧 修复DDP冲突：安全地访问DDP模型进行生成
            if is_geochat:
                images = inputs.get("images")
            else:
                images = inputs.get("pixel_values", None)
            with torch.no_grad():
                generation_model = model.module if hasattr(model, "module") else model
                generation_model.eval()
                if images is not None:
                    if is_geochat:
                        generated_sequences = generation_model.generate(
                            input_ids=input_ids_aug,
                            attention_mask=attention_mask_aug,
                            images=images,
                            **gen_kwargs
                        )
                    else:
                        generated_sequences = generation_model.generate(
                            input_ids=input_ids_aug,
                            attention_mask=attention_mask_aug,
                            pixel_values=images,
                            image_grid_thw=inputs.get("image_grid_thw", None),
                            **gen_kwargs
                        )
                else:
                    generated_sequences = generation_model.generate(
                        input_ids=input_ids_aug,
                        attention_mask=attention_mask_aug,
                        **gen_kwargs
                    )
                generation_model.train()

             # Also generate with base model (adapters disabled), keep as generated_sequences_base
            generated_sequences_base = None
            try:
                wrapped_model = model.module if hasattr(model, "module") else model
                base_model = self.accelerator.unwrap_model(wrapped_model)
                disable_ctx = getattr(base_model, "disable_adapter", None)
                ctx = disable_ctx() if callable(disable_ctx) else contextlib.nullcontext()
                with torch.no_grad():
                    with ctx:
                        base_model.eval()
                        if images is not None:
                            if is_geochat:
                                generated_sequences_base = base_model.generate(
                                    input_ids=input_ids_aug,
                                    attention_mask=attention_mask_aug,
                                    images=images,
                                    **gen_kwargs
                                )
                            else:
                                generated_sequences_base = base_model.generate(
                                    input_ids=input_ids_aug,
                                    attention_mask=attention_mask_aug,
                                    pixel_values=images,
                                    image_grid_thw=inputs.get("image_grid_thw", None),
                                    **gen_kwargs
                                )
                        else:
                            generated_sequences_base = base_model.generate(
                                input_ids=input_ids_aug,
                                attention_mask=attention_mask_aug,
                                **gen_kwargs
                            )

                        ans_len_start = input_ids_aug.shape[1]
                        base_answer_tokens = generated_sequences_base[:, ans_len_start:]  # 提取答案部分
                        # 为 base 生成的完整序列构造匹配长度的 attention_mask
                        base_answer_len = base_answer_tokens.shape[1]
                        base_answer_mask = torch.ones(
                            generated_sequences_base.shape[0], base_answer_len,
                            device=attention_mask_aug.device, dtype=attention_mask_aug.dtype
                        )
                        extended_attention_mask_base = torch.cat([attention_mask_aug, base_answer_mask], dim=1)

                        if is_geochat:
                            base_outputs = base_model(
                                input_ids=generated_sequences_base,  # augmented prompt + base 生成的答案
                                attention_mask=extended_attention_mask_base,
                                images=images,
                            )
                        else:
                            base_outputs = base_model(
                                input_ids=generated_sequences_base,  # augmented prompt + base 生成的答案
                                attention_mask=extended_attention_mask_base,
                                pixel_values=images,
                                image_grid_thw=inputs.get("image_grid_thw", None)
                            )
                        base_aug_log_probs = self.compute_log_prob(base_outputs.logits, base_answer_tokens)
            except Exception as e:
                logger.warning_rank0(f"[TTL] Base-model generation failed (kept LoRA generation only): {e}")

            # Optional debug: compare both generations' answer parts
            try:
                if os.environ.get("TTL_DEBUG_INPUTS", "0") == "1":
                    is_rank0 = getattr(self.accelerator, "process_index", 0) == 0
                    if is_rank0:
                        ans_len_start = input_ids_aug.shape[1]
                        # LoRA-enabled answer
                        ans_lora = generated_sequences[:, ans_len_start:]
                        decoded_lora = self._decode_input_ids(ans_lora)[:2]
                        for i, s in enumerate(decoded_lora):
                            show = s if len(s) <= 300 else s[:300] + " ..."
                            logger.info_rank0(f"[TTL Debug] answer_tokens_lora[{i}]: {show}")
                        # Base-model answer (if available)
                        if generated_sequences_base is not None:
                            ans_base = generated_sequences_base[:, ans_len_start:]
                            decoded_base = self._decode_input_ids(ans_base)[:2]
                            for i, s in enumerate(decoded_base):
                                show = s if len(s) <= 300 else s[:300] + " ..."
                                logger.info_rank0(f"[TTL Debug] answer_tokens_base[{i}]: {show}")
            except Exception as e:
                logger.warning_rank0(f"[TTL Debug] Failed to decode/compare generations: {e}")
            
            # 简单拼接：原始问题 + 预测答案
            question_len = input_ids_aug.shape[1]
            answer_tokens = generated_sequences[:, question_len:]  # 提取答案部分
            complete_input_ids = torch.cat([inputs["input_ids"], answer_tokens], dim=1)  # 拼接
            
            # 为拼接后的序列构造对应的attention_mask
            answer_len = answer_tokens.shape[1]
            
            # generated_sequences的attention_mask (原question_mask + answer_mask)
            answer_mask_for_generated = torch.ones(
                generated_sequences.shape[0], answer_len, 
                device=attention_mask_aug.device, dtype=attention_mask_aug.dtype
            )
            extended_attention_mask_aug = torch.cat([attention_mask_aug, answer_mask_for_generated], dim=1)
            extended_attention_mask_complete = torch.cat([inputs["attention_mask"], answer_mask_for_generated], dim=1)
             #attention_mask_aug中为0的idx的 后几个token位置 和图像的相似度
            # 🔧 修复DDP问题：合并两次forward为一次，避免"marked as ready twice"错误
            combined_input_ids = torch.cat([generated_sequences, complete_input_ids], dim=0)  # [2, seq_len]
            combined_attention_mask = torch.cat([extended_attention_mask_aug, extended_attention_mask_complete], dim=0)  # [2, seq_len]
            combined_images = torch.cat([images, images], dim=0) if images is not None else None
            if not is_geochat:
                image_grid_thw=inputs.get("image_grid_thw", None)
                combined_image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0) if image_grid_thw is not None else None
            
            # 一次forward处理两个输入
            if combined_images is not None:
                if is_geochat:
                    combined_outputs = model(
                        input_ids=combined_input_ids,
                        attention_mask=combined_attention_mask,
                        images=combined_images,
                    )
                else:
                    combined_outputs = model(
                        input_ids=combined_input_ids,
                        attention_mask=combined_attention_mask,
                        pixel_values=combined_images,
                        image_grid_thw=combined_image_grid_thw,
                    )
            else:
                combined_outputs = model(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                )

            # 分离两个样本的logits进行计算
            aug_log_probs = self.compute_log_prob(combined_outputs.logits[0:1], answer_tokens)  # 第1个样本(augmented)
            log_probs = self.compute_log_prob(combined_outputs.logits[1:2], answer_tokens)      # 第2个样本(original)

            # 检查 shape 是否一致，不一致则置为 None
            if base_aug_log_probs is not None and aug_log_probs.shape != base_aug_log_probs.shape:
                logger.warning_rank0(f"[TTL] Shape mismatch: aug_log_probs {aug_log_probs.shape} vs base_aug_log_probs {base_aug_log_probs.shape}, setting base_aug_log_probs to None")
                base_aug_log_probs = None

            base_kl_loss = None
            if base_aug_log_probs is not None:
                # 计算kl散度
                # 使用封装的 KL 计算（近似，数值稳定且非负）：D_KL(ref || cur)
                base_kld = self.compute_kl(
                    log_probs=aug_log_probs,
                    ref_log_probs=base_aug_log_probs.detach(),
                    kl_penalty='low_var_kl',
                )
                # base_kl_loss = self.average_loss(base_kld, base_answer_mask, mode='seq')
                base_kl_loss = base_kld.mean()
                # base_kl_loss = torch.clamp(base_kl_loss, max=0.2)
                print('##########################################base_kl_prcp_loss:', base_kl_loss)

            # compute kl_prcp
            aug_kld = self.compute_kl(
                log_probs=log_probs,
                ref_log_probs=aug_log_probs.detach(),
                kl_penalty='low_var_kl',
            )
            # aug_kld = self.kl_aug(ans_logits_aug=aug_log_probs.detach(), ans_logits_cur=log_probs)
            # aug_kld = torch.clamp(aug_kld, min=0, max=0.2)
            # aug_kld = torch.clamp(aug_kld, min=0)
            # kl_prcp_loss = self.average_loss(aug_kld, answer_mask_for_generated, mode='seq')
            kl_prcp_loss = aug_kld.mean()
            kl_prcp_loss = torch.clamp(kl_prcp_loss, max=0.4)

            print('##########################################kl_prcp_loss:', kl_prcp_loss)
            aug_log_probs_full = self.compute_log_prob(combined_outputs.logits[0:1], answer_tokens, drop_last_token=False)
            aug_entropy_loss = -self.masked_mean(aug_log_probs_full, answer_mask_for_generated)
            aug_entropy_weight = getattr(self.finetuning_args, "aug_entropy_weight", 0.3)
            total_loss = -kl_prcp_loss + aug_entropy_loss * aug_entropy_weight
            # # only for test
            # total_loss = aug_entropy_loss * 0.2
            if base_kl_loss is not None:
                total_loss = total_loss + base_kl_loss

            if use_sft_loss:
                # inputs["input_ids"]  = 原始否定问句 + 回答（ground_truth 来自 messages 中 assistant 的内容）
                # inputs["input_ids_aug"] 被排除，不参与SFT
                # labels 经顶部预处理后：prompt → -100，图像占位符 → -200，答案 → 真实 token
                # 将 -200 替换为 -100，确保 CE loss 只计算答案 token
                sft_model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
                sft_labels = sft_model_inputs["labels"].clone()
                sft_labels[sft_labels == -200] = -100  # 图像位置不参与文字CE loss
                sft_model_inputs = {**sft_model_inputs, "labels": sft_labels}
                sft_outputs = model(**sft_model_inputs)
                sft_loss = sft_outputs.loss
                print(f'##########################################sft_loss: {sft_loss}')
                total_loss = total_loss + sft_loss_weight * sft_loss

            # 🔧 简单方案：为online_ttl传入None，跳过详细日志
            self.log_to_file(
                kl_prcp_loss.clone().detach(),
                aug_entropy_loss.clone().detach(),
                None,
                None,
                total_loss.item(),
                base_kl_loss=base_kl_loss.clone().detach() if base_kl_loss is not None else None,
            )
        
        elif self.finetuning_args.setting == "tent":
            """
            TENT for MLLM: Negative Log-Likelihood on Model's Own Predictions
            自训练版本：让模型对自己的预测更有信心
            
            核心思想：
            1. 模型预测最可能的 token (argmax)
            2. 计算这些预测 token 的负对数似然 (NLL)
            3. 最小化 NLL，增强模型对自己预测的置信度
            """
            # 1. 准备模型输入（移除不需要的字段）
            model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
            
            # 2. 前向传播获取 logits
            outputs = model(**model_inputs)
            logits = outputs.logits  # [bs, seq_len, vocab_size]
            
            # 🔑 TTL多模态修复：调整labels长度
            inputs = self._fix_multimodal_labels(logits, inputs)
            
            # 3. 获取模型的预测（argmax）
            # 使用 detach() 避免影响梯度流
            with torch.no_grad():
                pred_tokens = logits.argmax(dim=-1)  # [bs, seq_len] - 模型预测的 token
            
            # 4. 计算预测 token 的 log 概率
            log_probs = F.log_softmax(logits, dim=-1)  # [bs, seq_len, vocab_size]
            
            # 使用 gather 提取预测 token 的 log 概率
            # pred_tokens.unsqueeze(-1): [bs, seq_len, 1]
            pred_log_probs = log_probs.gather(
                dim=-1, 
                index=pred_tokens.unsqueeze(-1)
            ).squeeze(-1)  # [bs, seq_len]
            
            # 5. 计算 NLL：-log p(predicted_token)
            nll = -pred_log_probs  # [bs, seq_len]
            
            # 6. 只计算答案部分的 NLL（使用 labels mask）
            if "labels" in inputs:
                labels = inputs["labels"]
                answer_mask = (labels != -100).float()  # [bs, seq_len]
                
                # 使用 masked_mean 计算答案部分的平均 NLL
                total_loss = self.masked_mean(nll, answer_mask)  # 标量
                
                # 计算每个样本的平均 NLL（用于日志）
                sample_nll = self.masked_mean(nll, answer_mask, dim=1)  # [bs,]
            else:
                # 如果没有 labels，计算所有位置的平均 NLL
                total_loss = nll.mean()
                sample_nll = nll.mean(dim=1)  # [bs,]
            
            # 7. 记录日志
            batch_size = logits.size(0)
            dummy_mask = torch.ones(batch_size, device=logits.device)
            dummy_coeff = torch.ones(batch_size, device=logits.device)
            self.log_to_file(
                sample_nll.clone().detach(),  # 用 NLL 代替 CE
                sample_nll.clone().detach(),  # 用 NLL 代替 KL
                dummy_mask, 
                dummy_coeff, 
                total_loss.item()
            )
        
        elif self.finetuning_args.setting == "sar":
            """
            SAR: Sharpness-Aware and Reliable Entropy Minimization
            可靠熵最小化：只对高置信度（低熵）样本进行训练
            
            核心思想（ICLR 2023）：
            1. 计算模型输出的熵（衡量不确定性）
            2. 基于熵阈值过滤样本（只保留可靠样本）
            3. 只对可靠样本最小化熵（避免强化错误预测）
            
            与 TENT 的区别：
            - TENT: 对所有样本都训练（可能强化错误）
            - SAR: 只对高置信度样本训练（更稳定）
            """
            # 1. 准备模型输入（移除不需要的字段）
            model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
            
            # 2. 前向传播获取 logits
            outputs = model(**model_inputs)
            logits = outputs.logits  # [bs, seq_len, vocab_size]
            
            # 🔑 TTL多模态修复：调整labels长度
            inputs = self._fix_multimodal_labels(logits, inputs)
            
            # 3. 计算熵：H = -Σ p*log(p)
            log_probs = F.log_softmax(logits, dim=-1)  # [bs, seq_len, vocab_size]
            probs = torch.exp(log_probs)  # [bs, seq_len, vocab_size]
            
            # 计算每个位置的熵
            entropy = -(probs * log_probs).sum(dim=-1)  # [bs, seq_len]
            
            # 4. 计算答案部分的平均熵（用于可靠性判断）
            if "labels" in inputs:
                labels = inputs["labels"]
                answer_mask = (labels != -100).float()  # [bs, seq_len]
                
                # 每个样本的平均熵
                sample_entropy = self.masked_mean(entropy, answer_mask, dim=1)  # [bs,]
            else:
                # 如果没有 labels，计算所有位置的平均熵
                sample_entropy = entropy.mean(dim=1)  # [bs,]
            
            # 5. 可靠性过滤（SAR 的核心）
            # 熵阈值：只保留低熵（高置信度）样本
            vocab_size = logits.size(-1)
            
            # SAR 论文推荐：threshold = 0.4 * ln(vocab_size)
            entropy_threshold = 0.4 * torch.log(torch.tensor(vocab_size, dtype=torch.float32))
            
            # 可靠性 mask：entropy < threshold 的样本
            reliable_mask = (sample_entropy < entropy_threshold).float()  # [bs,]
            
            # 6. 计算损失（只对可靠样本）
            if reliable_mask.sum() > 0:
                # 对可靠样本最小化熵
                if "labels" in inputs:
                    # Token 级熵 [bs, seq_len]
                    # 只计算可靠样本的答案部分
                    reliable_mask_expanded = reliable_mask.unsqueeze(1)  # [bs, 1]
                    combined_mask = answer_mask * reliable_mask_expanded  # [bs, seq_len]
                    
                    # 使用 masked_mean 计算可靠样本的平均熵
                    total_loss = self.masked_mean(entropy, combined_mask)
                else:
                    # 没有 labels 的情况
                    total_loss = (sample_entropy * reliable_mask).sum() / reliable_mask.sum()
            else:
                # 所有样本都不可靠，返回零损失（不更新模型）
                # 使用模型参数创建一个有梯度的零loss
                total_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
            
            # 7. 记录日志
            batch_size = logits.size(0)
            dummy_coeff = torch.ones(batch_size, device=logits.device)
            self.log_to_file(
                sample_entropy.clone().detach(),  # 用熵代替 CE
                sample_entropy.clone().detach(),  # 用熵代替 KL
                reliable_mask.clone().detach(),   # 记录哪些样本是可靠的
                dummy_coeff, 
                total_loss.item()
            )
        
        elif self.finetuning_args.setting == "eata":
            """
            EATA: Efficient Anti-forgetting Test-Time Adaptation
            防遗忘的测试时适应
            
            核心思想（NeurIPS 2022）：
            1. 样本选择（基于熵阈值）- 只训练可靠样本
            2. L2 正则化（防止遗忘）- 约束参数不要偏离初始值太远
            
            完整版 EATA 使用 Fisher 信息矩阵，但需要在训练集上预计算
            这里实现简化版：使用 L2 正则化代替 Fisher 正则化
            
            Loss = entropy_loss + λ * L2_regularization
            """
            # 1. 准备模型输入（移除不需要的字段）
            model_inputs = {k: v for k, v in inputs.items() if k not in ["question_id", "input_ids_aug"]}
            
            # 2. 前向传播获取 logits
            outputs = model(**model_inputs)
            logits = outputs.logits  # [bs, seq_len, vocab_size]
            
            # 🔑 TTL多模态修复：调整labels长度
            inputs = self._fix_multimodal_labels(logits, inputs)
            
            # 3. 计算熵：H = -Σ p*log(p)
            log_probs = F.log_softmax(logits, dim=-1)  # [bs, seq_len, vocab_size]
            probs = torch.exp(log_probs)  # [bs, seq_len, vocab_size]
            
            # 计算每个位置的熵
            entropy = -(probs * log_probs).sum(dim=-1)  # [bs, seq_len]
            
            # 4. 计算答案部分的平均熵（用于可靠性判断）
            if "labels" in inputs:
                labels = inputs["labels"]
                answer_mask = (labels != -100).float()  # [bs, seq_len]
                
                # 每个样本的平均熵
                sample_entropy = self.masked_mean(entropy, answer_mask, dim=1)  # [bs,]
            else:
                # 如果没有 labels，计算所有位置的平均熵
                sample_entropy = entropy.mean(dim=1)  # [bs,]
            
            # 5. 可靠性过滤（与 SAR 相同）
            vocab_size = logits.size(-1)
            entropy_threshold = 0.4 * torch.log(torch.tensor(vocab_size, dtype=torch.float32))
            reliable_mask = (sample_entropy < entropy_threshold).float()  # [bs,]
            
            # 6. 计算熵损失（只对可靠样本）
            if reliable_mask.sum() > 0:
                if "labels" in inputs:
                    # Token 级熵 [bs, seq_len]
                    reliable_mask_expanded = reliable_mask.unsqueeze(1)  # [bs, 1]
                    combined_mask = answer_mask * reliable_mask_expanded  # [bs, seq_len]
                    entropy_loss = self.masked_mean(entropy, combined_mask)
                else:
                    entropy_loss = (sample_entropy * reliable_mask).sum() / reliable_mask.sum()
            else:
                # 所有样本都不可靠
                entropy_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
            
            # 7. L2 正则化（防遗忘）- EATA 的核心
            # 惩罚参数偏离初始值太远
            l2_reg = 0.0
            
            # 获取或初始化初始参数
            if not hasattr(self, '_eata_initial_params'):
                # 第一次调用：保存当前参数作为初始值
                self._eata_initial_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self._eata_initial_params[name] = param.data.clone().detach()
            
            # 计算 L2 正则化：||θ_current - θ_init||^2
            for name, param in model.named_parameters():
                if param.requires_grad and name in self._eata_initial_params:
                    param_diff = param - self._eata_initial_params[name]
                    l2_reg += (param_diff ** 2).sum()
            
            # 8. 总损失 = 熵损失 + λ × L2正则化
            # λ 控制防遗忘的强度，从 finetuning_args 获取，默认 1.0
            lambda_reg = getattr(self.finetuning_args, 'eata_reg_coeff', 1.0)
            total_loss = entropy_loss + lambda_reg * l2_reg
            
            # 9. 记录日志
            batch_size = logits.size(0)
            dummy_coeff = torch.ones(batch_size, device=logits.device)
            self.log_to_file(
                sample_entropy.clone().detach(),  # 用熵代替 CE
                sample_entropy.clone().detach(),  # 用熵代替 KL
                reliable_mask.clone().detach(),   # 记录可靠样本
                dummy_coeff, 
                total_loss.item()
            )
        
        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return(total_loss / self.args.gradient_accumulation_steps, outputs)
            else:
                return total_loss / self.args.gradient_accumulation_steps
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def kl_aug(self, ans_logits_aug, ans_logits_cur):  
         # D_KL(Pref || Q) = sum_V P * (logP - logQ)
        per_token_kl = (ans_logits_aug.exp() * (ans_logits_aug - ans_logits_cur)).sum(dim=-1)  # [1, L]

        return per_token_kl  # [1, L]

    def compute_log_prob(self, logits, answer_ids, drop_last_token=True):
        """
        计算答案部分特定token的log概率

        Args:
            logits: torch.Tensor, shape [1, seq_len, vocab_size] - 模型输出的logits
            answer_ids: torch.Tensor or list - 具体的answer token IDs

        Returns:
            log_probs: torch.Tensor, shape [1, len(answer_ids)] - 每个answer token的log概率
        """
        # 处理answer_ids的维度并获取长度
        if isinstance(answer_ids, list):
            answer_ids = torch.tensor(answer_ids, device=logits.device)

        if answer_ids.dim() == 1:
            answer_ids = answer_ids.unsqueeze(0)  # [1, answer_len]

        answer_len = answer_ids.size(1)  # 自动获取answer长度

        if drop_last_token:
            answer_logits = logits[:, -(answer_len + 1):-2, :]

            log_probs_all = F.log_softmax(answer_logits, dim=-1)

            log_probs = log_probs_all.gather(
                dim=-1,
                index=answer_ids[:, :-1].unsqueeze(-1)  # [1, answer_len, 1]
            ).squeeze(-1)  # [1, answer_len]
        else:
            # 获取用于预测answer tokens的logits位置
            answer_logits = logits[:, -(answer_len + 1):-1, :]  # [1, answer_len, vocab_size]

            # 转换为log概率分布
            log_probs_all = F.log_softmax(answer_logits, dim=-1)  # [1, answer_len, vocab_size]

            # 使用gather提取特定token的log概率
            log_probs = log_probs_all.gather(
                dim=-1,
                index=answer_ids.unsqueeze(-1)  # [1, answer_len, 1]
            ).squeeze(-1)  # [1, answer_len]

        return log_probs

    def compute_kl(self,
            log_probs: torch.FloatTensor,
            ref_log_probs: torch.FloatTensor,
            kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"],
    ) -> torch.Tensor:
        """Compute KL divergence given log_probs and ref_log_probs.

        Adapted from https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L1150

        Args:
            log_probs: torch.Tensor
            ref_log_probs: torch.Tensor
            kl_penalty: str ("kl", "abs", "mse", "low_var_kl", "full")

        Returns:
            kl_div: torch.Tensor

        """
        log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()

        # if log_probs.shape != ref_log_probs.shape:
        #     return torch.tensor(0.0, device=log_probs.device)

        if kl_penalty == "kl":
            return log_probs - ref_log_probs

        if kl_penalty == "abs":
            return (log_probs - ref_log_probs).abs()

        if kl_penalty == "mse":
            return 0.5 * (log_probs - ref_log_probs).square()

        # J. Schulman. Approximating kl divergence, 2020.
        # URL http://joschu.net/blog/kl-approx.html
        if kl_penalty == "low_var_kl":
            # For numerical stability
            kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
            kld = (kl.exp() - kl - 1).contiguous()
            return torch.clamp(kld, min=-10.0, max=10.0)

        if kl_penalty == "full":
            return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    def average_loss(self,
            values: torch.Tensor, mask: torch.Tensor, mode: Literal["token", "seq"], eps: float = 1e-8
    ) -> torch.Tensor:
        """Average the policy loss.

        Args:
            values: `(torch.Tensor)`
                shape: (bs, response_length)
            mask: `(torch.Tensor)`
                shape: (bs, response_length)
            mode: `(Literal["token", "seq"])`
                "token": average the loss in the whole batch
                "seq": average the loss in each sequence then average the mean of the means
            eps: `(float)`
                epsilon value

        Returns:
            loss: `a scalar torch.Tensor`
        """
        if mode == "token":
            return self.masked_mean(values, mask, eps=eps)
        elif mode == "seq":
            return ((values * mask).sum(-1) / (mask.sum(-1) + eps)).mean()
        else:
            raise NotImplementedError(f"Unknown mode: {mode}.")

    def masked_mean(self, values: torch.Tensor, mask: torch.Tensor, dim: int = None, eps: float = 1e-8) -> torch.Tensor:
        """Compute mean of tensor with a masked values."""
        return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)

    def _prepare_model_inputs(self, inputs, model):
        """Prepare inputs for GeoChat model compatibility"""
        
        # Check if this is a GeoChat model by looking at the model class name
        model_class_name = model.__class__.__name__ if hasattr(model, '__class__') else str(type(model))
        
        # Unwrap the model to find the actual GeoChat model
        actual_model = model
        
        # Handle DistributedDataParallel wrapping
        if hasattr(model, 'module'):
            actual_model = model.module
        
        # Handle PEFT models - get the base model
        if hasattr(actual_model, 'base_model') and hasattr(actual_model.base_model, 'model'):
            base_model = actual_model.base_model.model
            base_model_class_name = base_model.__class__.__name__
        else:
            base_model = actual_model
            base_model_class_name = actual_model.__class__.__name__
        
        # If it's a GeoChat model, handle multimodal inputs properly
        if 'GeoChat' in base_model_class_name or 'GeoChat' in actual_model.__class__.__name__:
            logging.get_logger(__name__).debug(f"Detected GeoChat model: {base_model_class_name}, processing multimodal inputs")
            
            # Extract images from pixel_values
            images = inputs.get('pixel_values', None)
            
            # Prepare inputs and labels for multimodal processing
            if hasattr(base_model, 'prepare_inputs_labels_for_multimodal') and images is not None:
                input_ids = inputs.get('input_ids')
                attention_mask = inputs.get('attention_mask')
                labels = inputs.get('labels')
                past_key_values = None
                
                # Call the model's prepare_inputs_labels_for_multimodal method
                (input_ids, attention_mask, past_key_values, 
                 inputs_embeds, labels) = base_model.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, images
                )
                
                # Return prepared model inputs with ONLY inputs_embeds (not input_ids)
                model_inputs = {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'return_dict': True
                }
                
                # Add optional parameters if they exist in the original inputs
                for key in ['position_ids', 'past_key_values', 'use_cache', 'output_attentions', 'output_hidden_states']:
                    if key in inputs:
                        model_inputs[key] = inputs[key]
                
                # IMPORTANT: Ensure input_ids is NOT included
                # The model expects either input_ids OR inputs_embeds, not both
                return model_inputs
            
            # Fallback: just rename pixel_values to images
            model_inputs = {}
            for k, v in inputs.items():
                if k == 'pixel_values':
                    model_inputs['images'] = v

                    logging.get_logger(__name__).debug(f"Converted pixel_values to images parameter for GeoChat")
                else:
                    model_inputs[k] = v
            

            return model_inputs
        else:

            # For other models (standard LLaVA), keep all inputs
            return inputs
        
