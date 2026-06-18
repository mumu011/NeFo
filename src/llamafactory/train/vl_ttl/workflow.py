import warnings
import json
warnings.filterwarnings('ignore')
import sys
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
# 特别忽略 transformers 的弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*Trainer.tokenizer is now deprecated.*')
from typing import TYPE_CHECKING, List, Optional


from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import cal_effective_tokens, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
from datasets import Dataset

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import os

from ...extras import logging


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments

logger = logging.get_logger(__name__)

class VLTTLModel(nn.Module):
    def __init__(self,
                 data_args: "DataArguments",
                 model_args: "ModelArguments",
                 training_args: "Seq2SeqTrainingArguments",
                 finetuning_args: "FinetuningArguments",
                 generating_args: "GeneratingArguments",
                tokenizer_module,
                template,
                model
                ):
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.finetuning_args = finetuning_args
        self.model_args = model_args
        self.generating_args = generating_args
        self.template = template

        self.tokenizer_module = tokenizer_module
        self.tokenizer = self.tokenizer_module["tokenizer"]
        self.processor = self.tokenizer_module.get("processor", None)

        self.model = model

        self.trainer = None

        self.base_output_dir = self.training_args.output_dir

        # 检查是否为视觉语言模型
        self.is_vision_model = hasattr(self.model, 'get_vision_tower') or hasattr(self.model, 'vision_tower')
        if self.is_vision_model:
            logger.info_rank0("Detected Vision-Language Model, enabling visual features processing")


    def reset_trainer(self, train_dataset, **kwargs):
        # InternVL：generate() 要求 img_context_token_id 为合法 token id（不能 -200，否则 embedding 会 CUDA assert）
        # 与 loader 一致：用 tokenizer 的 <IMG_CONTEXT> 或 <image> 的 id；trainer 里会把 -200 替换为该 id
        _path_lower = (getattr(self.model_args, "model_name_or_path", None) or "").lower()
        if "internvl" in _path_lower:
            _img_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if _img_id is None or getattr(self.tokenizer, "unk_token_id", None) is not None and _img_id == self.tokenizer.unk_token_id:
                _img_id = self.tokenizer.convert_tokens_to_ids("<image>")
            if _img_id is None or getattr(self.tokenizer, "unk_token_id", None) is not None and _img_id == self.tokenizer.unk_token_id:
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["<IMG_CONTEXT>"]})
                _img_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            _candidates = [self.model]
            _b = getattr(self.model, "base_model", None)
            if _b is not None:
                _candidates.append(_b)
                _inner = getattr(_b, "model", None)
                if _inner is not None:
                    _candidates.append(_inner)
            for _m in _candidates:
                if _m is not None and hasattr(_m, "img_context_token_id"):
                    _m.img_context_token_id = _img_id
            logger.info_rank0(f"Set InternVL img_context_token_id={_img_id} for generate() (chain len={len(_candidates)}).")

        data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            # pad_to_multiple_of=8 if self.training_args.do_train else None,  # for shift short attention
            pad_to_multiple_of= None,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=self.model_args.block_diag_attn,
            attn_implementation=getattr(self.model.config, "_attn_implementation", None),
            compute_dtype=self.model_args.compute_dtype,
            **self.tokenizer_module,
        )

        self.trainer = CustomSeq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            finetuning_args=self.finetuning_args,
            model_args=self.model_args,
            data_args=self.data_args,
            generating_args=self.generating_args,  # 🔧 修复：加回missing的generating_args
            data_collator=data_collator,
            train_dataset=train_dataset,
            **self.tokenizer_module,
            **kwargs
        )

    def forward(self, train_batch, predict_batch):
        if self.finetuning_args.setting == "offline_ttl":
            self.forward_for_offline(train_batch=train_batch, predict_batch=predict_batch)

        elif self.finetuning_args.setting == "online_ttl" or self.finetuning_args.setting == "online_ttl_TLM":
            self.forward_for_online(train_batch=train_batch, predict_batch=predict_batch)

        elif self.finetuning_args.setting == "tent":
            # TENT 使用 online 流程（先预测，再训练）
            self.forward_for_online(train_batch=train_batch, predict_batch=predict_batch)

        elif self.finetuning_args.setting == "sar":
            # SAR 使用 online 流程（先预测，再训练）
            self.forward_for_online(train_batch=train_batch, predict_batch=predict_batch)

        else:
            raise ValueError(
                f'NO such setting: {self.finetuning_args.setting}'
            )

    def forward_for_offline(self, train_batch, predict_batch):
        """
        First Train, then Predict.
        This is the offline TTL setting for Vision-Language Models, where we first train the model using only the inputs, then use the trained model to predict the results of the training data.
        """
        logger.info_rank0("🚀 Starting OFFLINE TTL Training for Vision-Language Model...")
        logger.info_rank0(f"Training dataset size: {len(train_batch)} samples")
        logger.info_rank0(f"Evaluation dataset size: {len(predict_batch)} samples")

        # train
        self.tokenizer.padding_side = "right"  # use right-padding in training
        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset
        self.reset_trainer(train_dataset=train_batch)

        # 添加简单的进度显示
        logger.info_rank0("🎯 Training started...")
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        logger.info_rank0("✅ Training completed!")
        self.trainer.save_model()

        self.unwrap_model()

        # predict
        logger.info_rank0("🔮 Starting prediction phase...")
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        # decoder-only models must use left-padding for batched generation.
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"  # use left-padding in generation
        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'

        self.reset_trainer(train_dataset=None)
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
        logger.info_rank0("✅ Prediction completed and results saved!")


    def forward_for_online(self, train_batch, predict_batch):
        """
        First Predict, then Train.
        This is the online TTL setting, where we first predict the results of the training data, then train the model with only the inputs.
        """
        ####################################
        # use the latest model to predict
        ####################################
        print(f"🚀 Starting ONLINE TTL - Prediction Phase...")
        print(f"📊 Current batch size: {len(train_batch)} samples")

        self.training_args.output_dir = self.base_output_dir + f'/predict-temperature_{self.generating_args.temperature}-max_new_tokens_{self.generating_args.max_new_tokens}'  # the folder to save prediction results
        # Keyword arguments for `model.generate`
        gen_kwargs = self.generating_args.to_dict()
        gen_kwargs["eos_token_id"] = [self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs["logits_processor"] = get_logits_processor()
        # decoder-only models must use left-padding for batched generation.
        if self.training_args.predict_with_generate:
            self.tokenizer.padding_side = "left"  # use left-padding in generation

        self.training_args.generation_max_length = self.training_args.generation_max_length or self.data_args.cutoff_len
        self.training_args.generation_num_beams = self.data_args.eval_num_beams or self.training_args.generation_num_beams
        self.training_args.remove_unused_columns = False  # important for multimodal dataset

        self.reset_trainer(train_dataset=None)
        print(f"🔮 Running prediction on current batch...")
        predict_results = self.trainer.predict(predict_batch, metric_key_prefix="predict", **gen_kwargs)
        self.trainer.save_predictions(predict_batch, predict_results)
        print(f"✅ Prediction completed!")

        self.unwrap_model()

        # 训练阶段
        print(f"🎯 Starting training phase...")
        self.training_args.output_dir = self.base_output_dir  # 保存 adapter 的文件夹
        self.tokenizer.padding_side = "right"

        def _flatten_dataset(batch: Dataset) -> Dataset:
            print(f"📦 _flatten_dataset input: {len(batch)} samples")
            if len(batch) != 1:
                print(f"   └─ Not a single sample, returning as-is")
                return batch
            sample = batch[0]
            list_keys = [key for key, value in sample.items() if isinstance(value, list)]
            print(f"   └─ List keys found: {list_keys}")
            if not list_keys:
                print(f"   └─ No list keys, returning as-is")
                return batch
            target_len = len(sample[list_keys[0]])
            print(f"   └─ Flattening {target_len} sub-samples from key '{list_keys[0]}'")
            records = []
            for i in range(target_len):
                record = {}
                for key, value in sample.items():
                    if isinstance(value, list):
                        record[key] = value[i]
                    else:
                        record[key] = value
                records.append(record)
            flattened = Dataset.from_list(records)
            print(f"   └─ Flattened to {len(flattened)} samples")
            return flattened

        if not isinstance(train_batch, Dataset):
            raise TypeError("train_batch must be a Dataset in the simplified single-GPU mode.")

        train_dataset = _flatten_dataset(train_batch)
        self.reset_trainer(train_dataset=train_dataset)
        print(f"🔄 Training on current batch with {len(train_dataset)} samples...")
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()    # 保存模型到 training_args.output_dir
        print(f"✅ Training completed and model saved!")

        self.unwrap_model()

    def unwrap_model(self):
        self.model = self.trainer.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False)


def run_vl_ttl(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    Run Vision-Language TTL (Test-Time Learning) training.

    This function extends the standard TTL approach to handle vision-language models
    by properly processing multimodal inputs (text + images) during training and inference.
    """
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # 添加否定词过滤的特殊token（如果启用了否定词过滤）
    if getattr(data_args, 'enable_negation_filtering', False):
        if '<Neg_Mask>' not in tokenizer.get_vocab():
            special_tokens_dict = {'additional_special_tokens': ['<Neg_Mask>']}  # 🔧 恢复原来验证成功的版本
            num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
            logger.info_rank0(f"Added {num_added_tokens} special tokens to tokenizer")

            # 更新tokenizer_module
            tokenizer_module["tokenizer"] = tokenizer
        else:
            logger.info_rank0("<Neg_Mask> token already exists")

    # Load model first to get GeoChat vision tower processor if needed
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    # For GeoChat models, get processor from vision tower and update tokenizer_module
    if "geochat" in model_args.model_name_or_path.lower():
        image_processor = None
        vision_tower = model.get_vision_tower()
        image_processor = vision_tower.image_processor
        logger.info_rank0("✅ Successfully loaded GeoChat image processor from vision tower: image_processor")

        # Create a wrapper processor object that matches GeoChat's architecture and processing
        if image_processor is not None:
            class ProcessorWrapper:
                def __init__(self, image_processor):
                    self.original_image_processor = image_processor
                    # Match GeoChat's real architecture parameters
                    self.patch_size = 14  # GeoChat uses patch_size=14 (from clip_encoder.py)
                    self.vision_feature_select_strategy = 'default'
                    # Calculate real token count: (504/14)^2 = 1296 tokens
                    self.num_additional_image_tokens = 1  # CLS token
                    self.image_max_pixels = 504 * 504  # GeoChat standard
                    self.image_min_pixels = 32 * 32
                    # Real GeoChat image sequence length: 36*36 = 1296 patches
                    self.image_seq_length = (504 // 14) ** 2  # 1296 tokens for 504x504 image

                @property
                def image_processor(self):
                    """返回自己作为image_processor，让mm_plugin直接使用我们的处理"""
                    return self

                def preprocess(self, images, return_tensors="pt", **kwargs):
                    """mm_plugin调用的主要方法 - 用于数据预处理阶段"""
                    return self.__call__(images, return_tensors, **kwargs)

                def __call__(self, images, return_tensors="pt", **kwargs):
                    """Process images using GeoChat's expand2square + resize methodology."""
                    from PIL import Image
                    import torch

                    if not isinstance(images, list):
                        images = [images]

                    processed_images = []
                    for image in images:
                        # Step 1: Convert to PIL if needed
                        if isinstance(image, str):
                            image = Image.open(image).convert('RGB')
                        elif hasattr(image, 'convert'):
                            image = image.convert('RGB')

                        # Step 2: GeoChat's expand2square function
                        image = self._expand2square(image, tuple(int(x*255) for x in self.original_image_processor.image_mean))

                        # Step 3: GeoChat's resize to 504x504 (exactly as in geochat/mm_utils.py)
                        image = self.original_image_processor.preprocess(
                            image,
                            crop_size={'height': 504, 'width': 504},
                            size={'shortest_edge': 504},
                            return_tensors='pt'
                        )['pixel_values'][0]

                        processed_images.append(image)

                    if processed_images:
                        pixel_values = torch.stack(processed_images, dim=0)
                        return {'pixel_values': pixel_values}
                    else:
                        return {}

                def _expand2square(self, pil_img, background_color):
                    #将非正方形图像扩展为正方形，使用背景色填充
                    """GeoChat's expand2square function - exactly copied from geochat/mm_utils.py"""
                    from PIL import Image

                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                def save_pretrained(self, output_dir):
                    """Save the wrapped processor to the specified directory."""
                    # For GeoChat, we save the underlying original image processor (not self)
                    if hasattr(self.original_image_processor, 'save_pretrained'):
                        self.original_image_processor.save_pretrained(output_dir)
                    else:
                        # If the image processor doesn't have save_pretrained,
                        # we can skip saving as it's likely a standard CLIP processor
                        logger.info_rank0(f"🔧 ProcessorWrapper: Skipping processor save (GeoChat uses standard CLIP processor)")

            wrapped_processor = ProcessorWrapper(image_processor)
            tokenizer_module["processor"] = wrapped_processor
            logger.info_rank0("🔧 Created GeoChat processor wrapper with authentic expand2square + 504x504 resize (1296 tokens)")

    # For RS-LLaVA models (BigData-KSU/RS-LLaVA), 参照 src/RS-LLaVA/llava/mm_utils.py 的 process_images / expand2square
    elif "rs-llava" in model_args.model_name_or_path.lower():
        vision_tower = model.get_vision_tower() if hasattr(model, 'get_vision_tower') else getattr(model, 'vision_tower', None)
        image_processor = vision_tower.image_processor if (vision_tower and hasattr(vision_tower, 'image_processor')) else None
        if image_processor is not None:
            model_cfg = model.config

            def _expand2square(pil_img, background_color):
                """与 RS-LLaVA llava/mm_utils.py 中 expand2square 一致"""
                from PIL import Image
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            class RSLLaVAProcessorWrapper:
                """RS-LLaVA processor 包装，逻辑与 llava/mm_utils.py process_images 一致"""
                def __init__(self, image_processor, model_config):
                    self._image_processor = image_processor
                    self._model_config = model_config
                    self.image_aspect_ratio = getattr(model_config, "image_aspect_ratio", None)
                    self.vision_feature_select_strategy = "default"
                    self.num_additional_image_tokens = 1
                    if hasattr(image_processor, "crop_size") and isinstance(image_processor.crop_size, dict):
                        self.image_size = image_processor.crop_size.get("height", 336)
                    elif hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
                        self.image_size = image_processor.size.get("shortest_edge", 336)
                    else:
                        self.image_size = 336
                    self.patch_size = 14
                    # RS-LLaVA 在 forward 中用一个占位符接收整图特征，input_ids 中每张图只保留 1 个 -200
                    self.image_seq_length = 1
                    self.image_max_pixels = getattr(model_config, "image_max_pixels", 768 * 768)
                    self.image_min_pixels = getattr(model_config, "image_min_pixels", 32 * 32)
                    logger.info_rank0(
                        f"🔧 RS-LLaVA processor: size={self.image_size}, patch={self.patch_size}, "
                        f"image_seq_length={self.image_seq_length} (one placeholder per image), image_aspect_ratio={self.image_aspect_ratio}"
                    )

                @property
                def image_processor(self):
                    return self

                def preprocess(self, images, return_tensors="pt", **kwargs):
                    return self(images, return_tensors=return_tensors, **kwargs)

                def __call__(self, images, return_tensors="pt", **kwargs):
                    from PIL import Image
                    if not isinstance(images, list):
                        images = [images]
                    pil_images = []
                    for image in images:
                        if isinstance(image, str):
                            image = Image.open(image).convert("RGB")
                        elif hasattr(image, "convert"):
                            image = image.convert("RGB")
                        pil_images.append(image)
                    # 与 mm_utils.process_images 一致：pad 时 expand2square + preprocess；否则直接 image_processor(images)
                    if self.image_aspect_ratio == "pad":
                        new_images = []
                        for image in pil_images:
                            image = _expand2square(
                                image, tuple(int(x * 255) for x in self._image_processor.image_mean)
                            )
                            image = self._image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                            new_images.append(image)
                        if not new_images:
                            return {}
                        # 与 mm_utils.process_images 一致，形状一致时 stack（pad 后通常一致）
                        return {"pixel_values": torch.stack(new_images, dim=0)}
                    else:
                        out = self._image_processor(pil_images, return_tensors=return_tensors, **kwargs)
                        return {"pixel_values": out["pixel_values"]} if isinstance(out, dict) and "pixel_values" in out else out

                def save_pretrained(self, output_dir):
                    if hasattr(self._image_processor, "save_pretrained"):
                        self._image_processor.save_pretrained(output_dir)

            tokenizer_module["processor"] = RSLLaVAProcessorWrapper(image_processor, model_cfg)
            logger.info_rank0("🔧 Created RS-LLaVA processor wrapper (logic from llava/mm_utils.py)")
        else:
            logger.warning_rank0("⚠️ RS-LLaVA: vision_tower or image_processor not found")

    # For InternVL3: use AutoProcessor and wrap for mm_plugin (pixel_values, image_seq_length, etc.)
    elif "internvl3" in model_args.model_name_or_path.lower():
        try:
            from transformers import AutoProcessor as _AutoProcessor
            _init_kwargs = {"trust_remote_code": True}
            raw_proc = _AutoProcessor.from_pretrained(
                model_args.model_name_or_path, **_init_kwargs
            )
            model_cfg = getattr(model, "config", None)
            vision_cfg = getattr(model_cfg, "vision_config", None) or getattr(model_cfg, "visual_config", None)
            _img_size = int(getattr(vision_cfg, "image_size", 448)) if vision_cfg is not None else 448
            # 必须与模型一致：使用 model.num_image_token（含 downsample_ratio²），否则 input_embeds[selected] 与 vit_embeds 形状不匹配
            _image_seq_length = None
            _b = getattr(model, "base_model", None)
            for _m in (_b, getattr(_b, "model", None) if _b is not None else None, model):
                if _m is not None and hasattr(_m, "num_image_token"):
                    _image_seq_length = _m.num_image_token
                    break
            if _image_seq_length is None and vision_cfg is not None:
                _patch = getattr(vision_cfg, "patch_size", 14)
                _image_seq_length = (_img_size // int(_patch)) ** 2
            if _image_seq_length is None:
                _image_seq_length = (448 // 14) ** 2  # 1024 fallback

            def _internvl3_images_to_pixel_values(images, image_size: int = 448):
                """Fallback: HF processor 非纯图像时，用 PIL + resize + normalize 得到 pixel_values。"""
                from PIL import Image
                import numpy as np
                if not isinstance(images, list):
                    images = [images]
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                tensors = []
                for img in images:
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    elif hasattr(img, "convert"):
                        img = img.convert("RGB")
                    else:
                        img = Image.fromarray(np.asarray(img)).convert("RGB")
                    img = img.resize((image_size, image_size), getattr(getattr(Image, "Resampling", None), "BICUBIC", 3))
                    t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
                    t = (t - torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)) / torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
                    tensors.append(t)
                if not tensors:
                    return {}
                return {"pixel_values": torch.stack(tensors)}

            class InternVL3ProcessorWrapper:
                """Wrapper: image_processor 指向 self，图像处理用 HF 图像组件或 fallback 纯 resize+normalize。"""
                def __init__(self, processor, image_seq_length: int = 1024, image_size: int = 448):
                    self._processor = processor
                    self.image_seq_length = image_seq_length
                    self.image_size = image_size
                    self.vision_feature_select_strategy = "default"
                    self.patch_size = 14
                    self.num_additional_image_tokens = 0
                    self.image_processor = self
                    _ip = getattr(processor, "image_processor", None)
                    # 仅当明确是图像处理器（类名含 ImageProcessor/ImageProcessing）时才用，否则一律 fallback
                    self._use_hf_image_proc = (
                        _ip is not None
                        and "tokeniz" not in type(_ip).__name__.lower()
                        and (hasattr(_ip, "preprocess") or "image" in type(_ip).__name__.lower())
                    )
                    self._img_proc = _ip if self._use_hf_image_proc else None

                def preprocess(self, images, return_tensors="pt", **kwargs):
                    return self(images, return_tensors=return_tensors, **kwargs)

                def __call__(self, images, return_tensors="pt", **kwargs):
                    if not isinstance(images, list):
                        images = [images]
                    n_images = len(images)
                    out = {}
                    if self._img_proc is not None and hasattr(self._img_proc, "preprocess"):
                        try:
                            out = self._img_proc.preprocess(images, return_tensors=return_tensors, **kwargs)
                        except (TypeError, ValueError):
                            out = {}
                    if not isinstance(out, dict) or "pixel_values" not in out:
                        out = _internvl3_images_to_pixel_values(images, self.image_size)
                    if isinstance(out, dict) and "pixel_values" in out:
                        # intern_vl plugin expects num_patches (one per image; standard InternVL3 = 1 patch per image)
                        if "num_patches" not in out:
                            out["num_patches"] = [1] * n_images
                        return out
                    if isinstance(out, torch.Tensor):
                        return {"pixel_values": out.unsqueeze(0) if out.dim() == 3 else out, "num_patches": [1] * n_images}
                    out = _internvl3_images_to_pixel_values(images, self.image_size)
                    out["num_patches"] = [1] * n_images
                    return out

                def save_pretrained(self, output_dir):
                    if hasattr(self._processor, "save_pretrained"):
                        self._processor.save_pretrained(output_dir)

            tokenizer_module["processor"] = InternVL3ProcessorWrapper(raw_proc, _image_seq_length, image_size=_img_size)
            logger.info_rank0(
                f"🔧 Created InternVL3 processor wrapper (image_seq_length={_image_seq_length})"
            )
            # 避免 open-end generation 时 "Setting pad_token_id to eos_token_id" 警告
            _tok = tokenizer_module.get("tokenizer")
            if _tok is not None and _tok.pad_token_id is None:
                _tok.pad_token_id = _tok.eos_token_id
                if getattr(_tok, "pad_token", None) is None:
                    _tok.pad_token = _tok.eos_token
        except Exception as e:
            logger.warning_rank0(f"⚠️ InternVL3 processor setup failed: {e}, trying model.vision_tower")

    # Ensure processor is available for dataset processing (other VLMs)
    if tokenizer_module.get("processor") is None:
        logger.warning_rank0("⚠️ No processor found, trying to get from model directly")
        try:
            if hasattr(model, 'get_vision_tower'):
                vision_tower = model.get_vision_tower()
                if vision_tower and hasattr(vision_tower, 'image_processor'):
                    tokenizer_module["processor"] = vision_tower.image_processor
                    logger.info_rank0("✅ Got processor from model.get_vision_tower()")
            elif hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'image_processor'):
                tokenizer_module["processor"] = model.vision_tower.image_processor
                logger.info_rank0("✅ Got processor from model.vision_tower")
        except Exception as e:
            logger.warning_rank0(f"⚠️ Failed to get processor from model: {e}")

    logger.info_rank0(f"Final processor: {type(tokenizer_module.get('processor')).__name__ if tokenizer_module.get('processor') else 'None'}")


    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage="vl_ttl",
        tokenizer=tokenizer_module["tokenizer"],
        processor=tokenizer_module.get("processor")
    )


    train_dataset = dataset_module['train_dataset']
    eval_dataset = dataset_module['eval_dataset']

    # 将 train_dataset 的 question_id 持久化到本地，便于后续在训练集上做回测/评估
    try:
        os.makedirs(training_args.output_dir, exist_ok=True)

        def _collect_question_ids(ds: Dataset):
            ids = []
            # 逐条收集，兼容可能存在的嵌套 sub_samples 结构
            for item in ds:
                # 顶层 question_id
                if 'question_id' in item:
                    qid = item['question_id']
                    if isinstance(qid, list):
                        ids.extend([str(x) for x in qid])
                    else:
                        ids.append(str(qid))
                # 子样本中的 question_id
                if 'sub_samples' in item and isinstance(item['sub_samples'], list):
                    for s in item['sub_samples']:
                        if isinstance(s, dict) and 'question_id' in s:
                            ids.append(str(s['question_id']))
            # 去重，保持稳定顺序
            seen = set()
            uniq = []
            for x in ids:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            return uniq

        train_qids = _collect_question_ids(train_dataset)
        qids_path = os.path.join(training_args.output_dir, "train_question_ids.jsonl")
        with open(qids_path, 'w', encoding='utf-8') as f:
            for q in train_qids:
                f.write(json.dumps({"question_id": q}, ensure_ascii=False) + "\n")
        logger.info_rank0(f"📝 Saved {len(train_qids)} train question_id(s) to {qids_path}")
    except Exception as e:
        logger.warning_rank0(f"⚠️ Failed to persist train question_ids: {e}")

    # 显示数据集信息
    logger.info_rank0("📊 Vision-Language TTL Dataset Information:")
    logger.info_rank0(f"   Training dataset: {len(train_dataset)} samples")
    logger.info_rank0(f"   Evaluation dataset: {len(eval_dataset)} samples")
    
    # 🔍 检查第一个样本的结构
    if len(train_dataset) > 0:
        first_sample = train_dataset[0]
        logger.info_rank0(f"   First sample keys: {list(first_sample.keys())}")
        if 'sub_samples' in first_sample:
            logger.info_rank0(f"   First sample has {len(first_sample['sub_samples'])} sub_samples")
        else:
            logger.info_rank0(f"   First sample has NO 'sub_samples' field (flat structure)")
    
    logger.info_rank0(f"   Training batch size: {training_args.per_device_train_batch_size}")
    logger.info_rank0(f"   Total training steps: {training_args.max_steps or 'auto'}")
    logger.info_rank0(f"   Total training epochs: {training_args.num_train_epochs}")


    vl_ttl_model = VLTTLModel(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
        tokenizer_module=tokenizer_module,
        template=template,
        model=model
    )

    if finetuning_args.setting == "offline_ttl":
        logger.info_rank0(f"🎯 Running OFFLINE TTL for Vision-Language Model...")
        vl_ttl_model.forward(train_batch=train_dataset, predict_batch=eval_dataset)
    elif finetuning_args.setting == "online_ttl" or finetuning_args.setting == "online_ttl_TLM":
        print(f"🎯 Running ONLINE TTL...")
        streaming_batch_size = finetuning_args.streaming_batch_size
        num_of_batch = len(train_dataset) // streaming_batch_size
        if len(train_dataset) % streaming_batch_size != 0:
            num_of_batch += 1

        print(f"📊 Online TTL Configuration:")
        print(f"   Total parent samples: {len(train_dataset)}")
        print(f"   Streaming batch size: {streaming_batch_size}")
        print(f"   Total streaming batches: {num_of_batch}")

        for k in range(num_of_batch):
            print(f"\n{'='*80}")
            print(f"🔄 Processing streaming batch {k + 1}/{num_of_batch}")
            print(f"{'='*80}")
            if (k + 1) * streaming_batch_size > len(train_dataset):
                end_index = len(train_dataset)
            else:
                end_index = (k + 1) * streaming_batch_size
            sub_trainset = train_dataset.select(range(k * streaming_batch_size, end_index))
            sub_evalset = eval_dataset.select(range(k * streaming_batch_size, end_index))

            print(f"   Selected parent samples: {len(sub_trainset)} (indices {k * streaming_batch_size} to {end_index-1})")
            print(f"   About to flatten and train...")
            vl_ttl_model.forward(train_batch=sub_trainset, predict_batch=sub_evalset)
            print(f"✅ Completed streaming batch {k + 1}/{num_of_batch}")
    elif finetuning_args.setting == "tent":
        print(f"🎯 Running TENT (Test-Time Entropy Minimization)...")
        streaming_batch_size = finetuning_args.streaming_batch_size
        num_of_batch = len(train_dataset) // streaming_batch_size
        if len(train_dataset) % streaming_batch_size != 0:
            num_of_batch += 1

        print(f"📊 TENT Configuration:")
        print(f"   Total parent samples: {len(train_dataset)}")
        print(f"   Streaming batch size: {streaming_batch_size}")
        print(f"   Total streaming batches: {num_of_batch}")

        for k in range(num_of_batch):
            print(f"\n{'='*80}")
            print(f"🔄 Processing TENT batch {k + 1}/{num_of_batch}")
            print(f"{'='*80}")
            if (k + 1) * streaming_batch_size > len(train_dataset):
                end_index = len(train_dataset)
            else:
                end_index = (k + 1) * streaming_batch_size
            sub_trainset = train_dataset.select(range(k * streaming_batch_size, end_index))
            sub_evalset = eval_dataset.select(range(k * streaming_batch_size, end_index))

            print(f"   Selected parent samples: {len(sub_trainset)} (indices {k * streaming_batch_size} to {end_index-1})")
            print(f"   Minimizing entropy on test samples...")
            vl_ttl_model.forward(train_batch=sub_trainset, predict_batch=sub_evalset)
            print(f"✅ Completed TENT batch {k + 1}/{num_of_batch}")
    elif finetuning_args.setting == "sar":
        print(f"🎯 Running SAR (Sharpness-Aware Reliable Entropy Minimization)...")
        streaming_batch_size = finetuning_args.streaming_batch_size
        num_of_batch = len(train_dataset) // streaming_batch_size
        if len(train_dataset) % streaming_batch_size != 0:
            num_of_batch += 1

        print(f"📊 SAR Configuration:")
        print(f"   Total parent samples: {len(train_dataset)}")
        print(f"   Streaming batch size: {streaming_batch_size}")
        print(f"   Total streaming batches: {num_of_batch}")
        print(f"   Entropy threshold: 0.4 * ln(vocab_size) (auto-computed)")

        for k in range(num_of_batch):
            print(f"\n{'='*80}")
            print(f"🔄 Processing SAR batch {k + 1}/{num_of_batch}")
            print(f"{'='*80}")
            if (k + 1) * streaming_batch_size > len(train_dataset):
                end_index = len(train_dataset)
            else:
                end_index = (k + 1) * streaming_batch_size
            sub_trainset = train_dataset.select(range(k * streaming_batch_size, end_index))
            sub_evalset = eval_dataset.select(range(k * streaming_batch_size, end_index))

            print(f"   Selected parent samples: {len(sub_trainset)} (indices {k * streaming_batch_size} to {end_index-1})")
            print(f"   Reliable entropy minimization with sample filtering...")
            vl_ttl_model.forward(train_batch=sub_trainset, predict_batch=sub_evalset)
            print(f"✅ Completed SAR batch {k + 1}/{num_of_batch}")
    else:
        raise ValueError(
            f'NO such setting: {finetuning_args.setting}'
        )