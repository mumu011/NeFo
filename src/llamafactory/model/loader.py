# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

import sys
import os

# Add RS-LLaVA support
try:
    rs_llava_path = os.path.join(os.getcwd(), "src", "RS-LLaVA")
    if rs_llava_path not in sys.path:
        sys.path.append(rs_llava_path)
    from llava.model.builder import load_pretrained_model as rs_load_pretrained_model
    from llava.mm_utils import get_model_name_from_path as rs_get_model_name_from_path
    HAS_RS_LLAVA = True
except ImportError:
    HAS_RS_LLAVA = False

# Add GeoChat model support
try:
    from geochat.model import GeoChatLlamaForCausalLM
    GEODCHAT_AVAILABLE = True
except ImportError:
    GEODCHAT_AVAILABLE = False

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    if model_args.new_special_tokens is not None:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=model_args.new_special_tokens),
            replace_additional_special_tokens=False,
        )
        logger.info_rank0("Add {} to special tokens.".format(",".join(model_args.new_special_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    patch_tokenizer(tokenizer)
    
    # Special handling for GeoChat / InternVL3 - processor set in workflow
    if "geochat" in model_args.model_name_or_path.lower():
        logger.info_rank0("GeoChat model detected - processor will be loaded from vision tower in load_model")
        processor = None
    elif "internvl3" in model_args.model_name_or_path.lower():
        logger.info_rank0("InternVL3 model detected - processor will be set in vl_ttl workflow")
        processor = None
    else:
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, **init_kwargs)
        except Exception as e:
            logger.debug(f"Processor was not found: {e}.")
            processor = None

    # Patch processor if available
    if processor is not None:
        try:
            patch_processor(processor, config, tokenizer, model_args)
        except Exception as e:
            logger.warning_rank0(f"Failed to patch processor: {e}")
            # Don't set processor to None, keep it for vision processing

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    
    # Special handling for GeoChat models
    if "geochat" in model_args.model_name_or_path.lower():
        try:
            # First try to load with trust_remote_code
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **init_kwargs)
            logger.info_rank0(f"Successfully loaded GeoChat config with trust_remote_code=True")
            # Force set model_type to llava for compatibility with existing code
            if hasattr(config, 'model_type') and config.model_type == 'geochat':
                config.model_type = 'llava'
                logger.info_rank0("Set GeoChat config model_type to 'llava' for compatibility")
            return config
        except Exception as e:
            logger.warning_rank0(f"Failed to load GeoChat config with trust_remote_code=True: {e}")
            # Fallback: try to load as LLaVA config
            logger.info_rank0("Attempting to load GeoChat config as LLaVA config")
            config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
            # Force set model_type to llava for compatibility
            if hasattr(config, 'model_type'):
                config.model_type = 'llava'
                logger.info_rank0("Set GeoChat config model_type to 'llava' for compatibility")
            return config
    
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            # Special handling for GeoChat models
            if hasattr(config, 'model_type') and config.model_type == 'llava' and "geochat" in model_args.model_name_or_path.lower():
                # Use GeoChat model directly if available
                if GEODCHAT_AVAILABLE:
                    logger.info_rank0("Detected GeoChat model, using GeoChatLlamaForCausalLM")
                    if model_args.train_from_scratch:
                        model = GeoChatLlamaForCausalLM.from_config(config, trust_remote_code=True)
                    else:
                        model = GeoChatLlamaForCausalLM.from_pretrained(**init_kwargs)
                else:
                    # Fallback to AutoModelForVision2Seq
                    logger.info_rank0("GeoChat not available, falling back to AutoModelForVision2Seq")
                    load_class = AutoModelForVision2Seq
                    if model_args.train_from_scratch:
                        model = load_class.from_config(config, trust_remote_code=True)
                    else:
                        model = load_class.from_pretrained(**init_kwargs)
            # Special handling for InternVL3 (AutoModel + trust_remote_code)
            elif "internvl3" in model_args.model_name_or_path.lower():
                logger.info_rank0("Detected InternVL3 model, using AutoModel.from_pretrained")
                load_class = AutoModel
                if model_args.train_from_scratch:
                    model = load_class.from_config(config, trust_remote_code=True)
                else:
                    model = load_class.from_pretrained(**init_kwargs)
            # Special handling for RS-LLaVA
            elif "rs-llava" in model_args.model_name_or_path.lower() and HAS_RS_LLAVA:
                logger.info_rank0("Detected RS-LLaVA model, using native RS-LLaVA loader")
                model_path = model_args.model_name_or_path
                model_name = rs_get_model_name_from_path(model_path)
                
                # Try to infer base model path
                model_base = None
                if "Merged" not in model_path:
                    # Check common paths
                    base_candidates = [
                        "/mnt/nvme1/wj/Model/neural-chat-7b-v3-3",
                        "Intel/neural-chat-7b-v3-3"
                    ]
                    for cand in base_candidates:
                        if os.path.exists(cand) or "/" not in cand: # basic check
                            model_base = cand
                            if os.path.exists(cand): break
                    
                logger.info_rank0(f"Loading RS-LLaVA from {model_path} with base {model_base}")
                
                # Note: rs_load_pretrained_model returns (tokenizer, model, image_processor, context_len)
                # We only need the model here. Tokenizer is already loaded by load_tokenizer
                # But rs_load_pretrained_model might modify tokenizer/model embeddings
                # Since we can't easily pass our tokenizer to it, we let it load its own, then discard it
                # effectively just getting the model.
                
                # Careful with device placement - init_kwargs has device_map if set
                device_map = init_kwargs.get("device_map", "auto")
                
                _, model, _, _ = rs_load_pretrained_model(
                    model_path=model_path,
                    model_base=model_base,
                    model_name=model_name,
                    device_map=device_map,
                    device=init_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
                )
                logger.info_rank0("RS-LLaVA model loaded successfully")

            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
                load_class = AutoModelForVision2Seq
                if model_args.train_from_scratch:
                    model = load_class.from_config(config, trust_remote_code=True)
                else:
                    model = load_class.from_pretrained(**init_kwargs)
            else:
                load_class = AutoModelForCausalLM
                if model_args.train_from_scratch:
                    model = load_class.from_config(config, trust_remote_code=True)
                else:
                    model = load_class.from_pretrained(**init_kwargs)

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        
        # Special handling for GeoChat models - get processor from vision tower
        if "geochat" in model_args.model_name_or_path.lower() and hasattr(model, 'get_vision_tower'):
            try:
                vision_tower = model.get_vision_tower()
                if vision_tower is not None and hasattr(vision_tower, 'image_processor'):
                    # We need to set the processor in the tokenizer_module
                    # This will be handled by returning it from load_model
                    logger.info_rank0("GeoChat vision tower processor detected")
                else:
                    logger.warning_rank0("GeoChat vision tower or image processor not found")
            except Exception as e:
                logger.warning_rank0(f"Failed to access GeoChat vision tower: {e}")

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # 仅 InternVL：init_adapter 后若为 PeftModel，generate 会调用 base_model，补 patch cache_position
    _path_lower = (getattr(model_args, "model_name_or_path", None) or "").lower()
    if "internvl3" in _path_lower or "internvl" in _path_lower:
        import inspect
        _m = getattr(model, "base_model", model)
        _inner = getattr(_m, "model", _m)
        for _target in (_m, _inner):
            if _target is None:
                continue
            try:
                _sig = inspect.signature(_target.forward)
                if "cache_position" not in _sig.parameters:
                    _orig = _target.forward
                    def _strip(*args, **kwargs):
                        kwargs.pop("cache_position", None)
                        kwargs.pop("position_ids", None)
                        kwargs.pop("inputs_embeds", None)
                        return _orig(*args, **kwargs)
                    _target.forward = _strip
                    logger.info_rank0("Post-init_adapter: patched forward to strip cache_position/position_ids/inputs_embeds for InternVL.")
                    break
            except (TypeError, ValueError, AttributeError):
                continue

        # InternVL：generate() 时 _validate_model_kwargs 会拒绝 labels/pixel_values（prepare_inputs_for_generation 签名不含）。
        # PEFT 会调用 base_model.generate()，校验在 base_model 上执行，故需同时 patch 顶层 model 与 base_model。
        # 原方法签名为 (self, model_kwargs)，wrapper 需接收 self 并正确传给原方法。
        def _internvl_validate_fn(orig_validate):
            def _internvl_validate(self, model_kwargs):
                if model_kwargs is not None:
                    model_kwargs.pop("labels", None)
                    model_kwargs.pop("pixel_values", None)
                return orig_validate(model_kwargs)
            return _internvl_validate

        _patched = False
        for _patch_target in (getattr(model, "base_model", None), model):
            if _patch_target is None:
                continue
            if hasattr(_patch_target, "_validate_model_kwargs"):
                _patch_target._validate_model_kwargs = _internvl_validate_fn(_patch_target._validate_model_kwargs)
                _patched = True
        if _patched:
            logger.info_rank0("Patched _validate_model_kwargs for InternVL (allow labels/pixel_values in generate).")

        # InternVL：generate() 用 img_context_token_id 找图像位置；input_ids 须为合法 token id（不能 -200，否则 embedding 会触发 CUDA assert）
        # 用 tokenizer 的 <IMG_CONTEXT> 或 <image> 的 id，并在 trainer 里把 -200 替换为该 id
        _img_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if _img_id is None or _img_id == tokenizer.unk_token_id:
            _img_id = tokenizer.convert_tokens_to_ids("<image>")
        if _img_id is None or _img_id == tokenizer.unk_token_id:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<IMG_CONTEXT>"]})
            _img_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        _candidates = [model]
        _b = getattr(model, "base_model", None)
        if _b is not None:
            _candidates.append(_b)
            _inner = getattr(_b, "model", None)
            if _inner is not None:
                _candidates.append(_inner)
        for _m in _candidates:
            if _m is not None and hasattr(_m, "img_context_token_id"):
                _m.img_context_token_id = _img_id
        logger.info_rank0(f"Set InternVL img_context_token_id={_img_id} at load (chain len={len(_candidates)}).")

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )

    return model
