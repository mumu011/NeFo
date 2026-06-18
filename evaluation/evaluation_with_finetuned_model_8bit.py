import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import trange
from transformers import BitsAndBytesConfig

from geochat.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from geochat.conversation import conv_templates
from geochat.mm_utils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from geochat.utils import disable_torch_init
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer
from transformers import AutoConfig, AutoModel  # for robust fallback

try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None
try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except Exception:  # pragma: no cover
    Qwen3VLForConditionalGeneration = None
try:
    # 官方多模态辅助函数（图像/视频预处理）
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # 若环境未提供，后面提供一个简易回退
    process_vision_info = None  # type: ignore
from geochat.model import GeoChatLlamaForCausalLM

try:
    # 尝试动态添加 RS-LLaVA 路径
    import sys

    rs_llava_path = os.path.join(os.getcwd(), "src", "RS-LLaVA")
    if rs_llava_path not in sys.path:
        sys.path.append(rs_llava_path)

    from llava.constants import (
        IMAGE_TOKEN_INDEX as RS_IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN as RS_DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN as RS_DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN as RS_DEFAULT_IM_END_TOKEN,
    )
    from llava.conversation import conv_templates as rs_conv_templates
    from llava.mm_utils import (
        tokenizer_image_token as rs_tokenizer_image_token,
        process_images as rs_process_images,
        KeywordsStoppingCriteria as RSKeywordsStoppingCriteria,
        get_model_name_from_path as rs_get_model_name_from_path,
    )
    from llava.model.builder import load_pretrained_model as rs_load_pretrained_model
    from llava.utils import disable_torch_init as rs_disable_torch_init

    HAS_RS_LLAVA = True
except ImportError:
    HAS_RS_LLAVA = False
    print("Warning: RS-LLaVA module not found or failed to import.")

# ----------------------------------------------------------------------
# InternVL2.5 Helper Functions
# ----------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_aspect_ratio[0])) * image_size,
            (i // (target_aspect_ratio[0])) * image_size,
            (i % (target_aspect_ratio[0]) + 1) * image_size,
            (i // (target_aspect_ratio[0]) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl(image, input_size=448, max_num=12):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GeoChat with fine-tuned LoRA weights")
    parser.add_argument(
        "--base-model",
        type=Path,
        required=False,
        default=Path("/home/hanhch/pretrained_models/geochat-7B"),
        help="Path to the base GeoChat-7B model",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        required=False,
        default=None,  # 修改为 None
        help="Directory containing the fine-tuned LoRA adapter (optional, if not provided, use base model only)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=False,
        default=Path("data/the7/Scene_Classification/UCmerced_negated_llava_format_shuffle.jsonl"),
        help="Path to the evaluation dataset (JSONL)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=False,
        default=Path("/home/hanhch/geo_data/UCMerced_LandUse/Images"),
        help="Directory containing the dataset images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=False,
        default=Path("outputs/geochat_lora_predictions.jsonl"),
        help="File to save the generated predictions",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum new tokens during generation")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="llava_v1",
        help="Conversation template to use (default: llava_v1)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="geochat",
        choices=["geochat", "qwen2_5_vl", "rs_llava", "internvl2_5", "georeason", "qwen3vl"],
        help="Model backend: geochat | qwen2_5_vl (仅保留 2.5 版本) | rs_llava | internvl2_5 | georeason | qwen3vl",
    )
    parser.add_argument(
        "--qwen2-vl-model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HF hub id 或本地路径: 如 Qwen/Qwen2.5-VL-7B-Instruct，可覆盖以使用其他尺寸/版本",
    )
    parser.add_argument(
        "--qwen3-vl-model",
        type=str,
        default="/mnt/nvme1/wj/Model/Qwen3-VL-8B-Instruct/",
        help="HF hub id 或本地路径: 如 /mnt/nvme1/wj/Model/Qwen3-VL-8B-Instruct/",
    )
    parser.add_argument(
        "--internvl-model",
        type=str,
        default="OpenGVLab/InternVL2_5-8B",
        help="HF hub id 或本地路径: 如 OpenGVLab/InternVL2_5-8B",
    )
    parser.add_argument(
        "--georeason-model",
        type=str,
        default=None,
        help="HF hub id 或本地路径: GeoReason 模型路径（使用 Qwen2.5-VL）",
    )
    parser.add_argument(
        "--qids-file",
        type=Path,
        required=False,
        default=None,
        help="仅评估该 JSONL 文件中列出的 question_id（每行形如 {\"question_id\": \"...\"}）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        default=None,
        help="Limit the number of samples to evaluate (for debugging or quick testing)",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> List[dict]:
    samples: List[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_model(base_model: Path, lora_path: Path, device: torch.device):
    base_model = base_model.expanduser()

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # GeoChat加载逻辑与训练阶段保持一致
    model = GeoChatLlamaForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_safetensors=False
    )

    if hasattr(model, "get_vision_tower"):
        vision_tower = model.get_vision_tower()
        if hasattr(vision_tower, "is_loaded") and not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)

    # 只有当提供了 lora_path 且路径存在时才加载 LoRA
    if lora_path is not None:
        lora_path = lora_path.expanduser()
        if lora_path.exists():
            print(f"Loading LoRA weights from {lora_path}")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
        else:
            print(f"Warning: LoRA path {lora_path} does not exist, using base model only")
    else:
        print("No LoRA path provided, using base model only")

    model.to(device)
    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
        image_processor = processor.image_processor  # type: ignore[attr-defined]
    except Exception:
        # fall back to model vision tower if available
        if hasattr(model, "get_vision_tower"):
            vision_tower = model.get_vision_tower()
            if hasattr(vision_tower, "load_model") and not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch.float16)
            image_processor = vision_tower.image_processor
        else:
            raise RuntimeError("无法获取图像处理器，请确认模型包含视觉塔或提供 processor")

    return tokenizer, model, image_processor


def build_qwen2_5_vl(model_path: str, device: torch.device, lora_path: Path | None = None):
    """构建 Qwen2.5-VL 模型与处理器；若类缺失则回退到 AutoModel(trust_remote_code)。"""
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # 推荐使用 bfloat16，如果设备不支持则回退 float16
    dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda,
                                                                     'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = None
    if getattr(config, 'model_type', '') == 'qwen2_5_vl' and Qwen2_5_VLForConditionalGeneration is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            # torch_dtype=dtype,
            quantization_config=bnb_config,
            device_map={"": device},
            # attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    else:
        # 回退路径：使用 AutoModel 依赖 trust_remote_code 正确映射
        model = AutoModel.from_pretrained(
            model_path,
            # torch_dtype=dtype,
            quantization_config=bnb_config,
            # attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    # 可选加载 LoRA
    if lora_path is not None:
        lora_path = lora_path.expanduser()
        if lora_path.exists():
            print(f"Loading LoRA weights for Qwen2.5-VL from {lora_path}")
            try:
                model = PeftModel.from_pretrained(model, lora_path)
                # 合并以获得纯基座权重，推理更高效；若不支持则保留为 PEFT 包装
                try:
                    model = model.merge_and_unload()
                    print("LoRA merged and unloaded for inference.")
                except Exception as e:
                    print(f"Warn: merge_and_unload failed, keep PEFT adapters active. Detail: {e}")
            except Exception as e:
                print(f"Warn: Failed to load LoRA from {lora_path}: {e}. Continue with base model.")
        else:
            print(f"Warn: LoRA path {lora_path} does not exist, using base model only.")

    # model.to(device)
    model.eval()
    return processor, model


def build_qwen3_vl(model_path: str, device: torch.device, lora_path: Path | None = None):
    """构建 Qwen3-VL 模型与处理器，使用本地文件路径"""
    # 确保使用本地文件路径，展开用户路径并解析为绝对路径
    model_path = os.path.expanduser(model_path)
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise ValueError(
            f"Model path does not exist: {model_path}. Please ensure the path is correct and points to a local directory.")

    print(f"Loading Qwen3-VL from local path: {model_path}...")

    # 使用本地文件加载，不尝试从 HuggingFace Hub 下载
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True  # 强制使用本地文件
    )
    processor.tokenizer.padding_side = "left"

    # 推荐使用 bfloat16
    dtype = torch.bfloat16 if (torch.cuda.is_available() and hasattr(torch.cuda,
                                                                     'is_bf16_supported') and torch.cuda.is_bf16_supported()) else torch.float16

    # attn_implementation = None
    # try:
    #     import flash_attn
    #     attn_implementation = "flash_attention_2"
    #     print(f"Using flash_attention_2 for faster inference (Qwen3-VL)")
    # except ImportError:
    #     attn_implementation = "sdpa"
    #     print(f"flash-attn not available, using sdpa attention")

    # 尝试使用 Qwen3VLForConditionalGeneration（如果有的话），否则使用 AutoModel
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    if getattr(config, 'model_type', '') == 'qwen3_vl' and Qwen3VLForConditionalGeneration is not None:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,
        )
    else:
        # 回退路径：使用 AutoModel 依赖 trust_remote_code 正确映射
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True,  # 强制使用本地文件
        )

    # 可选加载 LoRA
    if lora_path is not None:
        lora_path = lora_path.expanduser()
        if lora_path.exists():
            print(f"Loading LoRA weights for Qwen3-VL from {lora_path}")
            try:
                model = PeftModel.from_pretrained(model, lora_path)
                try:
                    model = model.merge_and_unload()
                    print("LoRA merged and unloaded for inference.")
                except Exception as e:
                    print(f"Warn: merge_and_unload failed, keep PEFT adapters active. Detail: {e}")
            except Exception as e:
                print(f"Warn: Failed to load LoRA from {lora_path}: {e}. Continue with base model.")
        else:
            print(f"Warn: LoRA path {lora_path} does not exist, using base model only.")

    model.to(device)
    model.eval()
    return processor, model


def build_rs_llava(model_path: Path, lora_path: Path, device: torch.device):
    if not HAS_RS_LLAVA:
        raise ImportError("RS-LLaVA module not loaded.")

    rs_disable_torch_init()

    # RS-LLaVA 是 LoRA 模型，需要指定 Base Model
    # 参考 src/RS-LLaVA/README.md
    # model_path: 'BigData-KSU/RS-llava-v1.5-7b-LoRA' (LoRA权重)
    # model_base: 'Intel/neural-chat-7b-v3-3' (基座模型)

    # 自动推断本地路径
    # 用户提供的 model_path 参数应指向 LoRA 目录（例如 /mnt/nvme1/wj/Model/RS-llava-v1.5-7b-LoRA）
    model_path_str = str(model_path.expanduser())
    model_name = rs_get_model_name_from_path(model_path_str)

    # 硬编码 Base Model 路径 (根据用户请求和本地环境推断)
    # 如果加载的是已合并的模型(Merged)，则不需要 Base Model
    if "Merged" in model_path_str:
        model_base = None
        print("Detected Merged model, setting model_base to None.")
    else:
        model_base = "/mnt/nvme1/wj/Model/neural-chat-7b-v3-3"
        if not os.path.exists(model_base):
            print(
                f"Warning: Base model path {model_base} does not exist. Trying HF Hub ID 'Intel/neural-chat-7b-v3-3'...")
            model_base = 'Intel/neural-chat-7b-v3-3'

    print(f"Loading RS-LLaVA from:")
    print(f"  Model Path: {model_path_str}")
    print(f"  Base Model: {model_base}")

    tokenizer, model, image_processor, context_len = rs_load_pretrained_model(
        model_path=model_path_str,
        model_base=model_base,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
        device=device
    )

    # Fix for generation: decoder-only models need left padding
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # 注意：rs_load_pretrained_model 在 model_base 不为空时，会自动加载 base model 并应用 model_path 中的 LoRA
    # 所以通常不需要再额外合并一次 LoRA，除非有第二层 LoRA (args.lora_path)

    # 尝试加载外部 LoRA (也就是 args.lora_path，如果是微调后的权重)
    if lora_path is not None:
        lora_path_str = str(lora_path.expanduser())
        if os.path.exists(lora_path_str):
            print(f"Loading additional LoRA weights for RS-LLaVA from {lora_path_str}")
            model = PeftModel.from_pretrained(model, lora_path_str)
            model = model.merge_and_unload()
            print("Additional LoRA merged and unloaded.")
        else:
            print(f"Warning: Additional LoRA path {lora_path_str} does not exist.")

    return tokenizer, model, image_processor


def build_internvl2_5(model_path: str, device: torch.device, lora_path: Path | None = None):
    """构建 InternVL2.5 模型与处理器"""
    print(f"Loading InternVL2.5 from {model_path}...")
    model_path = os.path.expanduser(model_path)

    # InternVL2.5 推荐 torch.bfloat16
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    )

    if lora_path is not None:
        lora_path = lora_path.expanduser()
        if lora_path.exists():
            print(f"Loading LoRA weights for InternVL2.5 from {lora_path}")
            try:
                model = PeftModel.from_pretrained(model, lora_path)
                try:
                    model = model.merge_and_unload()
                    print("LoRA merged and unloaded for inference.")
                except Exception as e:
                    print(f"Warn: merge_and_unload failed, keep PEFT adapters active. Detail: {e}")
            except Exception as e:
                print(f"Warn: Failed to load LoRA from {lora_path}: {e}. Continue with base model.")
        else:
            print(f"Warn: LoRA path {lora_path} does not exist, using base model only.")

    model = model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    return tokenizer, model


def build_georeason(model_path: str, device: torch.device, lora_path: Path | None = None):
    """构建 GeoReason 模型（基于 Qwen2.5-VL）"""
    if Qwen2_5_VLForConditionalGeneration is None:
        raise ImportError(
            "Qwen2_5_VLForConditionalGeneration not available. Please install transformers with Qwen2.5-VL support.")

    print(f"Loading GeoReason model from {model_path}...")
    model_path = os.path.expanduser(model_path)

    # GeoReason 使用 bfloat16 和 flash_attention_2
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # 不使用 device_map="auto"，避免模型被分配到多个设备导致设备不匹配
    # 如果使用分布式，每个进程应该只使用一个 GPU
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map=None,  # 不使用自动设备映射，手动控制设备
        trust_remote_code=True
    )

    if lora_path is not None:
        lora_path = lora_path.expanduser()
        if lora_path.exists():
            print(f"Loading LoRA weights for GeoReason from {lora_path}")
            try:
                model = PeftModel.from_pretrained(model, lora_path)
                try:
                    model = model.merge_and_unload()
                    print("LoRA merged and unloaded for inference.")
                except Exception as e:
                    print(f"Warn: merge_and_unload failed, keep PEFT adapters active. Detail: {e}")
            except Exception as e:
                print(f"Warn: Failed to load LoRA from {lora_path}: {e}. Continue with base model.")
        else:
            print(f"Warn: LoRA path {lora_path} does not exist, using base model only.")

    # 手动将模型移到指定设备
    model = model.to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    return processor, model


def extract_georeason_content(text: str) -> Tuple[str, str]:
    """
    从 GeoReason 输出中提取 <think> 和 <answer> 内容

    Returns:
        (think_content, answer_content) tuple
    """
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    answer_content = answer_match.group(1).strip() if answer_match else text.strip()
    return think_content, answer_content


def prepare_prompt(text: str, model) -> str:
    if model.config.mm_use_im_start_end:
        replace_token = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}"
    else:
        replace_token = DEFAULT_IMAGE_TOKEN
    return text.replace("<image>", replace_token)


def collate_batch(batch, tokenizer, model, image_processor, image_dir: Path, conv_mode: str, device: torch.device):
    prompts: List[str] = []
    images: List[Image.Image] = []
    question_ids: List[str] = []

    for sample in batch:
        message = sample["messages"][0]
        text = prepare_prompt(message["content"], model)

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        prompts.append(prompt)

        # 兼容 images 字段为 str 或 list 的情况
        images_field = sample["images"]
        if isinstance(images_field, list):
            image_rel_path = images_field[0]
        else:  # str
            image_rel_path = images_field

        image_path = image_dir / image_rel_path
        images.append(Image.open(image_path).convert("RGB"))

        # 同样处理 question_id 字段
        qid_field = sample.get("question_id", image_rel_path)
        if isinstance(qid_field, list):
            qid = qid_field[0]
        else:
            qid = qid_field
        question_ids.append(qid)

    tokenized = [
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        for prompt in prompts
    ]

    # 修改这里：使用 tokenizer 的 pad 功能而不是 pad_sequence
    # 先将 tensor 转为 list
    tokenized_lists = [t.tolist() for t in tokenized]

    # 使用 tokenizer 进行左填充
    padded = tokenizer.pad(
        {'input_ids': tokenized_lists},
        padding=True,
        return_tensors='pt'
    )

    input_ids = padded['input_ids'].to(device)
    attention_mask = padded['attention_mask'].to(device)

    # 参照 batch_geochat_vqa.py 修改图像预处理逻辑
    pixel_values = image_processor.preprocess(
        images,
        crop_size={'height': 504, 'width': 504},
        size={'shortest_edge': 504},
        return_tensors='pt'
    )['pixel_values']
    pixel_values = pixel_values.to(device=device, dtype=torch.float16)
    # print(f"pixel_values.shape: {pixel_values.shape}")

    return input_ids, attention_mask, pixel_values, question_ids


@torch.inference_mode()
def generate_predictions(
        samples: List[dict],
        indices: List[int],
        tokenizer,
        model,
        image_processor,
        image_dir: Path,
        conv_mode: str,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        rank: int,
):
    results = []

    total = len(indices)
    if total == 0:
        return results

    progress = trange(0, total, batch_size, desc="Generating", disable=rank != 0)
    for start in progress:
        batch_indices = indices[start: start + batch_size]
        batch = [samples[i] for i in batch_indices]
        input_ids, attention_mask, pixel_values, question_ids = collate_batch(
            batch, tokenizer, model, image_processor, image_dir, conv_mode, device
        )

        stopping_criteria = KeywordsStoppingCriteria(
            keywords=[conv_templates[conv_mode].sep if conv_mode in conv_templates else "###"],
            tokenizer=tokenizer,
            input_ids=input_ids,
        )

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

        generated_text = tokenizer.batch_decode(generated[:, input_ids.size(1):], skip_special_tokens=True)

        for sample_idx, sample, answer in zip(batch_indices, batch, generated_text):
            # 兼容 question_id 字段为 str 或 list 的情况
            qid_field = sample.get("question_id")
            if isinstance(qid_field, list):
                qid = qid_field[0] if qid_field else None
            else:
                qid = qid_field

            results.append(
                {
                    "index": sample_idx,
                    "question_id": qid,
                    "ground_truth": sample.get("ground_truth"),
                    "answer": answer.strip(),
                }
            )

    return results


@torch.inference_mode()
def generate_predictions_qwen2_vl(
        samples: List[dict],
        indices: List[int],
        processor,  # AutoProcessor
        model,  # Qwen2.5-VL 模型
        image_dir: Path,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        rank: int,
):
    """使用 Qwen2.5-VL 进行批量多模态生成，支持多图/纯文本消息。

    逻辑参照官方示例：
    1. 将每个样本构造成 chat messages 列表（支持多张图片）。
    2. apply_chat_template 得到 texts。
    3. 使用 process_vision_info 得到 images / videos 输入。
    4. processor(text=texts, images=..., videos=..., padding=True) 得到批量张量。
    5. model.generate 并裁剪掉原始输入长度部分，仅保留新生成 token。
    """
    results: List[dict] = []
    total = len(indices)
    if total == 0:
        return results

    progress = trange(0, total, batch_size, desc="Generating(Qwen2.5-VL)", disable=rank != 0)
    for start in progress:
        batch_indices = indices[start:start + batch_size]
        batch = [samples[i] for i in batch_indices]

        # 构造当前 batch 的消息列表（每个元素是一个样本的 messages 列表）
        batch_messages: List[List[dict]] = []
        question_ids: List[str] = []
        for sample in batch:
            # 数据集中可能字段结构为 {"messages": [ {"role": "user", "content": "..."} ], "images": ... }
            raw_messages = sample.get("messages")
            # 收集 question_id
            qid_field = sample.get("question_id")
            if isinstance(qid_field, list):
                qid = qid_field[0] if qid_field else ""
            else:
                qid = qid_field if qid_field is not None else ""
            question_ids.append(qid)

            # 解析图像列表
            img_field = sample.get("images", [])
            if isinstance(img_field, list):
                img_list = img_field
            else:
                img_list = [img_field]

            # 取第一条用户文本（兼容原结构）
            if raw_messages and isinstance(raw_messages, list):
                # 假设第一条是用户内容
                first = raw_messages[0]
                raw_text = first.get("content", "") if isinstance(first, dict) else str(first)
            else:
                raw_text = sample.get("text", "")

            # 移除可能的 <image> 占位符，改用结构化多模态 content
            raw_text_clean = raw_text.replace("<image>", "").strip()

            content_items: List[dict] = []
            for rel_path in img_list:
                abs_path = (image_dir / rel_path).expanduser().resolve()
                content_items.append({"type": "image", "image": f"file://{abs_path}"})
            # 添加文本部分（允许为空字符串）
            content_items.append({"type": "text", "text": raw_text_clean or ""})

            # 可选：如果有系统消息（例如 sample["system"] 或 sample["system_prompt"]）
            system_prompt = sample.get("system") or sample.get("system_prompt")
            messages: List[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_items})

            batch_messages.append(messages)

        # 应用模板得到文本序列
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        # 处理视觉信息（官方工具），若不可用则做简单回退
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(batch_messages)
        else:
            # 回退：只收集图片路径 -> 由 processor 重新处理
            # 注意：此路径下 pixel 预处理可能与官方工具不同，建议安装 qwen_vl_utils
            image_inputs = []
            for m in batch_messages:
                imgs = []
                for msg in m:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image":
                                imgs.append(part.get("image"))
                image_inputs.append(imgs)
            video_inputs = None

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )

        # 裁剪掉原始输入长度，只保留新增 tokens
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for sample_idx, answer, qid in zip(batch_indices, output_texts, question_ids):
            results.append({
                "index": sample_idx,
                "question_id": qid,
                "ground_truth": samples[sample_idx].get("ground_truth"),
                "answer": answer.strip(),
            })

    return results


@torch.inference_mode()
def generate_predictions_qwen3_vl(
        samples: List[dict],
        indices: List[int],
        processor,  # AutoProcessor
        model,  # Qwen3-VL 模型
        image_dir: Path,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        rank: int,
):
    """使用 Qwen3-VL 进行批量多模态生成 (Reference official docs)"""
    results: List[dict] = []
    total = len(indices)
    if total == 0:
        return results

    progress = trange(0, total, batch_size, desc="Generating(Qwen3-VL)", disable=rank != 0)
    for start in progress:
        batch_indices = indices[start:start + batch_size]
        batch = [samples[i] for i in batch_indices]

        # 构造当前 batch 的消息列表
        batch_messages: List[List[dict]] = []
        question_ids: List[str] = []
        for sample in batch:
            raw_messages = sample.get("messages")
            qid_field = sample.get("question_id")
            if isinstance(qid_field, list):
                qid = qid_field[0] if qid_field else ""
            else:
                qid = qid_field if qid_field is not None else ""
            question_ids.append(qid)

            img_field = sample.get("images", [])
            if isinstance(img_field, list):
                img_list = img_field
            else:
                img_list = [img_field]

            if raw_messages and isinstance(raw_messages, list):
                first = raw_messages[0]
                raw_text = first.get("content", "") if isinstance(first, dict) else str(first)
            else:
                raw_text = sample.get("text", "")

            raw_text_clean = raw_text.replace("<image>", "").strip()

            content_items: List[dict] = []
            for rel_path in img_list:
                abs_path = (image_dir / rel_path).expanduser().resolve()
                content_items.append({"type": "image", "image": f"file://{abs_path}"})

            content_items.append({"type": "text", "text": raw_text_clean or ""})

            system_prompt = sample.get("system") or sample.get("system_prompt")
            messages: List[dict] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content_items})

            batch_messages.append(messages)

        # Apply chat template
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        # Process vision info
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(batch_messages)
        else:
            # Fallback
            image_inputs = []
            for m in batch_messages:
                imgs = []
                for msg in m:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image":
                                imgs.append(part.get("image"))
                image_inputs.append(imgs)
            video_inputs = None

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for sample_idx, answer, qid in zip(batch_indices, output_texts, question_ids):
            results.append({
                "index": sample_idx,
                "question_id": qid,
                "ground_truth": samples[sample_idx].get("ground_truth"),
                "answer": answer.strip(),
            })

    return results


def generate_predictions_internvl2_5(
        samples: List[dict],
        indices: List[int],
        tokenizer,
        model,
        image_dir: Path,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        rank: int,
):
    results: List[dict] = []
    total = len(indices)
    if total == 0:
        return results

    # InternVL .chat() 接口通常单次调用，为简化起见，这里按 batch_size=1 处理或尝试手动批处理
    # 官方示例通常是单张图 chat。如果要利用 model.chat，我们只能逐个处理。
    # 为了效率，我们尽量使用 batch_size=1 或者看是否能用 generate 批量。
    # InternVL2.5 的 chat 方法内部处理较多（添加 special tokens, pixel_values 形状变化等），批量化较复杂。
    # 这里我们采用 batch_size 循环，但实际上内部逐一处理（简化实现），或者如果 batch_size=1 则无所谓。
    # 用户可以在参数中指定 batch_size=1。

    progress = trange(0, total, desc="Generating(InternVL2.5)", disable=rank != 0)
    for i in progress:
        idx = indices[i]
        sample = samples[idx]

        # 收集 question_id
        qid_field = sample.get("question_id")
        if isinstance(qid_field, list):
            qid = qid_field[0] if qid_field else ""
        else:
            qid = qid_field if qid_field is not None else ""

        # 解析图像
        img_field = sample.get("images", [])
        if isinstance(img_field, list):
            img_list = img_field
        else:
            img_list = [img_field]

        pixel_values = None
        if img_list:
            # 暂时只支持单图，如果有多图，InternVL 也支持，但需要拼接 pixel_values
            # 这里实现单图逻辑，多图需参考官方多图示例
            # 如果是多图，需要把多张图的 pixel_values 拼起来 (dim=0)
            pixel_values_list = []
            for rel_path in img_list:
                image_path = image_dir / rel_path
                image = Image.open(image_path).convert("RGB")
                pv = load_image_internvl(image, max_num=12).to(torch.bfloat16).to(device)
                pixel_values_list.append(pv)
            pixel_values = torch.cat(pixel_values_list, dim=0)

        # 获取文本问题
        raw_messages = sample.get("messages")
        if raw_messages and isinstance(raw_messages, list):
            first = raw_messages[0]
            question = first.get("content", "") if isinstance(first, dict) else str(first)
        else:
            question = sample.get("text", "")

        question = question.replace("<image>", "<image>\n")  # 确保有 <image> 标记

        generation_config = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0.0, temperature=temperature)

        # 调用 model.chat
        # 注意：model.chat 是单条数据的 helper
        try:
            answer = model.chat(tokenizer, pixel_values, question, generation_config)
        except Exception as e:
            print(f"Error generating for index {idx}: {e}")
            answer = ""

        results.append({
            "index": idx,
            "question_id": qid,
            "ground_truth": sample.get("ground_truth"),
            "answer": answer.strip(),
        })

    return results


@torch.inference_mode()
def generate_predictions_georeason(
        samples: List[dict],
        indices: List[int],
        processor,  # AutoProcessor
        model,  # Qwen2.5-VL 模型
        image_dir: Path,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        rank: int,
):
    """使用 GeoReason (Qwen2.5-VL) 进行批量生成，支持 <think> 和 <answer> 提取"""
    results: List[dict] = []
    total = len(indices)
    if total == 0:
        return results

    progress = trange(0, total, batch_size, desc="Generating(GeoReason)", disable=rank != 0)
    for start in progress:
        batch_indices = indices[start:start + batch_size]
        batch = [samples[i] for i in batch_indices]

        # 构造当前 batch 的消息列表
        batch_messages: List[List[dict]] = []
        question_ids: List[str] = []

        for sample in batch:
            # 收集 question_id
            qid_field = sample.get("question_id")
            if isinstance(qid_field, list):
                qid = qid_field[0] if qid_field else ""
            else:
                qid = qid_field if qid_field is not None else ""
            question_ids.append(qid)

            # 解析图像列表
            img_field = sample.get("images", [])
            if isinstance(img_field, list):
                img_list = img_field
            else:
                img_list = [img_field]

            # 获取问题文本
            raw_messages = sample.get("messages")
            if raw_messages and isinstance(raw_messages, list):
                first = raw_messages[0]
                raw_text = first.get("content", "") if isinstance(first, dict) else str(first)
            else:
                raw_text = sample.get("text", "")

            # 移除可能的 <image> 占位符
            raw_text_clean = raw_text.replace("<image>", "").strip()

            # GeoReason 特殊格式：添加 <think> 和 <answer> 提示
            new_question = f"{raw_text_clean}\nPlease provide reasoning in <think> tags, and the correct answer in <answer> tags. Please reason step by step. Format strictly as <think>...</think><answer>...</answer>"

            # 构造消息
            content_items: List[dict] = []
            for rel_path in img_list:
                abs_path = (image_dir / rel_path).expanduser().resolve()
                content_items.append({"type": "image", "image": f"file://{abs_path}"})
            content_items.append({"type": "text", "text": new_question})

            messages: List[dict] = [
                {
                    "role": "system",
                    "content": "You are an expert in the field of remote sensing. You can analyze problems by thinking and finally provide accurate answers."
                },
                {
                    "role": "user",
                    "content": content_items
                }
            ]

            batch_messages.append(messages)

        # 应用模板得到文本序列
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]

        # 处理视觉信息
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(batch_messages)
        else:
            # 回退：只收集图片路径
            image_inputs = []
            for m in batch_messages:
                imgs = []
                for msg in m:
                    if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                        for part in msg["content"]:
                            if part.get("type") == "image":
                                imgs.append(part.get("image"))
                image_inputs.append(imgs)
            video_inputs = None

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )

        # 裁剪掉原始输入长度，只保留新增 tokens
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for sample_idx, output_text, qid in zip(batch_indices, output_texts, question_ids):
            # 提取 <think> 和 <answer>
            think_content, answer_content = extract_georeason_content(output_text)

            # 保存 answer_content 作为主要答案，think_content 作为额外信息
            results.append({
                "index": sample_idx,
                "question_id": qid,
                "ground_truth": samples[sample_idx].get("ground_truth"),
                "answer": answer_content.strip(),  # 主要答案
                "think": think_content.strip(),  # 推理过程（可选）
            })

    return results


def collate_batch_rs_llava(batch, tokenizer, model, image_processor, image_dir: Path, conv_mode: str,
                           device: torch.device):
    input_ids_list = []
    images_list = []
    question_ids = []

    for sample in batch:
        message = sample["messages"][0]
        text = message["content"]

        # RS-LLaVA 使用的 template
        conv = rs_conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 读取图片
        images_field = sample["images"]
        if isinstance(images_field, list):
            image_rel_path = images_field[0]
        else:
            image_rel_path = images_field

        image_path = image_dir / image_rel_path
        image = Image.open(image_path).convert("RGB")
        images_list.append(image)

        # tokenization
        input_ids = rs_tokenizer_image_token(prompt, tokenizer, RS_IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids_list.append(input_ids)

        qid_field = sample.get("question_id", image_rel_path)
        if isinstance(qid_field, list):
            qid = qid_field[0]
        else:
            qid = qid_field
        question_ids.append(qid)

    # Handle potentially missing pad_token_id
    pad_val = tokenizer.pad_token_id
    if pad_val is None:
        pad_val = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Manual Left Padding for Generation
    max_len = max(x.size(0) for x in input_ids_list)
    padded_input_ids = []

    for ids in input_ids_list:
        cur_len = ids.size(0)
        padding_len = max_len - cur_len
        if padding_len > 0:
            pad_tensor = torch.full((padding_len,), pad_val, dtype=ids.dtype)
            padded_ids = torch.cat([pad_tensor, ids], dim=0)
        else:
            padded_ids = ids
        padded_input_ids.append(padded_ids)

    input_ids_tensor = torch.stack(padded_input_ids).to(device)

    # Attention mask
    attention_mask = input_ids_tensor.ne(pad_val).to(device)

    # Image Processing
    image_tensor = rs_process_images(images_list, image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)

    return input_ids_tensor, attention_mask, image_tensor, question_ids


@torch.inference_mode()
def generate_predictions_rs_llava(
        samples: List[dict],
        indices: List[int],
        tokenizer,
        model,
        image_processor,
        image_dir: Path,
        conv_mode: str,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        device: torch.device,
        rank: int,
):
    results = []
    total = len(indices)
    if total == 0:
        return results

    progress = trange(0, total, batch_size, desc="Generating(RS-LLaVA)", disable=rank != 0)
    for start in progress:
        batch_indices = indices[start: start + batch_size]
        batch = [samples[i] for i in batch_indices]

        input_ids, attention_mask, images_tensor, question_ids = collate_batch_rs_llava(
            batch, tokenizer, model, image_processor, image_dir, conv_mode, device
        )

        # 确定 stop string
        conv = rs_conv_templates[conv_mode]
        # 简单判断：如果有 sep2 则用 sep2 (style TWO)，否则用 sep
        stop_str = conv.sep2 if hasattr(conv, 'sep2') and conv.sep2 else conv.sep
        keywords = [stop_str]
        stopping_criteria = RSKeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            attention_mask=attention_mask,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id
        )

        # decode
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

        for sample_idx, answer, qid in zip(batch_indices, outputs, question_ids):
            results.append({
                "index": sample_idx,
                "question_id": qid,
                "ground_truth": samples[sample_idx].get("ground_truth"),
                "answer": answer.strip()
            })

    return results


def setup_distributed() -> Tuple[int, int, int]:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()

    rank, world_size, local_rank = setup_distributed()

    if world_size == 1:
        # 兼容单卡：若用户未手动指定CUDA则默认使用可用GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    disable_torch_init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_dataset(args.dataset)

    # 可选：按 question_id 文件进行过滤，仅评估指定的样本
    filtered_indices = None
    if args.qids_file is not None:
        qids_path: Path = args.qids_file.expanduser()
        if qids_path.exists():
            # 读取 JSONL 文件中的 question_id
            qids_set = set()
            with qids_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        qid = obj.get("question_id")
                        if qid is not None:
                            qids_set.add(str(qid))
                    except Exception:
                        continue

            def _get_qid(sample: dict) -> str:
                q = sample.get("question_id")
                if isinstance(q, list):
                    return str(q[0]) if q else ""
                return str(q) if q is not None else ""

            filtered_indices = [i for i, s in enumerate(samples) if _get_qid(s) in qids_set]
            if len(filtered_indices) == 0:
                print(f"Warn: No samples matched question_ids from {qids_path}")
            else:
                print(f"Filter: {len(filtered_indices)} / {len(samples)} samples matched provided question_ids")
        else:
            print(f"Warn: qids-file not found: {qids_path}, skip filtering")

    if args.limit is not None:
        if filtered_indices is None:
            filtered_indices = list(range(len(samples)))
        filtered_indices = filtered_indices[:args.limit]

    # 分布式下做条带划分；若启用过滤，则对过滤后的索引做条带划分
    if filtered_indices is None:
        indices = list(range(rank, len(samples), world_size)) if world_size > 1 else list(range(len(samples)))
    else:
        if world_size > 1:
            # 将过滤后的索引按 rank 取步长切片
            indices = filtered_indices[rank::world_size]
        else:
            indices = filtered_indices

    if args.backend == "geochat":
        tokenizer, model, image_processor = build_model(args.base_model, args.lora_path, device)
        local_records = generate_predictions(
            samples=samples,
            indices=indices,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            image_dir=args.image_dir.expanduser(),
            conv_mode=args.conv_mode,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            rank=rank,
        )
    elif args.backend == "rs_llava":
        tokenizer, model, image_processor = build_rs_llava(args.base_model, args.lora_path, device)
        local_records = generate_predictions_rs_llava(
            samples=samples,
            indices=indices,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            image_dir=args.image_dir.expanduser(),
            conv_mode=args.conv_mode,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            rank=rank,
        )
    elif args.backend == "internvl2_5":
        tokenizer, model = build_internvl2_5(args.internvl_model, device, args.lora_path)
        local_records = generate_predictions_internvl2_5(
            samples=samples,
            indices=indices,
            tokenizer=tokenizer,
            model=model,
            image_dir=args.image_dir.expanduser(),
            batch_size=args.batch_size,  # Note: InternVL implementaion above ignores batch_size and does 1 by 1
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            rank=rank,
        )
    elif args.backend == "georeason":
        if args.georeason_model is None:
            raise ValueError("--georeason-model must be specified when using --backend=georeason")
        processor, model = build_georeason(args.georeason_model, device, args.lora_path)
        local_records = generate_predictions_georeason(
            samples=samples,
            indices=indices,
            processor=processor,
            model=model,
            image_dir=args.image_dir.expanduser(),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            rank=rank,
        )
    elif args.backend == "qwen3vl":
        processor, model = build_qwen3_vl(args.qwen3_vl_model, device, args.lora_path)
        # Use dedicated Qwen3-VL generation function
        local_records = generate_predictions_qwen3_vl(
            samples=samples,
            indices=indices,
            processor=processor,
            model=model,
            image_dir=args.image_dir.expanduser(),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            rank=rank,
        )
    else:  # qwen2_5_vl
        processor, model = build_qwen2_5_vl(args.qwen2_vl_model, device, args.lora_path)
        local_records = generate_predictions_qwen2_vl(
            samples=samples,
            indices=indices,
            processor=processor,
            model=model,
            image_dir=args.image_dir.expanduser(),
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
            rank=rank,
        )

    if world_size > 1:
        dist.barrier()
        gathered: List[List[dict]] = [None] * world_size  # type: ignore
        dist.all_gather_object(gathered, local_records)
    else:
        gathered = [local_records]

    if rank == 0:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        merged = [record for shard in gathered for record in shard]
        merged.sort(key=lambda r: r["index"])

        with output_path.open("w", encoding="utf-8") as writer:
            for record in merged:
                record = {k: v for k, v in record.items() if k != "index"}
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Predictions saved to {output_path}, total {len(merged)} records.")

    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()

