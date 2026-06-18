#!/usr/bin/env python3
"""
使用 InternVL2.5-8B 对 Ref_VRS 指代式 bbox 评测数据进行推理并保存预测结果。

参照：
- src/InternVL/internvl_chat/eval/refcoco/evaluate_grounding.py（模型 chat + bbox 解析）
- src/evaluation_with_finetuned_model.py（InternVL2.5 加载与 JSONL 数据集格式）

用法示例：
  单卡：
  python evaluation/the7/eval_refvrs_internvl2_5.py \\
    --model /mnt/nvme1/wj/Model/InternVL2_5-8B \\
    --dataset data/the7/Ref_VRS/...jsonl \\
    --image-dir /path/to/Ref_VRS/images \\
    --output outputs/Ref_VRS_internvl2_5_predictions.jsonl

  多卡（参照 evaluate_grounding.py，按 rank 分片推理后 all_gather 合并）：
  torchrun --nproc_per_node=4 evaluation/the7/eval_refvrs_internvl2_5.py \\
    --model /mnt/nvme1/wj/Model/InternVL3-8B \\
    --dataset data/the7/Ref_VRS/...jsonl \\
    --image-dir /path/to/images \\
    --output outputs/Ref_VRS_internvl3_predictions.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 优先使用本地 InternVL 的 load_model_and_tokenizer，加载完整 InternVL 模型（含 .chat()）
_INTERNVL_CHAT_PATH = (Path(__file__).resolve().parent.parent.parent / "src" / "InternVL" / "internvl_chat")
if _INTERNVL_CHAT_PATH.exists() and str(_INTERNVL_CHAT_PATH) not in sys.path:
    sys.path.insert(0, str(_INTERNVL_CHAT_PATH))
try:
    from internvl.model import load_model_and_tokenizer as _internvl_load_model_and_tokenizer
    _HAS_INTERNVL_LOADER = True
except ImportError:
    _HAS_INTERNVL_LOADER = False


# ---------------------------------------------------------------------------
# 多 GPU 分布式（参照 evaluate_grounding.py：InferenceSampler + all_gather_object）
# ---------------------------------------------------------------------------
def _get_local_indices(total_size: int, world_size: int, rank: int):
    """当前 rank 负责的样本下标 [begin, end)。"""
    shard_size = total_size // world_size
    left = total_size % world_size
    shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
    begin = sum(shard_sizes[:rank])
    end = min(begin + shard_sizes[rank], total_size)
    return list(range(begin, end))


def _init_distributed():
    """若通过 torchrun 启动（WORLD_SIZE>1），初始化进程组并返回 (rank, world_size, local_rank)。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0, 1, 0
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
        )
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# ---------------------------------------------------------------------------
# 图像预处理（与 evaluation_with_finetuned_model 中 InternVL 逻辑一致）
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
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
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(target_aspect_ratio[0] * target_aspect_ratio[1]):
        box = (
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            (i % target_aspect_ratio[0] + 1) * image_size,
            (i // target_aspect_ratio[0] + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl(image, input_size=448, max_num=12):
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def build_internvl2_5(model_path: str, device: torch.device):
    """加载 InternVL2.5 模型与 tokenizer。优先用本地 InternVL 的 loader，得到带 .chat() 的完整模型。"""
    model_path = os.path.expanduser(model_path)
    print(f"Loading InternVL2.5 from {model_path}...")

    if _HAS_INTERNVL_LOADER:
        try:
            # 与 evaluate_grounding.py 一致：使用 internvl.model.load_model_and_tokenizer
            class _Args:
                checkpoint = model_path
                auto = False
                load_in_8bit = False
                load_in_4bit = False
            args = _Args()
            model, tokenizer = _internvl_load_model_and_tokenizer(args)
            model = model.eval().to(device)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if getattr(tokenizer, "pad_token", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("Loaded via internvl.model.load_model_and_tokenizer (full model with .chat()).")
            return tokenizer, model
        except Exception as e:
            print(f"internvl loader failed: {e}, falling back to AutoModel.")

    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    )
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def build_internvl3(model_path: str, device: torch.device):
    """加载 InternVL3 模型与 tokenizer（AutoModel + model.chat，与 evaluation_with_finetuned_model 一致）。"""
    model_path = os.path.expanduser(model_path)
    print(f"Loading InternVL3 from {model_path}...")
    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    )
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


# ---------------------------------------------------------------------------
# Prompt（与 evaluate_grounding.py 一致）
# ---------------------------------------------------------------------------
REFCOCO_PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{}</ref>"


def _extract_referring_sentence(content: str) -> str:
    """从 messages[0].content 中提取指代描述句，去掉 <image>、开头的 Detect、末尾的「请输出 bbox」等。"""
    s = content.strip()
    if s.startswith("<image>"):
        s = s[len("<image>"):].strip()
    if s.startswith("\n"):
        s = s.lstrip("\n").strip()
    # 去掉开头的 Detect
    if s.startswith("Detect "):
        s = s[7:].strip()
    elif s.startswith("Detect"):
        s = s[6:].strip()
    # 去掉常见结尾
    for suffix in (
        " Please output the bounding box coordinates.",
        " Please output the bounding box coordinates",
        "Please output the bounding box coordinates.",
        "Please output the bounding box coordinates",
    ):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
            break
    return s


# ---------------------------------------------------------------------------
# 数据集与推理
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def run_inference(
    samples: List[dict],
    tokenizer,
    model,
    image_dir: Path,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> List[dict]:
    results = []
    for sample in tqdm(samples, desc="InternVL Ref_VRS"):
        qid = sample.get("question_id")
        if isinstance(qid, list):
            qid = qid[0] if qid else ""
        img_list = sample.get("images", [])
        if isinstance(img_list, str):
            img_list = [img_list]
        raw_messages = sample.get("messages", [])
        if raw_messages and isinstance(raw_messages[0], dict):
            content = raw_messages[0].get("content", "")
        else:
            content = sample.get("text", "")
        sentence = _extract_referring_sentence(content)
        question = REFCOCO_PROMPT.format(sentence)
        question = "<image>\n" + question

        pixel_values = None
        if img_list and image_dir:
            image_path = image_dir / img_list[0]
            if not image_path.exists():
                results.append({
                    "question_id": qid,
                    "ground_truth": sample.get("ground_truth"),
                    "answer": "",
                })
                continue
            image = Image.open(image_path).convert("RGB")
            pv = load_image_internvl(image, max_num=12)
            pixel_values = pv.to(dtype=model.dtype, device=device)

        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
        try:
            answer = model.chat(
                tokenizer, pixel_values, question, generation_config
            )
            if isinstance(answer, (list, tuple)):
                answer = answer[0] if answer else ""
        except Exception as e:
            print(f"Error qid={qid}: {e}")
            answer = ""
        results.append({
            "question_id": qid,
            "ground_truth": sample.get("ground_truth"),
            "answer": (answer or "").strip(),
        })
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Ref_VRS referring bbox with InternVL2.5-8B"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/nvme1/wj/Model/InternVL3-8B",
        help="InternVL2.5 或 InternVL3 模型路径（如 InternVL3-8B）或 HF id",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/mnt/hanhc/TLM-main/data/the7/Ref_VRS/VRSBench_EVAL_referring_negated_polished_filtered_combined_original_question.jsonl"
        ),
        help="评测 JSONL 路径",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        required=True,
        help="图片目录（JSONL 中 images 如 P0003_0004.png 相对此目录；需包含 VRS 评测图片）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/Ref_VRS_internvl3_predictions.jsonl"),
        help="预测结果输出 JSONL 路径",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评测前 N 条样本，不指定则全部",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="推理完成后按 evaluate_grounding 方式做 IoU 评测（调用 evaluate_bbox）",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="可选，negation 元数据 JSON，用于按 negation_type 统计 IoU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = _init_distributed()
    args.dataset = args.dataset.expanduser().resolve()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")

    device = (
        torch.device(f"cuda:{local_rank}")
        if world_size > 1
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if rank == 0:
        print(f"World size: {world_size}, rank: {rank}, device: {device}")
    model_path_lower = args.model.lower()
    if "internvl3" in model_path_lower or "internvl3-8b" in model_path_lower:
        tokenizer, model = build_internvl3(args.model, device)
    else:
        tokenizer, model = build_internvl2_5(args.model, device)
    samples = load_dataset(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
        if rank == 0:
            print(f"Limited to first {len(samples)} samples (--limit {args.limit}).")
    # 多卡：当前 rank 只推理自己分片
    if world_size > 1:
        local_indices = _get_local_indices(len(samples), world_size, rank)
        samples_to_run = [samples[i] for i in local_indices]
        if rank == 0:
            print(f"Distributed: rank 0 has {len(samples_to_run)} samples (total {len(samples)}).")
    else:
        samples_to_run = samples
        local_indices = list(range(len(samples)))

    results = run_inference(
        samples_to_run,
        tokenizer,
        model,
        args.image_dir,
        device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if world_size > 1:
        for j, idx in enumerate(local_indices):
            results[j]["_global_index"] = idx
        torch.distributed.barrier()
        gather_list = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gather_list, results)
        merged = []
        for lst in gather_list:
            merged.extend(lst)
        merged.sort(key=lambda x: x["_global_index"])
        for r in merged:
            del r["_global_index"]
        results = merged

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Predictions saved to {args.output}, total {len(results)} records.")
        if getattr(args, "evaluate", False):
            _run_iou_evaluation(args)

    if world_size > 1:
        torch.distributed.barrier()


def _run_iou_evaluation(args):
    """调用 evaluation/the7/evaluate_bbox 做 IoU 评测（与 evaluate_grounding 一致）。"""
    _eval_dir = Path(__file__).resolve().parent
    if str(_eval_dir) not in sys.path:
        sys.path.insert(0, str(_eval_dir))
    from evaluate_bbox import evaluate

    pred_file = str(args.output)
    gt_file = str(args.dataset)
    image_dir = str(args.image_dir)
    metadata_file = str(args.metadata_file) if getattr(args, "metadata_file", None) else None
    model_type = "internvl3" if ("internvl3" in args.model.lower() or "internvl3-8b" in args.model.lower()) else None
    if not model_type:
        model_type = "internvl3"  # InternVL2.5 也常用 0-1000 坐标，可按需改为 None
    print("\nRunning IoU evaluation (evaluate_grounding style)...")
    evaluate(
        pred_file=pred_file,
        gt_file=gt_file,
        image_dir=image_dir,
        img_width=504,
        img_height=504,
        metadata_file=metadata_file,
        pred_normalized=False,
        model_type=model_type,
    )


if __name__ == "__main__":
    main()
