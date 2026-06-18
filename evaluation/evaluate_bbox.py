#!/usr/bin/env python3
"""
评估边界框预测结果的脚本
支持百分比格式的ground_truth和像素坐标格式的预测结果
"""

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from collections import defaultdict
from PIL import Image


def parse_ground_truth(gt_str: str) -> Optional[Tuple[float, float, float, float]]:
    """
    解析ground_truth格式: {<x1><y1><x2><y2>}
    返回百分比坐标 (x1, y1, x2, y2)，范围0-100
    """
    if not gt_str or not isinstance(gt_str, str):
        return None
    
    # 提取所有数字
    numbers = re.findall(r'<(\d+)>', gt_str)
    if len(numbers) != 4:
        return None
    
    try:
        x1, y1, x2, y2 = map(float, numbers)
        return (x1, y1, x2, y2)
    except ValueError:
        return None


def _parse_internvl3_style_answer(answer_str: str) -> Optional[Tuple[float, float, float, float]]:
    """
    解析 InternVL3 等模型的长文本 bbox 格式：
    - "Top-left corner (x1, y1): (150, 100)" 与 "Bottom-right corner (x2, y2): (250, 200)"
    - 或文末的 **(100, 100)** to **(300, 300)**、任意 (a, b) (c, d) 坐标对
    优先取「Top-left/top-left」与「Bottom-right/bottom-right」后首次出现的坐标对；
    否则取字符串中最后两个 (num, num) 坐标对作为 (x1,y1),(x2,y2)。
    """
    pair_re = re.compile(r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)')
    # 收集所有坐标对及其在字符串中的起始位置
    pairs_with_pos = [(m.start(), (float(m.group(1)), float(m.group(2)))) for m in pair_re.finditer(answer_str)]
    if len(pairs_with_pos) < 2:
        return None
    low = answer_str.lower()
    tl_pair = None
    br_pair = None
    for kw in ('top-left corner', 'top left corner', 'top-left', 'top left'):
        i = low.find(kw)
        if i != -1:
            # 第一个出现在该关键词之后的坐标对
            for pos, (a, b) in pairs_with_pos:
                if pos >= i:
                    tl_pair = (a, b)
                    break
            if tl_pair is not None:
                break
    for kw in ('bottom-right corner', 'bottom right corner', 'bottom-right', 'bottom right'):
        i = low.find(kw)
        if i != -1:
            for pos, (a, b) in pairs_with_pos:
                if pos >= i:
                    br_pair = (a, b)
                    break
            if br_pair is not None:
                break
    if tl_pair is not None and br_pair is not None and tl_pair != br_pair:
        (x1, y1), (x2, y2) = tl_pair, br_pair
        return (x1, y1, x2, y2)
    # 否则取最后两个坐标对（多数模型在文末给出最终 bbox）
    (_, (x1, y1)), (_, (x2, y2)) = pairs_with_pos[-2], pairs_with_pos[-1]
    return (x1, y1, x2, y2)


def parse_prediction_answer(answer_str: str) -> Optional[Tuple[float, float, float, float]]:
    """
    解析预测结果中的bbox_2d坐标
    格式: ```json\n[\n\t{"bbox_2d": [x1, y1, x2, y2], ...}\n]\n```
    或者直接是数字列表 [0.38, 67, 0.42, 69]
    或 InternVL3 等长文本: "Top-left corner: (150, 100)" 与 "Bottom-right corner: (250, 200)"
    返回: 原始坐标 (x1, y1, x2, y2)
    """
    if not answer_str:
        return None
    
    # 尝试解析 JSON
    try:
        # 有些输出包含 ```json ``` 包裹，有些没有
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', answer_str)
        json_str = json_match.group(1) if json_match else answer_str
        
        data = json.loads(json_str)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                bbox = data[0].get('bbox_2d')
                if bbox and len(bbox) == 4:
                    return tuple(map(float, bbox))
    except (json.JSONDecodeError, KeyError, ValueError, IndexError, AttributeError):
        pass

    # <box>[[x1,y1,x2,y2]]</box> 格式（InternVL 等）
    box_match = re.search(r'<box>\s*\[\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*\]\s*</box>', answer_str, re.IGNORECASE)
    if box_match:
        return tuple(map(float, box_match.group(1, 2, 3, 4)))

    # InternVL3 等：长文本中的 (x1,y1) 与 (x2,y2) 坐标对
    internvl_bbox = _parse_internvl3_style_answer(answer_str)
    if internvl_bbox is not None:
        return internvl_bbox

    # 备用：提取所有数字，取前4个（易受文中其他数字干扰）
    numbers = re.findall(r'(\d+(?:\.\d+)?)', answer_str)
    if len(numbers) >= 4:
        try:
            return tuple(map(float, numbers[:4]))
        except ValueError:
            pass
    
    return None

def normalize_mixed_coords(coords: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    将混合格式的归一化坐标 (0-1 和 0-100) 统一转换为 0-100 百分比格式
    规则：数值 <= 1.05 视为 0-1 范围，乘以 100；否则视为 0-100 范围，保持不变。
    """
    converted = []
    for val in coords:
        if val <= 1.05:
            converted.append(val * 100)
        else:
            converted.append(val)
    return tuple(converted)


def get_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    """
    从图片文件读取尺寸
    返回: (width, height) 或 None（如果无法读取）
    """
    if not image_path.exists():
        return None
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"Warning: Failed to read image size from {image_path}: {e}")
        return None


def _preload_image_sizes(
    image_dir: Path,
    image_filenames: List[str],
    image_size_cache: Dict[str, Tuple[int, int]],
    img_width: int = 504,
    img_height: int = 504,
    max_workers: int = 32,
) -> Tuple[int, int]:
    """
    预加载所有图片尺寸到 cache，避免主循环中逐条打开图片导致极慢。
    返回 (images_with_actual_size, images_with_default_size)。
    """
    unique = list(dict.fromkeys(image_filenames))  # 去重且保持顺序
    actual_count = 0
    default_count = 0

    def _load_one(fname: str) -> Tuple[str, Optional[Tuple[int, int]]]:
        path = image_dir / fname
        return (fname, get_image_size(path))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_load_one, fname): fname for fname in unique}
        for fut in as_completed(futures):
            fname, size = fut.result()
            if size:
                image_size_cache[fname] = size
                actual_count += 1
            else:
                image_size_cache[fname] = (img_width, img_height)
                default_count += 1
    return actual_count, default_count


def percent_to_pixel_on_processed_image(
    bbox_percent: Tuple[float, float, float, float], 
    original_width: int, 
    original_height: int,
    processed_size: int = 504
) -> Tuple[float, float, float, float]:
    """
    将百分比坐标（相对于原始图片）转换为预处理后图片（504x504）的像素坐标
    
    流程：
    1. 将百分比坐标转换为原始图片的像素坐标
    2. 考虑图片的expand2square处理（非正方形图片会被填充为正方形）
    3. 缩放到预处理后的尺寸（504x504）
    
    Args:
        bbox_percent: (x1%, y1%, x2%, y2%) 范围0-100，相对于原始图片
        original_width: 原始图片宽度
        original_height: 原始图片高度
        processed_size: 预处理后的图片尺寸（默认504）
    
    Returns:
        (x1, y1, x2, y2) 像素坐标，相对于预处理后的504x504图片
    """
    x1_pct, y1_pct, x2_pct, y2_pct = bbox_percent
    
    # Step 1: 转换为原始图片的像素坐标
    x1_orig = x1_pct / 100.0 * original_width
    y1_orig = y1_pct / 100.0 * original_height
    x2_orig = x2_pct / 100.0 * original_width
    y2_orig = y2_pct / 100.0 * original_height
    
    # Step 2: 应用expand2square（与训练/推理时的处理一致）
    # 如果是正方形，不需要填充；否则需要计算填充后的坐标
    if original_width == original_height:
        # 正方形图片，直接缩放
        scale = processed_size / original_width
        x1 = x1_orig * scale
        y1 = y1_orig * scale
        x2 = x2_orig * scale
        y2 = y2_orig * scale
    elif original_width > original_height:
        # 宽大于高，填充上下
        # 填充后的尺寸: (original_width, original_width)
        pad = (original_width - original_height) / 2
        # y坐标需要加上填充量
        scale = processed_size / original_width
        x1 = x1_orig * scale
        y1 = (y1_orig + pad) * scale
        x2 = x2_orig * scale
        y2 = (y2_orig + pad) * scale
    else:
        # 高大于宽，填充左右
        # 填充后的尺寸: (original_height, original_height)
        pad = (original_height - original_width) / 2
        # x坐标需要加上填充量
        scale = processed_size / original_height
        x1 = (x1_orig + pad) * scale
        y1 = y1_orig * scale
        x2 = (x2_orig + pad) * scale
        y2 = y2_orig * scale
    
    return (x1, y1, x2, y2)


def pixel_original_to_processed_image(
    bbox_orig_pixel: Tuple[float, float, float, float],
    original_width: int,
    original_height: int,
    processed_size: int = 504,
) -> Tuple[float, float, float, float]:
    """
    将「原始图片像素」坐标转换为预处理后图片（504x504）的像素坐标
    用于输出原图像素 bbox 的模型，与 percent_to_pixel_on_processed_image 的
    expand2square + scale 逻辑一致。
    """
    x1_orig, y1_orig, x2_orig, y2_orig = bbox_orig_pixel
    if original_width == original_height:
        scale = processed_size / original_width
        x1 = x1_orig * scale
        y1 = y1_orig * scale
        x2 = x2_orig * scale
        y2 = y2_orig * scale
    elif original_width > original_height:
        pad = (original_width - original_height) / 2
        scale = processed_size / original_width
        x1 = x1_orig * scale
        y1 = (y1_orig + pad) * scale
        x2 = x2_orig * scale
        y2 = (y2_orig + pad) * scale
    else:
        pad = (original_height - original_width) / 2
        scale = processed_size / original_height
        x1 = (x1_orig + pad) * scale
        y1 = y1_orig * scale
        x2 = (x2_orig + pad) * scale
        y2 = y2_orig * scale
    return (x1, y1, x2, y2)


def percent_to_pixel(bbox_percent: Tuple[float, float, float, float], 
                     img_width: int = 504, img_height: int = 504) -> Tuple[float, float, float, float]:
    """
    将百分比坐标转换为像素坐标（简单版本，用于向后兼容）
    bbox_percent: (x1%, y1%, x2%, y2%) 范围0-100
    返回: (x1, y1, x2, y2) 像素坐标
    """
    x1_pct, y1_pct, x2_pct, y2_pct = bbox_percent
    x1 = x1_pct / 100.0 * img_width
    y1 = y1_pct / 100.0 * img_height
    x2 = x2_pct / 100.0 * img_width
    y2 = y2_pct / 100.0 * img_height
    return (x1, y1, x2, y2)


def qwen3_coords_to_pixel(bbox_qwen3: Tuple[float, float, float, float],
                          img_width: int = 504, img_height: int = 504) -> Tuple[float, float, float, float]:
    """
    将qwen3模型的0-1000相对坐标转换为像素坐标
    bbox_qwen3: (x1, y1, x2, y2) 范围0-1000
    返回: (x1, y1, x2, y2) 像素坐标，范围0-504
    """
    x1_rel, y1_rel, x2_rel, y2_rel = bbox_qwen3
    x1 = x1_rel / 1000.0 * img_width
    y1 = y1_rel / 1000.0 * img_height
    x2 = x2_rel / 1000.0 * img_width
    y2 = y2_rel / 1000.0 * img_height
    return (x1, y1, x2, y2)


def calculate_iou(bbox1: Tuple[float, float, float, float], 
                  bbox2: Tuple[float, float, float, float]) -> float:
    """
    计算两个边界框的IoU (Intersection over Union)
    bbox格式: (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_accuracy_at_iou_threshold(iou_scores: List[float], threshold: float = 0.5) -> float:
    """计算在给定IoU阈值下的准确率"""
    if not iou_scores:
        return 0.0
    return sum(1 for iou in iou_scores if iou >= threshold) / len(iou_scores)


def load_metadata_negations(metadata_file: str) -> Dict[str, str]:
    """
    加载元数据文件，提取 {question_id: negation_type} 映射
    """
    print(f"Loading metadata from {metadata_file}...")
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        qid_to_negation = {}
        for item in data:
            qid = item.get('question_id')
            neg_type = item.get('negation_type')
            if qid is not None and neg_type:
                qid_to_negation[str(qid)] = neg_type
                
        print(f"Loaded {len(qid_to_negation)} negation types from metadata")
        return qid_to_negation
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return {}


def evaluate(pred_file: str, gt_file: str, image_dir: Optional[str] = None, 
             img_width: int = 504, img_height: int = 504,
             metadata_file: Optional[str] = None,
             pred_normalized: bool = False,
             model_type: Optional[str] = None):
    """
    评估边界框预测结果
    
    Args:
        pred_file: 预测结果文件路径 (JSONL格式)
        gt_file: ground truth文件路径 (JSONL格式，包含图片文件名)
        image_dir: 图片文件目录路径。如果提供，将从实际图片文件读取尺寸
        img_width: 图片宽度（像素），默认504。当image_dir为None时使用
        img_height: 图片高度（像素），默认504。当image_dir为None时使用
        metadata_file: 元数据文件路径（JSON列表格式），用于分析negation_type
        pred_normalized: 是否将预测结果视为归一化坐标(0-100范围)并进行缩放。默认为False。
        model_type: 模型类型。'qwen3'：将0-1000相对坐标转为像素；'internvl3'：将原图像素 bbox 转为 504 空间（需配合 --image-dir）。
    """
    # 加载 negation type 元数据
    qid_to_negation = {}
    if metadata_file:
        qid_to_negation = load_metadata_negations(metadata_file)

    # 加载ground truth，同时保存图片文件名信息
    gt_map = {}
    image_map = {}  # question_id -> image_filename
    print(f"Loading ground truth from {gt_file}...")
    with open(gt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                qid = item.get('question_id')
                gt = item.get('ground_truth')
                images = item.get('images', [])
                if qid and gt:
                    qid = str(qid)
                    gt_map[qid] = gt
                    # 获取图片文件名
                    if images:
                        if isinstance(images, list):
                            image_filename = images[0] if len(images) > 0 else None
                        else:
                            image_filename = images
                        if image_filename:
                            image_map[qid] = image_filename
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(gt_map)} ground truth entries")

    # 如果提供了图片目录，预加载所有图片尺寸（并行），避免主循环中逐条打开导致极慢
    image_size_cache = {}
    images_with_actual_size = 0
    images_with_default_size = 0
    if image_dir:
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            print(f"Warning: Image directory does not exist: {image_dir}")
            print("Falling back to default image size (504x504)")
            image_dir = None
        else:
            print(f"Using actual image sizes from: {image_dir}")
            all_filenames = list(image_map.values())
            if all_filenames:
                print(f"Preloading image sizes for {len(all_filenames)} unique images...")
                images_with_actual_size, images_with_default_size = _preload_image_sizes(
                    image_dir_path, all_filenames, image_size_cache,
                    img_width=img_width, img_height=img_height, max_workers=32,
                )
                print(f"Preload done: {images_with_actual_size} actual, {images_with_default_size} default.")

    # 统计数据
    total = 0
    valid = 0
    invalid_pred = 0
    missing_gt = 0
    iou_scores = []
    
    # Negation Type 统计
    # 结构: type -> {'ious': []}
    negation_stats = defaultdict(lambda: {'ious': []})
    
    invalid_records = []
    missing_records = []

    print(f"\nEvaluating predictions from {pred_file}...")
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                pred_item = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            qid = pred_item.get('question_id')
            answer = pred_item.get('answer', '')
            total += 1
            
            if not qid:
                missing_gt += 1
                missing_records.append({'qid': None, 'answer': answer[:100]})
                continue
            
            qid = str(qid)
            gt_str = gt_map.get(qid)
            
            if not gt_str:
                missing_gt += 1
                missing_records.append({'qid': qid, 'answer': answer[:100]})
                continue
            
            # 解析ground truth
            gt_bbox_percent = parse_ground_truth(gt_str)
            if not gt_bbox_percent:
                missing_gt += 1
                missing_records.append({'qid': qid, 'gt_str': gt_str, 'answer': answer[:100]})
                continue
            
            # 解析预测结果
            pred_bbox = parse_prediction_answer(answer)
            if not pred_bbox:
                # 对于无法解析的预测，视为 <0,0,0,0>
                invalid_pred += 1
                invalid_records.append({'qid': qid, 'answer': answer[:100]})
                pred_bbox = (0.0, 0.0, 0.0, 0.0)
            
            # 获取图片尺寸（已由预加载填入 image_size_cache，此处仅查表）
            if image_dir and qid in image_map:
                image_filename = image_map[qid]
                orig_width, orig_height = image_size_cache.get(
                    image_filename, (img_width, img_height)
                )
                # 使用实际图片尺寸转换ground truth到预处理后的504x504坐标
                gt_bbox_pixel = percent_to_pixel_on_processed_image(
                    gt_bbox_percent, orig_width, orig_height, processed_size=504
                )
            else:
                # 使用默认尺寸（假设图片已经是504x504）
                gt_bbox_pixel = percent_to_pixel(gt_bbox_percent, img_width, img_height)
                # 不使用image_dir时，不统计（因为所有都使用默认尺寸）
            
            # 将预测结果转换为像素坐标(0-504)
            # 假设模型输入的图片已经缩放到了 504*504 (含padding)
            # 注意：parse_prediction_answer 返回的是原始解析出的数值
            
            if model_type == 'qwen3':
                # qwen3模型输出0-1000的相对坐标，需要转换为0-504的像素坐标
                pred_bbox_pixel = qwen3_coords_to_pixel(pred_bbox, img_width, img_height)
            elif model_type == 'internvl3':
                # InternVL3 与 Qwen3 相同，输出 0-1000 相对坐标，转为 0-504 像素
                pred_bbox_pixel = qwen3_coords_to_pixel(pred_bbox, img_width, img_height)
            elif pred_normalized:
                # 开启归一化参数：即使这些数值看起来像像素，也强制视为归一化坐标进行处理
                # 第一步：处理混合格式 (0.39 -> 39)
                pred_bbox_norm = normalize_mixed_coords(pred_bbox)
                # 第二步：将 0-100 百分比坐标缩放到 0-504 像素坐标
                pred_bbox_pixel = percent_to_pixel(pred_bbox_norm, 504, 504)
            else:
                # 关闭归一化参数：直接视为像素坐标，不进行任何缩放或转换
                pred_bbox_pixel = pred_bbox
            
            # 计算IoU
            iou = calculate_iou(pred_bbox_pixel, gt_bbox_pixel)
            iou_scores.append(iou)
            valid += 1
            
            # 记录 Negation Type 统计
            if qid in qid_to_negation:
                neg_type = qid_to_negation[qid]
                negation_stats[neg_type]['ious'].append(iou)
            elif 'unknown' not in negation_stats and metadata_file:
                # 只有在提供了 metadata 文件但没找到 qid 时才记录 unknown
                negation_stats['unknown']['ious'].append(iou)
    
    # 计算评估指标
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    median_iou = np.median(iou_scores) if iou_scores else 0.0
    
    # 不同IoU阈值下的准确率
    acc_25 = calculate_accuracy_at_iou_threshold(iou_scores, 0.25)
    acc_50 = calculate_accuracy_at_iou_threshold(iou_scores, 0.5)
    acc_75 = calculate_accuracy_at_iou_threshold(iou_scores, 0.75)
    
    # 输出结果
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Total predictions: {total}")
    print(f"Valid predictions: {valid}")
    print(f"Invalid predictions (cannot parse bbox): {invalid_pred}")
    print(f"Missing ground truth: {missing_gt}")
    if image_dir:
        print(f"\nImage Size Statistics:")
        print(f"  Images with actual size loaded: {images_with_actual_size}")
        print(f"  Images using default size (504x504): {images_with_default_size}")
    print(f"\nIoU Metrics:")
    print(f"  Mean IoU: {mean_iou:.4f}")
    print(f"  Median IoU: {median_iou:.4f}")
    print(f"\nAccuracy at IoU Thresholds:")
    print(f"  IoU >= 0.25: {acc_25*100:.2f}%")
    print(f"  IoU >= 0.50: {acc_50*100:.2f}%")
    print(f"  IoU >= 0.75: {acc_75*100:.2f}%")
    
    # 输出 Negation Type 统计
    if negation_stats:
        print("\nAccuracy by Negation Type:")
        print(f"{'Negation Type':<25} {'Count':<8} {'Mean IoU':<10} {'Acc@0.5':<10}")
        print("-" * 60)
        
        # 按类型名称排序
        for neg_type in sorted(negation_stats.keys()):
            stats = negation_stats[neg_type]
            ious = stats['ious']
            count = len(ious)
            if count > 0:
                mean_iou_type = np.mean(ious)
                acc_50_type = calculate_accuracy_at_iou_threshold(ious, 0.5)
                print(f"{neg_type:<25} {count:<8} {mean_iou_type:.4f}     {acc_50_type*100:.2f}%")
    
    print("="*60)
    
    # 输出一些统计信息
    if iou_scores:
        print(f"\nIoU Distribution:")
        print(f"  Min: {min(iou_scores):.4f}")
        print(f"  25th percentile: {np.percentile(iou_scores, 25):.4f}")
        print(f"  75th percentile: {np.percentile(iou_scores, 75):.4f}")
        print(f"  Max: {max(iou_scores):.4f}")
    
    # 输出无效预测的示例（前10个）
    if invalid_records:
        print(f"\nSample invalid predictions (first 10):")
        for i, record in enumerate(invalid_records[:10]):
            print(f"  {i+1}. QID: {record['qid']}")
            print(f"     Answer: {record['answer']}")
    
    # 输出缺失ground truth的示例（前10个）
    if missing_records:
        print(f"\nSample missing ground truth (first 10):")
        for i, record in enumerate(missing_records[:10]):
            print(f"  {i+1}. QID: {record.get('qid', 'None')}")
            if 'gt_str' in record:
                print(f"     GT: {record['gt_str']}")
            print(f"     Answer: {record['answer']}")
    
    return {
        'total': total,
        'valid': valid,
        'invalid_pred': invalid_pred,
        'missing_gt': missing_gt,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'acc_25': acc_25,
        'acc_50': acc_50,
        'acc_75': acc_75,
        'iou_scores': iou_scores
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate bounding box predictions')
    parser.add_argument('--pred-file', type=str, required=True,
                        help='Path to prediction file (JSONL format)')
    parser.add_argument('--gt-file', type=str, required=True,
                        help='Path to ground truth file (JSONL format)')
    parser.add_argument('--image-dir', type=str, default=None,
                        help='Directory containing image files. If provided, actual image sizes will be used.')
    parser.add_argument('--img-width', type=int, default=504,
                        help='Image width in pixels (default: 504, used when --image-dir is not provided)')
    parser.add_argument('--img-height', type=int, default=504,
                        help='Image height in pixels (default: 504, used when --image-dir is not provided)')
    parser.add_argument('--metadata-file', type=str, default=None,
                        help='Path to metadata JSON file containing negation_type info')
    parser.add_argument('--pred-normalized', action='store_true',
                        help='Treat predictions as normalized coordinates (0-100 range) and scale to image size')
    parser.add_argument('--model-type', type=str, default=None,
                        help='Model type: "qwen3" or "internvl3" (both use 0-1000 relative coordinates -> pixel).')
    
    args = parser.parse_args()
    
    evaluate(args.pred_file, args.gt_file, args.image_dir, args.img_width, args.img_height, args.metadata_file, args.pred_normalized, args.model_type)
