#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate FloodNet Complex_Counting predictions.
Input: JSONL with question_id, ground_truth, answer (e.g. outputs/FloodNet_Complex_Counting_vl_base_system_prompt_qwen.jsonl).
Extracts the first number from answer and compares with ground_truth; reports exact-match accuracy.
"""
import json
import re
from pathlib import Path


def extract_number(text) -> str:
    """Extract the first integer from text (e.g. '3', 'The answer is 5.' -> '5')."""
    if text is None:
        return ""
    s = (str(text).strip() if isinstance(text, str) else str(text)).strip()
    if not s:
        return ""
    # Try whole string as int first
    try:
        int(s)
        return s
    except ValueError:
        pass
    # First integer in string
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else ""


def normalize_gt(gt) -> str:
    """Normalize ground_truth to string of a number."""
    if gt is None:
        return ""
    s = str(gt).strip()
    try:
        int(s)
        return s
    except ValueError:
        return s


def evaluate(pred_path: str) -> None:
    pred_path = Path(pred_path)
    if not pred_path.exists():
        raise SystemExit(f"Prediction file not found: {pred_path}")

    total = 0
    correct = 0
    invalid = 0
    invalid_records = []

    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec.get("question_id")
            gt_raw = rec.get("ground_truth")
            answer = rec.get("answer", "")

            total += 1
            gt_label = normalize_gt(gt_raw)
            pred_label = extract_number(answer)

            if not pred_label:
                invalid += 1
                invalid_records.append({"qid": qid, "answer": answer})
                continue
            if pred_label == gt_label:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0
    valid_total = total - invalid
    valid_acc = (correct / valid_total * 100) if valid_total > 0 else 0

    print(f"File: {pred_path}")
    print(f"Total: {total}, Correct: {correct}, Invalid (no number): {invalid}")
    print(f"Accuracy (all): {acc:.2f}%")
    print(f"Valid Accuracy (excluding invalid): {valid_acc:.2f}%")
    if invalid_records:
        print(f"\nFirst 10 invalid predictions:")
        for r in invalid_records[:10]:
            print(f"  qid={r['qid']}, answer={r['answer']!r}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate FloodNet Complex_Counting predictions")
    parser.add_argument(
        "--pred",
        type=str,
        default="/mnt/hanhc/TLM-main/outputs/FloodNet_Complex_Counting_vl_base_system_prompt_qwen.jsonl",
        help="Path to prediction JSONL (question_id, ground_truth, answer)",
    )
    args = parser.parse_args()
    evaluate(args.pred)


if __name__ == "__main__":
    main()
