#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate FloodNet Yes/No predictions.
Input: JSONL with question_id, ground_truth, answer (e.g. outputs/FloodNet_yesno_vl_base_system_prompt_qwen.jsonl).
Compares answer (normalized to Yes/No) with ground_truth and reports accuracy.
"""
import json
import re
from pathlib import Path
from collections import defaultdict


def extract_yesno(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, (int, float)):
        text = str(text)
    s = (text if isinstance(text, str) else "").strip().lower()
    if not s:
        return ""
    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"
    if re.search(r"\byes\b", s):
        return "Yes"
    if re.search(r"\bno\b", s):
        return "No"
    return ""


def normalize_gt(gt) -> str:
    """Normalize ground_truth to Yes or No."""
    if gt is None:
        return ""
    s = (str(gt).strip()).lower()
    if s in ("yes", "y"):
        return "Yes"
    if s in ("no", "n"):
        return "No"
    return str(gt).strip()  # keep as-is if not yes/no


def evaluate(pred_path: str) -> None:
    pred_path = Path(pred_path)
    if not pred_path.exists():
        raise SystemExit(f"Prediction file not found: {pred_path}")

    total = 0
    correct = 0
    invalid = 0  # answer could not be parsed as Yes/No
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
            pred_label = extract_yesno(answer)

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
    print(f"Total: {total}, Correct: {correct}, Invalid (no Yes/No): {invalid}")
    print(f"Accuracy (all): {acc:.2f}%")
    print(f"Valid Accuracy (excluding invalid): {valid_acc:.2f}%")
    if invalid_records:
        print(f"\nFirst 10 invalid predictions:")
        for r in invalid_records[:10]:
            print(f"  qid={r['qid']}, answer={r['answer']!r}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate FloodNet Yes/No predictions")
    parser.add_argument(
        "--pred",
        type=str,
        default="/mnt/hanhc/TLM-main/outputs/FloodNet_yesno_vl_base_system_prompt_qwen.jsonl",
        help="Path to prediction JSONL (question_id, ground_truth, answer)",
    )
    args = parser.parse_args()
    evaluate(args.pred)


if __name__ == "__main__":
    main()
