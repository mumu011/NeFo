# import json
# import re
# from collections import defaultdict
# from pathlib import Path
#
#
# def extract_answer(answer: str) -> str:
#     """从答案中提取选项字母A/B/C/D"""
#     answer = answer.strip()
#     if not answer:
#         return ""
#
#     # 直接是单个字母
#     if answer.upper() in ['A', 'B', 'C', 'D']:
#         return answer.upper()
#
#     # 以字母开头（如 "A. xxx" 或 "A xxx"）
#     match = re.match(r'^([A-Da-d])', answer)
#     if match:
#         return match.group(1).upper()
#
#     return ""
#
#
# def load_mcq_types(merged_mcq_path):
#     """从merged_mcq.json加载question_id到mcq_type的映射"""
#     qid_to_type = {}
#     merged_mcq_path = Path(merged_mcq_path)
#
#     if not merged_mcq_path.exists():
#         print(f"Warning: merged_mcq.json not found at {merged_mcq_path}")
#         return qid_to_type
#
#     with open(merged_mcq_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     for item in data:
#         filename = item.get('filename')
#         mcq_type = item.get('mcq_type', 'unknown')
#         if filename:
#             qid_to_type[filename] = mcq_type
#
#     return qid_to_type
#
#
# def evaluate(prediction_file, merged_mcq_path=None):
#     """评测预测结果"""
#     # 加载类型映射
#     if merged_mcq_path is None:
#         # 默认路径
#         default_path = Path(__file__).parent.parent.parent / "data" / "the7" / "MCQ_Merged" / "merged_mcq.json"
#         merged_mcq_path = default_path
#
#     qid_to_type = load_mcq_types(merged_mcq_path)
#
#     with open(prediction_file, 'r') as f:
#         predictions = [json.loads(line) for line in f if line.strip()]
#
#     total = len(predictions)
#     correct = 0
#     invalid = 0
#
#     # 按类型统计
#     type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'invalid': 0})
#
#     for pred in predictions:
#         question_id = pred.get('question_id', '')
#         ground_truth = pred['ground_truth'].strip().upper()
#         extracted = extract_answer(pred['answer'])
#
#         # 获取类型
#         mcq_type = qid_to_type.get(question_id, 'unknown')
#         type_stats[mcq_type]['total'] += 1
#
#         if not extracted:
#             invalid += 1
#             type_stats[mcq_type]['invalid'] += 1
#         elif extracted == ground_truth:
#             correct += 1
#             type_stats[mcq_type]['correct'] += 1
#
#     accuracy = correct / total * 100
#     valid_accuracy = correct / (total - invalid) * 100 if total > invalid else 0
#
#     print(f"Total: {total}")
#     print(f"Correct: {correct}")
#     print(f"Invalid: {invalid}")
#     print(f"Accuracy: {accuracy:.2f}%")
#     print(f"Valid Accuracy: {valid_accuracy:.2f}%")
#
#     # 按类型输出统计
#     print("\n" + "=" * 60)
#     print("Accuracy by Type:")
#     print("=" * 60)
#     for mcq_type in sorted(type_stats.keys()):
#         stats = type_stats[mcq_type]
#         type_total = stats['total']
#         type_correct = stats['correct']
#         type_invalid = stats['invalid']
#         type_accuracy = type_correct / type_total * 100 if type_total > 0 else 0
#         type_valid_accuracy = type_correct / (type_total - type_invalid) * 100 if (type_total - type_invalid) > 0 else 0
#
#         print(f"\nType: {mcq_type}")
#         print(f"  Total: {type_total}")
#         print(f"  Correct: {type_correct}")
#         print(f"  Invalid: {type_invalid}")
#         print(f"  Accuracy: {type_accuracy:.2f}%")
#         print(f"  Valid Accuracy: {type_valid_accuracy:.2f}%")
#
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Evaluate MCQ predictions")
#     parser.add_argument("prediction_file", help="Path to prediction file (JSONL format)")
#     parser.add_argument(
#         "--merged-mcq",
#         dest="merged_mcq_path",
#         default=None,
#         help="Path to merged_mcq.json file (optional, will use default path if not provided)"
#     )
#
#     args = parser.parse_args()
#
#     evaluate(args.prediction_file, args.merged_mcq_path)


import json
import re
from collections import defaultdict
from pathlib import Path


def extract_answer(answer: str) -> str:
    """从模型的长输出中提取选项字母A/B/C/D，按优先级依次尝试多种模式"""
    answer = answer.strip()
    if not answer:
        return ""

    # 直接是单个字母
    if answer.upper() in ['A', 'B', 'C', 'D']:
        return answer.upper()

    # 1. boxed{X} 格式（如 **boxed{C}**）
    match = re.search(r'boxed\{([A-Da-d])\}', answer)
    if match:
        return match.group(1).upper()

    # 2. "the answer is X" / "the answer is: X" / "the correct answer is X"
    match = re.search(r'(?:the\s+)?(?:correct\s+|final\s+)?answer\s+is\s*:?\s*\**([A-Da-d])\b', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 3. "So, X" / "So the answer is X" 在末尾
    match = re.search(
        r'(?:so|therefore|thus|hence)[,\s]+(?:the\s+(?:correct\s+|final\s+)?answer\s+is\s*:?\s*)?\**([A-Da-d])\b',
        answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 4. "answer: X" / "Answer: X"
    match = re.search(r'answer\s*:\s*\**([A-Da-d])\b', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 5. 末尾带加粗的选项（如 **C. xxxx**）—— 取最后一个出现的
    matches = re.findall(r'\*\*([A-Da-d])[.\s]', answer)
    if matches:
        return matches[-1].upper()

    # 6. "选项X" / "option X" / "choose X" / "select X"
    match = re.search(r'(?:option|choose|select|pick)\s*:?\s*\**([A-Da-d])\b', answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # 7. 最后几行中独立出现的选项字母（如行首 "C" 或 "C."）
    last_lines = answer.strip().split('\n')[-5:]
    for line in reversed(last_lines):
        line = line.strip().strip('*').strip()
        match = re.match(r'^([A-Da-d])[.\s\)]', line)
        if match:
            return match.group(1).upper()
        if line.upper() in ['A', 'B', 'C', 'D']:
            return line.upper()

    # 8. 回退：整个文本中最后出现的 "X." 或 "X)" 模式（X=A/B/C/D）
    matches = re.findall(r'\b([A-Da-d])[.\)]\s', answer)
    if matches:
        return matches[-1].upper()

    # 9. 最终回退：文本开头的字母
    match = re.match(r'^([A-Da-d])', answer)
    if match:
        return match.group(1).upper()

    return ""


def load_mcq_types(merged_mcq_path):
    """从merged_mcq.json加载question_id到mcq_type的映射"""
    qid_to_type = {}
    merged_mcq_path = Path(merged_mcq_path)

    if not merged_mcq_path.exists():
        print(f"Warning: merged_mcq.json not found at {merged_mcq_path}")
        return qid_to_type

    with open(merged_mcq_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        filename = item.get('filename')
        mcq_type = item.get('mcq_type', 'unknown')
        if filename:
            qid_to_type[filename] = mcq_type

    return qid_to_type


def evaluate(prediction_file, merged_mcq_path=None, show_failures=False):
    """评测预测结果"""
    if merged_mcq_path is None:
        default_path = Path(__file__).parent.parent.parent / "data" / "the7" / "MCQ_Merged" / "merged_mcq.json"
        merged_mcq_path = default_path

    qid_to_type = load_mcq_types(merged_mcq_path)

    with open(prediction_file, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    total = len(predictions)
    correct = 0
    invalid = 0

    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'invalid': 0})

    invalid_samples = []
    wrong_samples = []

    for pred in predictions:
        question_id = pred.get('question_id', '')
        ground_truth = pred['ground_truth'].strip().upper()
        extracted = extract_answer(pred['answer'])

        mcq_type = qid_to_type.get(question_id, 'unknown')
        type_stats[mcq_type]['total'] += 1

        if not extracted:
            invalid += 1
            type_stats[mcq_type]['invalid'] += 1
            invalid_samples.append({
                'question_id': question_id,
                'ground_truth': ground_truth,
                'raw_answer': pred['answer'][:200]  # 截取前200字符
            })
        elif extracted == ground_truth:
            correct += 1
            type_stats[mcq_type]['correct'] += 1
        else:
            wrong_samples.append({
                'question_id': question_id,
                'ground_truth': ground_truth,
                'extracted': extracted,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    valid_total = total - invalid
    valid_accuracy = correct / valid_total * 100 if valid_total > 0 else 0

    print(f"{'=' * 60}")
    print(f"Overall Results")
    print(f"{'=' * 60}")
    print(f"Total:          {total}")
    print(f"Correct:        {correct}")
    print(f"Wrong:          {valid_total - correct}")
    print(f"Invalid:        {invalid}  ({invalid / total * 100:.1f}%)")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"Valid Accuracy: {valid_accuracy:.2f}%")

    # 按类型输出统计
    print(f"\n{'=' * 60}")
    print(f"Accuracy by Type")
    print(f"{'=' * 60}")
    for mcq_type in sorted(type_stats.keys()):
        stats = type_stats[mcq_type]
        t = stats['total']
        c = stats['correct']
        inv = stats['invalid']
        acc = c / t * 100 if t > 0 else 0
        v_acc = c / (t - inv) * 100 if (t - inv) > 0 else 0

        print(f"\n  [{mcq_type}]")
        print(f"    Total: {t}  Correct: {c}  Invalid: {inv}")
        print(f"    Accuracy: {acc:.2f}%  Valid Accuracy: {v_acc:.2f}%")

    # 显示无法提取的样例
    if show_failures and invalid_samples:
        print(f"\n{'=' * 60}")
        print(f"Failed to extract answer ({len(invalid_samples)} samples):")
        print(f"{'=' * 60}")
        for s in invalid_samples[:20]:  # 最多显示20个
            print(f"\n  QID: {s['question_id']}  GT: {s['ground_truth']}")
            print(f"  Raw: {s['raw_answer']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate MCQ predictions (robust extraction)")
    parser.add_argument("prediction_file", help="Path to prediction file (JSONL format)")
    parser.add_argument(
        "--merged-mcq", dest="merged_mcq_path", default=None,
        help="Path to merged_mcq.json file"
    )
    parser.add_argument(
        "--show-failures", action="store_true",
        help="Show samples where answer extraction failed"
    )
    args = parser.parse_args()

    evaluate(args.prediction_file, args.merged_mcq_path, args.show_failures)