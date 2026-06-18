import json
import argparse
import sys

def extract_answer(raw_answer):
    """If answer contains </think>, use the content after it as the final answer (e.g. RS-EoT output)."""
    if not raw_answer or not isinstance(raw_answer, str):
        return raw_answer or ""
    if "</think>" in raw_answer:
        return raw_answer.split("</think>")[-1].strip()
    return raw_answer.strip()


def normalize(text):
    if not isinstance(text, str):
        return str(text).lower().strip()
    return text.lower().strip().rstrip('.')

def calculate_accuracy(file_path):
    total = 0
    correct = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    ground_truth = normalize(data.get('ground_truth', ''))
                    raw_answer = data.get('answer', '')
                    answer = normalize(extract_answer(raw_answer))
                    
                    if ground_truth == answer:
                        correct += 1
                    elif ground_truth.startswith('not ') and ' but ' in ground_truth:
                        if ground_truth.split(' but ')[-1].strip() == answer:
                            correct += 1
                    total += 1
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line}")
        
        if total == 0:
            print(f"No valid entries found in {file_path}")
            return

        accuracy = correct / total
        print(f"File: {file_path}")
        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy from a JSONL file.")
    parser.add_argument("file_path", help="Path to the JSONL file")
    args = parser.parse_args()
    
    calculate_accuracy(args.file_path)
