#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate VQA predictions but focus on classifying "positive" questions into
object_positive, attribute_positive, state_positive when grouping by qa_type.

This is a lightweight variant of evaluate_vqa_binary.py tailored to the
positive subset. It prefers any `type` field present in the gold/TLM jsonl.
If the TLM only contains a generic "positive", a small heuristic attempts
to infer the subtype from image-level fields and the question text.

Usage: same style as evaluate_vqa_binary.py; run with --group-by qa_type to
see per-type (including positive sub-types) metrics.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def extract_yesno(text: str):
	if text is None:
		return ""
	
	if "</think>" in text:
		text = text.split("</think>")[-1]

	s = text.strip().lower()
	if not s:
		return ""
	if s.startswith('yes'):
		return 'Yes'
	if s.startswith('no'):
		return 'No'
	if re.search(r'\byes\b', s):
		return 'Yes'
	if re.search(r'\bno\b', s):
		return 'No'
	return ''


def load_dataset_maps(dataset_path: str):
	p = Path(dataset_path)
	if not p.exists():
		raise SystemExit(f'dataset file not found: {p}')

	filename_fields = {}
	filename_q_to_type = {}

	if p.suffix.lower() == '.jsonl':
		opener = p.open('r', encoding='utf-8')
		with opener as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				rec = json.loads(line)
				fn = rec.get('filename') or (rec.get('images', [''])[0])
				if not fn:
					continue
				filename_fields[fn] = {
					'negative_state': rec.get('negative_state'),
					'negative_object': rec.get('negative_object'),
					'negative_attr': rec.get('negative_attr'),
				}
				qa_pairs = rec.get('qa_pairs') or []
				for qa in qa_pairs:
					q = qa.get('question')
					t = qa.get('type')
					if isinstance(q, str) and t:
						filename_q_to_type[(fn, q)] = t
	else:
		with p.open('r', encoding='utf-8') as f:
			data = json.load(f)
		for rec in data:
			fn = rec.get('filename')
			if not fn:
				continue
			filename_fields[fn] = {
				'negative_state': rec.get('negative_state'),
				'negative_object': rec.get('negative_object'),
				'negative_attr': rec.get('negative_attr'),
			}
			qa_pairs = rec.get('qa_pairs') or []
			for qa in qa_pairs:
				q = qa.get('question')
				t = qa.get('type')
				if isinstance(q, str) and t:
					filename_q_to_type[(fn, q)] = t

	return filename_fields, filename_q_to_type


def parse_tlm_meta(tlm_path: str):
	p = Path(tlm_path)
	if not p.exists():
		raise SystemExit(f'TLM file not found: {p}')
	qid_meta = {}
	with p.open('r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rec = json.loads(line)
			qid = rec.get('question_id')
			if isinstance(qid, list) and qid:
				qid = qid[0]
			qid = str(qid) if qid is not None else None
			images = rec.get('images') or []
			filename = images[0] if images else ''
			messages = rec.get('messages', [])
			user_q = ''
			if messages:
				user_q = messages[0].get('content', '')
			if isinstance(user_q, str) and user_q.startswith('<image>'):
				user_q = user_q[len('<image>'):].strip()
			gt = ''
			if len(messages) >= 2:
				gt = messages[1].get('content', '')
			gt_label = extract_yesno(gt)
			# preserve any explicit type field if present in the TLM record
			tlm_type = rec.get('type') or rec.get('qa_type')
			qid_meta[qid] = {
				'filename': filename,
				'question': user_q,
				'gt_label': gt_label,
				'qa_type': tlm_type,
			}
	return qid_meta


def build_qid_to_group(tlm_path: str, dataset_path: str, group_by: str):
	allowed_image_fields = {'negative_state', 'negative_object', 'negative_attr'}
	qid_meta = parse_tlm_meta(tlm_path)
	filename_fields, filename_q_to_type = load_dataset_maps(dataset_path)
	qid2group = {}
	for qid, meta in qid_meta.items():
		fn = meta['filename']
		q_text = meta['question']
		group_val = 'unknown'
		if group_by == 'qa_type':
			tlm_type = meta.get('qa_type')
			if tlm_type:
				# If TLM contains a generic 'positive', try heuristic inference
				if isinstance(tlm_type, str) and tlm_type.lower() in ('positive', 'pos'):
					lower_q = (q_text or '').lower()
					ffields = filename_fields.get(fn, {})
					neg_obj = ffields.get('negative_object')
					neg_attr = ffields.get('negative_attr')
					neg_state = ffields.get('negative_state')
					inferred = None
					if neg_obj and isinstance(neg_obj, str) and neg_obj.lower() in lower_q:
						inferred = 'object_positive'
					elif neg_attr and isinstance(neg_attr, str) and neg_attr.lower() in lower_q:
						inferred = 'attribute_positive'
					elif neg_state and isinstance(neg_state, str) and neg_state.lower() in lower_q:
						inferred = 'state_positive'
					group_val = inferred or tlm_type
				else:
					group_val = tlm_type
			else:
				group_val = filename_q_to_type.get((fn, q_text), 'unknown')
		elif group_by in allowed_image_fields:
			group_val = filename_fields.get(fn, {}).get(group_by, None)
		else:
			raise SystemExit(f'Unsupported group-by: {group_by}')
		if group_val is None:
			group_val = 'None'
		qid2group[str(qid)] = str(group_val)
	return qid2group


def evaluate(pred_file: str, gold_map: dict, qid2group=None, title_group=None):
	p = Path(pred_file)
	if not p.exists():
		raise SystemExit(f'prediction file not found: {p}')

	total = 0
	correct = 0
	invalid = 0
	missing = 0

	per_group_total = defaultdict(int)
	per_group_correct = defaultdict(int)
	per_group_invalid = defaultdict(int)
	per_group_missing = defaultdict(int)

	invalid_records = []
	missing_records = []

	with p.open('r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rec = json.loads(line)
			qid = rec.get('question_id')
			total += 1
			pred_text = rec.get('answer') if 'answer' in rec else rec.get('prediction', '')
			if qid is None:
				missing += 1
				missing_records.append({'qid': None, 'pred_text': pred_text})
				continue
			qid = str(qid)
			pred_label = extract_yesno(pred_text)
			if not pred_label:
				invalid += 1
				invalid_records.append({'qid': qid, 'pred_text': pred_text})
				if qid2group is not None:
					g = qid2group.get(qid, 'unknown')
					per_group_total[g] += 1
					per_group_invalid[g] += 1
				continue
			gt_label = gold_map.get(qid)
			if gt_label is None or gt_label == '':
				missing += 1
				missing_records.append({'qid': qid, 'pred_text': pred_text})
				if qid2group is not None:
					g = qid2group.get(qid, 'unknown')
					per_group_total[g] += 1
					per_group_missing[g] += 1
				continue
			if pred_label == gt_label:
				correct += 1
				if qid2group is not None:
					g = qid2group.get(qid, 'unknown')
					per_group_total[g] += 1
					per_group_correct[g] += 1
			else:
				if qid2group is not None:
					g = qid2group.get(qid, 'unknown')
					per_group_total[g] += 1

	accuracy = correct / total * 100 if total > 0 else 0
	valid_accuracy = correct / (total - invalid - missing) * 100 if (total - invalid - missing) > 0 else 0

	print(f"Total predictions: {total}")
	print(f"Correct: {correct}")
	print(f"Invalid predictions (couldn't extract Yes/No): {invalid}")
	print(f"Missing ground-truth: {missing}")
	if missing_records:
		print('\nDetails for missing ground-truth entries:')
		for mr in missing_records[:20]:
			print(f" - question_id: {mr.get('qid')}, predicted text: {mr.get('pred_text','')}")
	if invalid_records:
		print('\nDetails for invalid predictions (couldn\'t extract Yes/No):')
		for ir in invalid_records[:20]:
			print(f" - question_id: {ir.get('qid')}, predicted text: {ir.get('pred_text','')}")
	print(f"Accuracy: {accuracy:.2f}%")
	print(f"Valid Accuracy (excluding invalid/missing): {valid_accuracy:.2f}%")

	if qid2group is not None:
		header = title_group or 'Group'
		print(f"\nPer-{header} metrics:")
		for g in sorted(per_group_total.keys()):
			t = per_group_total[g]
			c = per_group_correct[g]
			inv = per_group_invalid[g]
			miss = per_group_missing[g]
			acc = c / t * 100 if t > 0 else 0.0
			denom = t - inv - miss
			vacc = c / denom * 100 if denom > 0 else 0.0
			print(f" - {g}: total={t}, correct={c}, invalid={inv}, missing={miss}, "
				  f"acc={acc:.2f}%, valid_acc={vacc:.2f}%")


def load_gold(tlm_path: str):
	gold = {}
	p = Path(tlm_path)
	if not p.exists():
		raise SystemExit(f'gold TLM file not found: {p}')
	with p.open('r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			rec = json.loads(line)
			qid = rec.get('question_id')
			if isinstance(qid, list) and qid:
				qid = qid[0]
			messages = rec.get('messages', [])
			gt = ''
			if len(messages) >= 2:
				gt = messages[1].get('content', '')
			gt_label = extract_yesno(gt)
			if not gt_label:
				if isinstance(gt, str) and gt.strip().lower().startswith('y'):
					gt_label = 'Yes'
				elif isinstance(gt, str) and gt.strip().lower().startswith('n'):
					gt_label = 'No'
			if qid:
				gold[str(qid)] = gt_label
	return gold


def main():
	import argparse

	parser = argparse.ArgumentParser(description='Evaluate VQA predictions (positive-types focused)')
	parser.add_argument('--gold', dest='gold_path', type=str,
						default='/mnt/hanhc/TLM-main/data/the7/VQA/dataset_nwpu_test_vqa_polished_positive_tlm_shuffle.jsonl',
						help='Path to gold TLM jsonl file')
	parser.add_argument('--pred', dest='pred_path', type=str,
						default='/mnt/hanhc/TLM-main/outputs/VQA_base_positive_system_prompt.jsonl',
						help='Path to prediction jsonl file')
	parser.add_argument('--dataset', dest='dataset_path', type=str,
						default='/mnt/nfs1/hanhc/TLM-main/data/the7/dataset_nwpu_test_vqa_polished.json',
						help='Path to original dataset json/jsonl (for grouping)')
	parser.add_argument('--group-by', dest='group_by', type=str, default='qa_type',
						choices=['qa_type', 'negative_state', 'negative_object', 'negative_attr', 'none', 'None', 'off'],
						help='Grouping field for per-type accuracy. Use qa_type for question type. Use none to disable.')

	args = parser.parse_args()

	gold_path = args.gold_path
	pred_path = args.pred_path
	dataset_path = args.dataset_path
	group_by = args.group_by

	print('Loading gold from', gold_path)
	gold = load_gold(gold_path)
	print(f'Loaded {len(gold)} gold labels')

	qid2group = None
	title_group = None
	if group_by not in ('none', 'None', 'off'):
		print(f'Building question_id -> {group_by} map using', dataset_path)
		try:
			qid2group = build_qid_to_group(gold_path, dataset_path, group_by)
			title_group = group_by
			print(f'Mapped {len(qid2group)} question_ids to {group_by}')
		except Exception as e:
			print('Failed to build group map:', e)
			qid2group = None

	print('Evaluating predictions in', pred_path)
	evaluate(pred_path, gold, qid2group=qid2group, title_group=title_group)


if __name__ == '__main__':
	main()

