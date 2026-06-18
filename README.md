<h1 align="center">
Evaluating and Enhancing Negation Comprehension in Remote Sensing MLLMs
</h1>


<p align="center">
Haochen Han, Jue Wang, Alex Jinpeng Wang, Xutao Wen, Fangming Liu<br>
<sub>Peng Cheng Laboratory, Tsinghua University, Central South University</sub>
</p>

## Overview

This repository contains the NeFo codebase for evaluating and improving negation comprehension in remote sensing multimodal large language models. It is built on top of LLaMA-Factory and adds VL test-time learning, negation-aware dataset definitions, model inference scripts, and evaluation scripts for VQA, MCQ, referring expression grounding, FloodNet, and scene classification.

## Benchmarks

- Benchmarks：https://huggingface.co/datasets/mumu-0011/RS-Neg

## Repository Structure

```text
NeFo/
├── data/                    # JSON/JSONL datasets and dataset_info.json
├── dataset_pipeline/         # Dataset construction and filtering scripts
├── evaluation/               # Inference and scoring scripts
├── examples/train_lora/      # Training configs
├── src/llamafactory/         # Modified LLaMA-Factory source code
```

`data/dataset_info.json` defines the dataset names used by LLaMA-Factory training configs. The current training config uses names such as `MCQ_Merged_sub_negated`, `MCQ_Merged_negated`, `VQA_Merged_sub_negated`, and `VQA_Merged_negated`.

## Installation

```bash
git clone git@github.com:mumu011/NeFo.git
cd NeFo

conda create -n nefo python=3.10 -y
conda activate nefo

pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Training

The main example config is:

```text
examples/train_lora/qwen2_5_vl_ttl.yaml
```

Run training from the command line:

```bash
llamafactory-cli train examples/train_lora/qwen2_5_vl_ttl.yaml
```

## Inference

Use `evaluation/evaluation_with_finetuned_model.py` to generate prediction JSONL files. It supports these backends:

```text
geochat, qwen2_5_vl, rs_llava, internvl2_5, georeason, qwen3vl
```

Example with a LoRA adapter:

```bash
python evaluation/evaluation_with_finetuned_model.py \
  --backend qwen2_5_vl \
  --qwen2-vl-model /path/to/Qwen2.5-VL-7B-Instruct \
  --dataset data/VQA_Merged/merged_vqa_tlm_yesno.jsonl \
  --image-dir /path/to/Merged_Dataset/images \
  --lora-path saves/qwen2_5vl-7b/vl_ttl/nefo_sft_qwen2_5vl_sample_100 \
  --output outputs/Merged_VQA_predictions.jsonl \
  --batch-size 16 \
  --temperature 0.0
```

Example without LoRA:

```bash
python evaluation/evaluation_with_finetuned_model.py \
  --backend qwen2_5_vl \
  --qwen2-vl-model /path/to/Qwen2.5-VL-7B-Instruct \
  --dataset data/MCQ_Merged/merged_caption_negated_llava_format.jsonl \
  --image-dir /path/to/Merged_MCQ_Dataset/images \
  --lora-path none \
  --output outputs/Merged_MCQ_base.jsonl \
  --batch-size 16 \
  --temperature 0.0
```

For 8-GPU distributed inference, use:

```bash
torchrun --nproc_per_node=8 evaluation/evaluation_with_finetuned_model.py \
  --backend qwen2_5_vl \
  --qwen2-vl-model /path/to/Qwen2.5-VL-7B-Instruct \
  --dataset data/VQA_Merged/merged_vqa_tlm_yesno.jsonl \
  --image-dir /path/to/Merged_Dataset/images \
  --lora-path /path/to/lora_adapter \
  --output outputs/Merged_VQA_predictions.jsonl \
  --batch-size 16 \
  --temperature 0.0
```

## Evaluation

Prediction files are JSONL files written to `outputs/`. 

### VQA

Negative VQA:

```bash
python evaluation/evaluate_vqa_negative.py \
  --gold data/VQA_Merged/merged_vqa_tlm_yesno.jsonl \
  --pred outputs/Merged_VQA_predictions.jsonl \
  --dataset data/VQA_Merged/merged_vqa.json \
  --group-by qa_type
```

Positive VQA:

```bash
python evaluation/evaluate_vqa_positive.py \
  --gold data/VQA_Merged/merged_vqa_tlm_yesno_pos.jsonl \
  --pred outputs/Merged_VQA_positive_predictions.jsonl \
  --dataset data/VQA_Merged/merged_vqa.json \
  --group-by qa_type
```

### MCQ

```bash
python evaluation/MCQ_evaluation.py \
  outputs/Merged_MCQ_predictions.jsonl \
  --merged-mcq data/MCQ_Merged/merged_mcq.json \
  --show-failures
```

### Referring Expression Grounding

```bash
python evaluation/evaluate_bbox.py \
  --pred-file outputs/Ref_VRS_predictions.jsonl \
  --gt-file data/Ref_VRS/VRSBench_EVAL_referring_negated_polished_filtered_combined_original_question.jsonl \
  --image-dir /path/to/VRSBench/Images_val \
  --metadata-file data/Ref_VRS/VRSBench_EVAL_referring_negated_polished.json \
  --pred-normalized
```

### FloodNet

```bash
python evaluation/evaluate_floodnet_yesno.py \
  --pred outputs/VQA_FloodNet_predictions.jsonl

python evaluation/evaluate_floodnet_counting.py \
  --pred outputs/Count_FloodNet_predictions.jsonl
```

### Scene Classification

```bash
python evaluation/score_accuracy.py outputs/UCmerced_cls_predictions.jsonl
```

## Citation

Thanks to the open-source code of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

If you find this work useful, please cite the related paper:

```bibtex
@inproceedings{hutest,
  title={Test-Time Learning for Large Language Models},
  author={Hu, Jinwu and Zhang, Zitian and Chen, Guohao and Wen, Xutao and Shuai, Chao and Luo, Wei and Xiao, Bin and Li, Yuanqing and Tan, Mingkui},
  booktitle={Forty-second International Conference on Machine Learning}
}
```
