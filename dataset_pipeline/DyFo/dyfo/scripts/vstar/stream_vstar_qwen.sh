#!/bin/bash

METHOD=${1:-"clean"}
DEBUG=${2:-"False"}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="Qwen/Qwen2-VL-7B-Instruct"
NAME="Qwen2-VL-7B-Instruct"
# CKPT="llava-hf/llava-1.5-7b-hf"
# NAME="llava-v1.5-7b"


DATASET="vstar"
DATASET_FORMAT="vstar"
SPLIT="test_questions"
# DATASET="pope"
# DATASET_FORMAT="pope"
# SPLIT="coco/coco_pope_random"


mkdir -p playground/data/eval/$DATASET/answers_ours/$CKPT/$SPLIT/$METHOD

# change module to ccot module
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} VLLM_PORT=8008 python dyfo/src/run.py \
        --model-path /mnt/wj/Model/Qwen2.5-VL-7B-Instruct \
        --question-file /mnt/wj/gen_neg/NWPU-Caption/validation_dataset.tsv \
        --answers-file /mnt/wj/gen_neg/NWPU-Caption/vstar_eval_dyfo_${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --image-size 5000 \
        --temperature 0 \
        --method_name vstar.mcts \
        --debug false \
        &
done

wait

output_file=/mnt/wj/gen_neg/NWPU-Caption/vstar_eval_dyfo.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/wj/gen_neg/NWPU-Caption/vstar_eval_dyfo_${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Eval
echo $METHOD $DATASET $SPLIT
# python playground/data/eval/$DATASET/eval.py \
#     --path $output_file

