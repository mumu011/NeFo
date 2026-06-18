#!/bin/bash

METHOD=${1:-"clean"}
DEBUG=${2:-"False"}
SPLIT=${3:-"gqa/gqa_pope_random"}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# CKPT="Qwen/Qwen2-VL-7B-Instruct"
# NAME="Qwen2-VL-7B-Instruct"
CKPT="llava-hf/llava-1.5-7b-hf"
NAME="llava-v1.5-7b"

# DATASET="vstar"
# DATASET_FORMAT="vstar"
# SPLIT="test_questions"
DATASET="pope"
DATASET_FORMAT="pope"
# SPLIT="gqa/gqa_pope_random"


mkdir -p playground/data/eval/$DATASET/answers_ours/$CKPT/$SPLIT/$METHOD

# change module to ccot module
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]}  python dyfo/src/run.py \
        --model-path $CKPT \
        --question-file ./playground/data/eval/$DATASET/$SPLIT.tsv \
        --answers-file ./playground/data/eval/$DATASET/answers_ours/$CKPT/$SPLIT/$METHOD/${CHUNKS}_${IDX}_${NAME}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --single-pred-prompt \
        --temperature 0 \
        --method_name $DATASET_FORMAT.$METHOD \
        --debug $DEBUG \
        &
done

wait

output_file=./playground/data/eval/$DATASET/answers_ours/$CKPT/$SPLIT/$METHOD/${NAME}.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/$DATASET/answers_ours/$CKPT/$SPLIT/$METHOD/${CHUNKS}_${IDX}_${NAME}.jsonl >> "$output_file"
done


# Eval
echo $METHOD $DATASET $SPLIT
python playground/data/eval/$DATASET/eval.py \
    --path $output_file

