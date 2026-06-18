#!/bin/bash

declare -a benchmarks=("coco/coco_pope_random" "coco/coco_pope_popular" "coco/coco_pope_adversarial" "aokvqa/aokvqa_pope_random" "aokvqa/aokvqa_pope_popular" "aokvqa/aokvqa_pope_adversarial" "gqa/gqa_pope_random" "gqa/gqa_pope_popular" "gqa/gqa_pope_adversarial")

for benchmark in "${benchmarks[@]}"; do
    bash $(dirname "$0")/stream_pope_qwen.sh mcts False $benchmark
done
