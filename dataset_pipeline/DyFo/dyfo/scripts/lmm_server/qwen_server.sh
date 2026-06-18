gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
gpu_count=${#GPULIST[@]}
port=$((8000 + ${GPULIST[0]}))

vllm serve --port $port /mnt/wj/Model/Qwen2.5-VL-7B-Instruct --max-model-len 32768 --dtype float16 --tensor-parallel-size $gpu_count