CUDA_VISIBLE_DEVICES=8,9,10,11 bash dyfo/scripts/lmm_server/qwen_server.sh &
CUDA_VISIBLE_DEVICES=4,5,6,7 bash dyfo/scripts/expert_server/start_server.sh 
