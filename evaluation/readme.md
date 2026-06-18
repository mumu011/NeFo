主要参考launch.json即可

对于Evaluate BBox
1. 如果使用qwen2.5，则按launch.json即可
2. 如果使用qwen3,则加上"--model-type", "qwen3"
3. 如果使用internvl3,则加上"--model-type", "internvl3"
4. 如果输出为（0.xx,...）如rs_llava,geochat，则加上"--pred-normalized"

对于Evaluate vqa
negative问题使用"name": "Evaluate VQA binary (local)"
positive问题使用"name": "Evaluate VQA binary (positive)"

对于FloodNet，训练时请参考src/llamafactory/launcher_vl_debug.py

对于Torchrun: evaluation
1. 如果使用rs_llava，请设为"--backend", "rs_llava" "--base-model", "/mnt/nvme1/wj/Model/RS-llava-v1.5-7b-Merged"
2. 如果使用RS-EoT-7B，请设为"--backend", "qwen2_5_vl" "--qwen2-vl-model", "/mnt/nvme1/wj/Model/RS-EoT-7B" "--max-new-tokens", "512"