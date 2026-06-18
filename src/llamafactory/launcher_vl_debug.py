#!/usr/bin/env python3
"""
按照网上解决思路：直接调试本地 LLaMA-Factory
"""
import sys
import os
current_dir = os.getcwd()
print(f"!!! ACTUAL WORKING DIR: {current_dir}")
import atexit
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")

# 强制设置环境变量，禁用分布式训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NPROC_PER_NODE"] = "1"
os.environ["NNODES"] = "1"
os.environ["NODE_RANK"] = "0"
os.environ["FORCE_TORCHRUN"] = "0"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "12365"

# 🔧 设置模型缓存路径，解决PyCharm环境变量问题
os.environ["TRANSFORMERS_CACHE"] = "/home/hanhch/pretrained_models"
os.environ["HF_HOME"] = "/home/hanhch/pretrained_models"
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式，不尝试网络下载
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 设置环境变量忽略警告
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_VERBOSITY"] = "error"

# 获取项目根目录
current_dir = os.getcwd()  # /mnt/hanhc/TLM-main
# print(f"DEBUG: Current directory: {current_dir}")

# 添加本地 src 目录到 Python 路径的最前面
src_dir = os.path.join(current_dir, 'src')  # /mnt/hanhc/TLM-main/src
sys.path.insert(0, src_dir)

# 强制设置环境变量，确保使用本地版本
os.environ["PYTHONPATH"] = src_dir

# 验证路径设置 - 注释掉冗余输出
# print(f"DEBUG: Added {src_dir} to Python path")
# print(f"DEBUG: Current sys.path[0]: {sys.path[0]}")
# print(f"DEBUG: Current sys.path[1]: {sys.path[1] if len(sys.path) > 1 else 'N/A'}")

# 检查本地模型路径 - 只保留错误检查
local_model_path = '/home/hanhch/pretrained_models/geochat-7B'  # 🔧 同步yaml配置
if not os.path.exists(local_model_path):
    print(f"ERROR: Model not found at {local_model_path}")

# 检查模型配置文件 - 只保留错误检查
config_file_path = os.path.join(local_model_path, 'config.json')
if not os.path.exists(config_file_path):
    print(f"ERROR: Model config.json not found at {config_file_path}")

# 详细检查数据集相关路径
# print(f"\nDEBUG: === 数据集路径检查 ===")
# print(f"DEBUG: 当前工作目录: {os.getcwd()}")

# 检查各种可能的 dataset_info.json 位置
# possible_paths = [
#     '/mnt/hanhc/TLM-main/data/dataset_info.json'
# ]

# 只保留必要的路径检查，注释掉详细输出
# for path in possible_paths:
#     exists = os.path.exists(path)
#     if not exists:
#         print(f"ERROR: Required file not found: {path}")
    # print(f"DEBUG: {path}: {'EXISTS' if exists else 'NOT EXISTS'}")
    # if exists:
    #     print(f"DEBUG:   Absolute path: {os.path.abspath(path)}")

# 检查 data 目录 - 只保留错误检查
data_dir = os.path.join(current_dir, 'data')
if not os.path.exists(data_dir):
    print(f"ERROR: Data directory not found: {data_dir}")

# 注释掉详细目录遍历输出
# print(f"\nDEBUG: Data directory: {data_dir}")
# print(f"DEBUG: Data directory exists: {os.path.exists(data_dir)}")
# if os.path.exists(data_dir):
#     print(f"DEBUG: Data directory contents:")
#     for item in os.listdir(data_dir):
#         item_path = os.path.join(data_dir, item)
#         if os.path.isdir(item_path):
#             print(f"  [DIR] {item}/")
#             try:
#                 sub_items = os.listdir(item_path)
#                 for sub_item in sub_items[:5]:
#                     print(f"    - {sub_item}")
#                 if len(sub_items) > 5:
#                     print(f"    ... and {len(sub_items) - 5} more items")
#             except PermissionError:
#                 print(f"    [Permission denied]")
#         else:
#             print(f"  [FILE] {item}")

# 注释掉llamafactory目录检查
llamafactory_data_dir = os.path.join(src_dir, 'llamafactory', 'data')
if not os.path.exists(llamafactory_data_dir):
    print(f"ERROR: LLaMA-Factory data directory not found: {llamafactory_data_dir}")

# print(f"\nDEBUG: LLaMA-Factory data directory: {llamafactory_data_dir}")
# print(f"DEBUG: LLaMA-Factory data directory exists: {os.path.exists(llamafactory_data_dir)}")
# if os.path.exists(llamafactory_data_dir):
#     print(f"DEBUG: LLaMA-Factory data directory contents:")
#     for item in os.listdir(llamafactory_data_dir):
#         item_path = os.path.join(llamafactory_data_dir, item)
#         if os.path.isdir(item_path):
#             print(f"  [DIR] {item}/")
#         else:
#             print(f"  [FILE] {item}")

# 强制设置环境变量，确保使用本地版本
os.environ["PYTHONPATH"] = src_dir

# 直接传递参数字典给 run_exp，避免命令行参数解析问题
args_dict = {
    # model
    # 'model_name_or_path': '/home/hanhch/pretrained_models/geochat-7B',  # 🔧 同步yaml配置
    'model_name_or_path': '/mnt/nvme1/wj/Model/Qwen2.5-VL-7B-Instruct',  # 🔧 同步yaml配置
    # 'model_name_or_path': '/mnt/nvme1/wj/Model/Qwen3-VL-8B-Instruct',  # 🔧 同步yaml配置
    # 'model_name_or_path': '/mnt/nvme1/wj/Model/GeoReason',
    #'model_name_or_path': '/mnt/nvme1/wj/Model/RS-llava-v1.5-7b-Merged',
    # 'model_name_or_path': '/mnt/nvme1/wj/Model/InternVL3-8B',  # online_ttl 兼容 InternVL3

    # method
    'stage': 'vl_ttl',
    'do_train': True,
    'do_predict': True,
    'finetuning_type': 'lora',
    'lora_target': 'q_proj,v_proj',  # 🔧 同步yaml配置
    # 'lora_target': 'all',
    'lora_rank': 8,
    'lora_alpha': 16,
    'trust_remote_code': True,
    'freeze_vision_tower': True,  # 🔧 同步yaml配置
    'disable_gradient_checkpointing': True,  # 🔧 同步yaml配置

    # ttl parameters
    'setting': 'online_ttl',
    # 'setting': 'tent',
    # 'setting': 'sar',
    # 'setting': 'online_ttl_TLM',
    'threshold': 3,
    'lamb': 0.1,
    'aug_entropy_weight': 0.0,  # weight for aug_entropy_loss: total_loss = -kl_prcp_loss + aug_entropy_loss * aug_entropy_weight
    'use_sft_loss': True,       # 是否在online_ttl中加入ground_truth的SFT CE损失
    'sft_loss_weight': 0.01,      # SFT损失权重: total_loss = ttl_loss + sft_loss_weight * sft_loss
    'streaming_batch_size': 1,

    # dataset
    # 'dataset': 'NWPUCaption_sub_negated',
    # 'dataset': 'UCmerced_sub_negated',
    # 'dataset': 'VQA_sub_negated',
    # 'dataset': 'VQA_NWPU_sub_negated',
    # 'dataset': 'NWPU_MCQ_sub_negated',
    # 'dataset': 'VQA_RSICD_sub_negated',
    # 'dataset': 'RSICD_MCQ_sub_negated',
    'dataset': 'VQA_Merged_sub_negated',
    # 'dataset': 'MCQ_Merged_sub_negated',
    # 'eval_dataset': 'NWPUCaption_negated',  # 🔧 同步yaml配置
    # 'eval_dataset': 'UCmerced_negated',
    # 'eval_dataset': 'VQA_negated',
    # 'eval_dataset': 'VQA_NWPU_negated',
    # 'eval_dataset': 'NWPU_MCQ_negated',
    # 'eval_dataset': 'VQA_RSICD_negated',
    # 'eval_dataset': 'RSICD_MCQ_negated',
    'eval_dataset': 'VQA_Merged_negated',
    # 'eval_dataset': 'MCQ_Merged_negated',
    # 'image_dir': '/mnt/hanhc/Remote_Data/MCQ/NWPU_Captions/RESISC45_IMG',  # 🔧 同步yaml配置
    # 'image_dir': '/mnt/nvme0/hhc_data/GeoChat_Dataset/Scene_Classification/UCMerced_LandUse/Images',
    # 'image_dir': '/mnt/nfs1/hanhc/CyCLIP/data/RESISC45/RESISC45_IMG',
    # 'image_dir': '/mnt/nfs1/wj/gen_neg/NWPU-Caption/NWPU_images_all',
    # 'image_dir': '/mnt/nfs1/wj/gen_neg/RSICD/RSICD_images',
    'image_dir': '/mnt/nfs1/wj/gen_neg/Merged_Dataset/images',
    # 'image_dir': '/mnt/nfs1/wj/gen_neg/Merged_MCQ_Dataset/images',
    # 'template': 'llava_no_system',
    #'template': 'llava',  # RS-LLaVA / GeoChat
    'template': 'qwen2_vl',
    # 'template': 'internvl3',  # InternVL3-8B
    'cutoff_len': 2048,
    'max_samples': 150,  # 🔧 同步yaml配置
    'overwrite_cache': True,
    'preprocessing_num_workers': 1,  # 🔧 同步yaml配置
    
    # 否定词过滤配置
    'enable_negation_filtering': True,
    # 'negation_words': ["no", "not", "without", "non-", "none", "absent", "devoid", "empty", "free", "freely", "void", "nobody", "nothing", "nowhere", "lack", "lacks"],
    'negation_words': ["no", "not", "without", "non", "none"],

    # output

    # 'output_dir': 'saves/qwen2_5vl-7b-scaling-data/vl_ttl/nefo_qwen2_5vl_sample_100',
    # 'output_dir': 'saves/qwen2_5vl-7b-scaling-data/vl_ttl/tent_qwen2_5vl_sample_100',
    # 'output_dir': 'saves/qwen2_5vl-7b-scaling-data/vl_ttl/sar_qwen2_5vl_sample_100',
    # 'output_dir': 'saves/qwen2_5vl-7b-scaling-data/vl_ttl/tlm_qwen2_5vl_sample_100',
    'output_dir': 'saves/qwen2_5vl-7b-scaling-data/vl_ttl/nefo_sft_qwen2_5vl_sample_100',
    # 'output_dir': 'saves/qwen3vl-8b/vl_ttl/VQA_Merged_sub_lamb_0.1-threshold_3-lr_5e-5-seed_42_system_prompt_qwen',
    # 'output_dir': 'saves/qwen3vl-8b/vl_ttl/MCQ_Merged_sub_lamb_0.1-threshold_3-lr_5e-4-seed_42_system_prompt_qwen',
    # 'output_dir': 'saves/georeason/vl_ttl/VQA_Merged_sub_lamb_0.1-threshold_3-lr_5e-5-seed_42_system_prompt_qwen',
    # 'output_dir': 'saves/georeason/vl_ttl/MCQ_Merged_sub_lamb_0.1-threshold_3-lr_5e-4-seed_42_system_prompt_qwen',
    # 'output_dir': 'saves/rs-llava-v1.5-7b/vl_ttl/VQA_Merged_sub_lamb_0.1-threshold_3-lr_5e-5-seed_42_system_prompt',
    #'output_dir': 'saves/rs-llava-v1.5-7b/vl_ttl/MCQ_Merged_sub_lamb_0.1-threshold_3-lr_5e-5-seed_42_system_prompt',
    # 'output_dir': 'saves/internvl3-8b/vl_ttl/MCQ_Merged_sub_lamb_0.1-threshold_3-lr_5e-5-seed_42_system_prompt',
    'logging_steps': 10,
    'save_steps': 300,  # 🔧 同步yaml配置
    'plot_loss': True,
    'overwrite_output_dir': True,

    # train
    'seed': 42,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 1,  # 🔧 同步yaml配置
    # 'learning_rate': 5.0e-5,  # 🔧 同步yaml配置
    # 'learning_rate': 5.0e-4,
    'learning_rate': 5.0e-5,
    'num_train_epochs': 1.0,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.0,
    'bf16': True,
    'fp16': False,  # 添加 fp16 参数
    'ddp_timeout': 180000000,
    'ddp_find_unused_parameters': True,  # 🔧 添加DDP修复参数

    # eval
    'do_eval': None,

    # predict
    'temperature': 0.0,
    'do_sample': False,
    'max_new_tokens': 512,
    'per_device_eval_batch_size': 1,
    'predict_with_generate': True,

    'report_to': 'none',
}

# print(f"\nDEBUG: Args dict prepared:")
# for key, value in args_dict.items():
#     print(f"  {key}: {value}")


# 进程组清理函数
def cleanup_process_group():
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            # print("DEBUG: Process group destroyed successfully")
    except Exception as e:
        print(f"ERROR: Process group cleanup failed: {e}")


# 注册退出时的清理函数
atexit.register(cleanup_process_group)

# 强制使用本地版本 - 先卸载已安装的包
try:
    import llamafactory
    # print(f"DEBUG: Found existing llamafactory at: {llamafactory.__file__}")
    
    # 如果导入的不是本地版本，强制重新导入
    if not llamafactory.__file__.startswith(src_dir):
        print("WARNING: Imported from pip package, forcing reload...")
        
        # 从 sys.modules 中移除已导入的模块
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('llamafactory'):
                del sys.modules[module_name]
                # print(f"DEBUG: Removed {module_name} from sys.modules")
        
        # 重新导入
        import llamafactory
        print(f"SUCCESS: Re-imported llamafactory from local version")
    
except ImportError:
    pass  # print("DEBUG: No existing llamafactory found, importing from local...")

# 验证导入的是本地版本
try:
    # print("\nDEBUG: Attempting to import from local LLaMA-Factory...")

    # 在导入前设置更严格的警告过滤
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # 先检查模块路径
    import llamafactory

    # print(f"DEBUG: llamafactory module path: {llamafactory.__file__}")

    # 检查是否来自本地路径
    module_root = os.path.dirname(llamafactory.__file__)
    # print(f"DEBUG: Module root directory: {module_root}")
    # print(f"DEBUG: Expected src_dir: {src_dir}")
    
    # 检查模块根目录是否在 src_dir 下
    if module_root.startswith(src_dir):
        print("SUCCESS: Using local LLaMA-Factory")
    else:
        print("WARNING: Using pip package, not local version!")
        print(f"  Expected: {src_dir}")
        print(f"  Actual: {module_root}")
        raise ImportError("Failed to import local version")

    # 强制设置工作目录为项目根目录，确保相对路径正确
    os.chdir(current_dir)
    # print(f"DEBUG: Changed working directory to: {os.getcwd()}")
    
    # 验证 dataset_info.json 是否可访问
    dataset_info_path = os.path.join(current_dir, 'data', 'dataset_info.json')
    if not os.path.exists(dataset_info_path):
        print(f"ERROR: dataset_info.json not found at: {dataset_info_path}")
        raise FileNotFoundError(f"dataset_info.json not found at {dataset_info_path}")

    from llamafactory.train.tuner import run_exp

    print("SUCCESS: Imported run_exp function")

    # 直接传递参数字典给 run_exp
    print("🚀 Starting training...")
    run_exp(args=args_dict)  # 直接传递参数字典
    print("✅ Training completed!")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()