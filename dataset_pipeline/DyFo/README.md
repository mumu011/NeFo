<p align="center" width="100%">
<a target="_blank"><img src="figs/image.png" alt="Dynamic Focus" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>


# DyFo: A Training-Free Dynamic Focus Visual Search for Enhancing LMMs in Fine-Grained Visual Understanding

This is the official repo for Dynamic Focus (Visual Search), a training-free visual search method for enhancing LMMs/MLLMs in Fine-Grained Visual Understanding by simulating human dynamic visual focus.

<div style='display:flex; gap: 0.25rem; '>
<a href='LICENSE.txt'><img src='https://img.shields.io/badge/License-Apache 2.0-g.svg'></a>
<a href='https://arxiv.org/abs/2504.14920'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

## 🔥 Update
* [2025-08-11]: 🚀 Updated to be compatible with the latest vllm. Merged DyFo and expert environments for easier setup.
* [2025-05-15]: 🚀 Codes released.
* [2025-04-21]: ⭐️ DyFo is selected as Poster Highlight in CVPR 2025! (Top 13.5% in accepted papers), check out [this link](https://arxiv.org/abs/2504.14920) for details.

## 🎯 Overview
<!-- ![VCD](figs/figure1.png) -->
- We introduce **DyFo (Dynamic Focus)**, a **training-free** visual search method that dynamically adjusts focus regions to enhance fine-grained visual understanding in large multimodal models (LMMs).  
- The focus adjustment is guided by a **bidirectional interaction** between LMMs and visual experts, optimized via a **Monte Carlo Tree Search (MCTS)** algorithm
- DyFo effectively **filters out irrelevant content** while avoiding the need for additional training or specialized localization modules, leading to improved **fine-grained visual understanding** and reduced hallucination in LMMs. 


## 🕹️ Usage
### 1. Environment Setup

DyFo combines two components: (1) A Large Multimodal Model (LMM) like `Qwen2-VL` and `LLaVA-1.5` ([vllm](https://github.com/vllm-project/vllm)), and (2) A visual expert like `Lang_SAM`([this link](https://github.com/luca-medeiros/lang-segment-anything)) to collaborative inference.

> [!NOTE]
> If you encounter network issues accessing GitHub or HuggingFace during installation, you can try using these mirror sites:  
> 1. GitHub Mirror: [this link](https://github.com/runningcheese/MirrorSite)  
> 2. HuggingFace Mirror: [this link](https://hf-mirror.com/)


#### 1. Python environment setup:

```bash
conda create -n dyfo python=3.11
conda activate dyfo
pip install -r requirements.txt
```

#### 2. Visual Expert (Lang_SAM) install:
 - Download github repository:
```bash
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
```

 - (Manual Action) Modify line 41 in `lang_sam/models/gdino.py` to support batch inference:
```
inputs = self.processor(images=images_pil, text=texts_prompt, return_tensors="pt", padding=True).to(self.model.device)
```
 - (Manual Action) Modify line 47 in `lang_sam/models/gdino.py` to adapt for latest transformers (version 4.55):
```
threshold=box_threshold,
```

### 2. Data Preparation
Download the dataset from [this link](https://huggingface.co/datasets/oking0197/Dyfo) and unzip `dataset.zip` to get the following directory structure:
```
.
├── dyfo
│   ├── scripts
│   └── src
└── playground (dataset)
    └── data
        └── eval
            ├── pope
            └── vstar
```

### 3. Evaluation 


#### 1. Starting Servers
To start both LMM and Visual Expert servers:

```bash
# Start LMM server (recommend tmux)
conda activate dyfo
CUDA_VISIBLE_DEVICES=0 bash dyfo/scripts/lmm_server/<qwen/llava>_server.sh 
```
```bash
# Start Visual Expert server (recommend tmux)
conda activate dyfo 
CUDA_VISIBLE_DEVICES=1 bash dyfo/scripts/expert_server/start_server.sh 
```

#### 2. Collaborative Inference
For [POPE](https://github.com/RUCAIBox/POPE) evaluation:
- Batch testing (all 9 sub-datasets about 6~7h):
```bash
conda activate dyfo
CUDA_VISIBLE_DEVICES=0 bash dyfo/scripts/pope/<qwen/llava>_batch.sh
```
- Single dataset testing (about 40~50mins):
```bash
# take gqa_random for example 
# other datasets: <coco/aokvqa/gqa>/<coco/aokvqa/gqa>_pope_<random/popular/adversarial>
conda activate dyfo
CUDA_VISIBLE_DEVICES=0 bash dyfo/scripts/pope/stream_pope_<qwen/llava>.sh mcts False gqa/gqa_pope_random
```

For [V*](https://github.com/penghao-wu/vstar) evaluation (about 30mins):
```bash
conda activate dyfo
CUDA_VISIBLE_DEVICES=0 bash dyfo/scripts/vstar/stream_vstar_<qwen/llava>.sh mcts False
```



## 🏅 Experiments

The experimental results of **new version** are shown below:


| Dataset   | Type       | Model      | Accuracy↑ | Precision | Recall  | F1Score↑ |
| :----- | :-------- | :-------- | :-------- | :-------- | :------ | :-------- |
| MSCOCO | random    | LLaVA1.5  | 92.03     | 93.94     | 89.87   | 91.86     |
|        |           | Qwen2-VL  | 92.33     | 96.49     | 87.87   | 91.97     |
|        | popular   | LLaVA1.5  | 88.77     | 87.69     | 90.20   | 88.93     |
|        |           | Qwen2-VL  | 89.20     | 90.50     | 87.60   | 89.02     |
|        | adversarial| LLaVA1.5  | 83.33     | 79.66     | 89.53   | 84.31     |
|        |           | Qwen2-VL  | 86.87     | 86.62     | 87.20   | 86.91     |
| A-OKVQA | random    | LLaVA1.5  | 90.43     | 87.42     | 94.47   | 90.80     |
|        |           | Qwen2-VL  | 92.33     | 92.05     | 92.67   | 92.36     |
|        | popular   | LLaVA1.5  | 84.83     | 79.04     | 94.80   | 86.21     |
|        |           | Qwen2-VL  | 89.17     | 87.07     | 92.00   | 89.47     |
|        | adversarial| LLaVA1.5  | 75.17     | 68.11     | 94.67   | 79.22     |
|        |           | Qwen2-VL  | 82.13     | 76.78     | 92.13   | 83.76     |
| GQA    | random    | LLaVA1.5  | 90.03     | 87.27     | 93.73   | 90.39     |
|        |           | Qwen2-VL  | 88.60     | 94.74     | 81.73   | 87.76     |
|        | popular   | LLaVA1.5  | 80.33     | 74.00     | 93.53   | 82.63     |
|        |           | Qwen2-VL  | 85.87     | 88.93     | 81.93   | 85.29     |
|        | adversarial| LLaVA1.5  | 75.03     | 68.33     | 93.33   | 78.90     |
|        |           | Qwen2-VL  | 81.87     | 82.12     | 81.47   | 81.79     |

*table 1. Results on POPE for MSCOCO/AOKVQA/GQA with LLaVA1.5 and Qwen2-VL.*

| Dataset | Model      | Attribute↑ | Spatial↑ | Overall↑ |
| :------ | :-------- | :---------- | :-------- | :-------- |
| V*      | DyFo-L    | 65.22       | 57.89     | 62.30     |
|         | DyFo-Q    | 80.87       | 78.95     | 80.10     |

*table 2. Results on V\*. DyFo-L and DyFo-Q represent our method with LLaVA1.5 and Qwen2-VL, respectively.*



- **Please refer to [our paper](https://arxiv.org/abs/2504.14920) for detailed experimental results.**


## 📑 Citation
If you find our project useful, we hope you can star our repo and cite our paper as follows:
```
@misc{li2025dyfotrainingfreedynamicfocus,
      title={DyFo: A Training-Free Dynamic Focus Visual Search for Enhancing LMMs in Fine-Grained Visual Understanding}, 
      author={Geng Li and Jinglin Xu and Yunzhen Zhao and Yuxin Peng},
      year={2025},
      eprint={2504.14920},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.14920}, 
}
```

## 📝 Related Projects
- [V*](https://github.com/penghao-wu/vstar): Guided Visual Search as a Core Mechanism in Multimodal LLMs
- [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL): Enhancing Vision-Language Model's Perception of the World at Any Resolution
- [LLaVA 1.5](https://github.com/haotian-liu/LLaVA): Improved Baselines with Visual Instruction Tuning
- [LangSam](https://github.com/luca-medeiros/lang-segment-anything): Language Segment-Anything (Cool Expert!)
- [vLLM](https://github.com/vllm-project/vllm): Efficient Memory Management for Large Language Model Serving with PagedAttention
- [VCD](https://github.com/DAMO-NLP-SG/VCD): Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding