#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用vLLM加载Qwen-8B模型来检测描述中是否含有修饰词
"""

import json, ast
import argparse
import random
import csv
from typing import Dict, List, Any
from vllm import LLM, SamplingParams
import logging
import torch

# 设置随机种子以确保结果可重现
random.seed(35)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", num_gpus: int = 1):
    logger.info(f"正在加载模型: {model_name}")
    
    if num_gpus <= 0: # num_gpus <= 0 表示使用所有可用GPU
        tp_size = torch.cuda.device_count()
        logger.info(f"检测到 {tp_size} 个GPU，将使用所有GPU进行张量并行处理。")
    else:
        tp_size = num_gpus
        logger.info(f"将使用 {tp_size} 个GPU进行张量并行处理。")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.8
    )
    
    logger.info("模型加载完成")

    return llm

def generate_extraction_prompt(caption: str):
     # 构建提示词
    prompt = f"""Task: You are given a caption that describes an image. Your goal is to identify and extract objects, attributes, and states mentioned in the caption.

        Definitions:
        1.objects: Tangible, concrete items that could be visually represented in an image
        2.attributes: Descriptive properties of objects (e.g., color, shape, texture; do NOT include size-related attributes like large, small; may be empty if no attribute is described)
        3.states: Actions or conditions of objects (e.g., parked, running, sitting; may be empty if no action is described)
        4.objects_with_attributes: Objects combined with their attributes (e.g., "white plane", "red mark"; may be empty if no object has attributes)
        5.objects_with_states: Objects combined with their states (e.g., "plane parked on an open area", "waves washed ashore"; may be empty if no object has states)

        Instructions:
        1. Identify and list all tangible objects that could be visually represented in an image.
        2. Extract descriptive attributes of objects (e.g., color, shape, texture). Do NOT extract size-related attributes.
        3. Extract states describing actions or conditions of objects (e.g., parked, running, sitting).
        4. Combine objects with their attributes when applicable (e.g., "white plane").
        5. Combine objects with their states when applicable, including contextual information (e.g., "plane parked on an open area").
        6. Output as a dictionary with five keys ('objects', 'attributes', 'states', 'objects_with_attributes', 'objects_with_states'), each containing a list.
        7. List items in the order they appear in the caption.
        8. Do not include any additional text.

        Output format:
        {{
        "objects": [list of objects],
        "attributes": [list of attributes],
        "states": [list of states],
        "objects_with_attributes": [list of objects with attributes],
        "objects_with_states": [list of objects with states]
        }}

        Here are some examples:

        Caption: "A white plane with a red mark parked on an open area."
        Output: {{"objects": ["plane", "mark"], "attributes": ["white", "red"], "states": ["parked"], "objects_with_attributes": ["white plane", "red mark"], "objects_with_states": ["plane parked on an open area"]}}

        Caption: "An airport with a runway on the farmland."
        Output: {{"objects": ["airport", "runway", "farmland"], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}}

        Caption: "The beach with brown sand and the white waves washed ashore."
        Output: {{"objects": ["beach", "sand", "waves"], "attributes": ["brown", "white"], "states": ["washed ashore"], "objects_with_attributes": ["brown sand", "white waves"], "objects_with_states": ["waves washed ashore"]}}

        Caption: "A basketball court next to a parking lot and some trees beside."
        Output: {{"objects": ["basketball court", "parking lot", "trees"], "attributes": [], "states": ["next to"], "objects_with_attributes": [], "objects_with_states": ["basketball court next to a parking lot"]}}

        Caption: "A lot of dense chaparrals grow in the desert."
        Output: {{"objects": ["chaparrals", "desert"], "attributes": ["dense"], "states": ["grow"], "objects_with_attributes": ["dense chaparrals"], "objects_with_states": ["chaparrals grow in the desert"]}}

        Caption: "Residential areas are mostly houses, With no open space ."
        Output: {{"objects": ["houses"], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}}

        Caption: "The sea ice in the middle is larger ."
        Output: {{"objects": ["sea ice"], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}}

        Caption: "Many trees are in the mobile home park ."
        Output: {{"objects": ["trees", "mobile home park"], "attributes": [], "states": ["in the mobile home park"], "objects_with_attributes": [], "objects_with_states": ["trees in the mobile home park"]}}

        Caption: "One large vehicle is positioned towards the middle-left side of the frame ."
        Output: {{"objects": ["vehicle"], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}}

        Caption: "There are two small vehicles captured in the scene ."
        Output: {{"objects": ["vehicles"], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}}

        Caption: {caption}
        Output:"""

    return prompt

def generate_negative_objects_prompt(caption, relevant_objects):
    """
    Generate a prompt for extracting negative objects based on a caption and a list of relevant objects.
    """
    prompt_template = f"""Task: You are given (1) a caption describing an image and (2) a structured list of relevant elements extracted from the caption:
        - objects: Tangible, concrete items that could be visually represented in an image.
        - attributes: Descriptive properties of objects (e.g., color, shape, texture). Do NOT include size-related attributes. May be empty if none are described.
        - states: Actions or conditions of objects (e.g., parked, running, sitting). May be empty if none are described.

        Your goal is to generate NEW NEGATIVE elements in the same three categories. “Negative” means: items/properties/states that are RELATED to the scene and plausible in context, but are ABSENT from BOTH the caption and the provided relevant elements.

        Instructions:
        1) Use the caption and relevant elements to infer the scene context.
        2) Propose negative items that are related yet unmentioned/absent:
        - objects: co-occurring or contextually related objects not listed.
        - attributes: plausible but unmentioned properties (avoid those already listed and avoid size-related attributes).
        - states: plausible but unmentioned actions/conditions (avoid those already listed).
        3) Keep each list concise and specific (aim for 1-3 items per non-empty category; use [] if nothing reasonable can be inferred).
        4) Do NOT repeat anything already present in the caption or relevant elements.
        5) Be creative but realistic - negative elements should be plausible in the scene context.
        6) Use lowercase single-word or short noun/verb phrases where natural. No explanations or extra text.
        7) Do not include any additional text.

        Output format:
        {{
        "objects": [list of objects],
        "attributes": [list of attributes],
        "states": [list of states]
        }}  

        Examples:

        Example 1
        Caption: "A white plane with the red mark parked on the open area."
        Relevant Elements: {{objects":["plane","mark"],"attributes":["white","red"],"states":["parked"]}}
        Output:
        {{
        "objects": ["cars","shadow"],
        "attributes": ["black","yellow"],
        "states": ["flying"]
        }}

        Example 2
        Caption: "An airport with a runway on the farmland ."
        Relevant Elements: {{"objects": ["airport", "runway", "farmland"], "attributes": [], "states": []}}
        Output:
        {{
        "objects": ["fuel truck", "railroad", "building"],
        "attributes": [],
        "states": []
        }}

        Example 3
        Caption: "The beach with brown sand and the white waves washed ashore ."
        Relevant Elements: {{"objects": ["beach","sand","waves"], "attributes": ["brown","white"], "states": ["washed ashore"]}}
        Output:
        {{   
        "objects": ["people", "tree", "bench"],
        "attributes": ["black", "green", "red"],
        "states": []
        }}

        Example 4
        Caption: "A basketball court next to a parking lot and some trees beside ."
        Relevant Elements: {{"objects": ["basketball court", "parking lot", "trees"], "attributes": [], "states": [next to]}}
        Output:
        {{
        "objects": ["cars", "people", "grass"],
        "attributes": [],
        "states": ["away from"]
        }}

        Example 5
        Caption: "A lot of dense chaparrals grow in the desert ."
        Relevant Elements: {{"objects": ["chaparrals ", "desert"], "attributes": ["dense"], "states": ["grow"]}}
        Output:
        {{
        "objects": ["trees", "rabbit", "water"],
        "attributes": [],
        "states": []
        }}

        Example 6
        Caption: "One large vehicle is positioned towards the middle-left side of the frame ."
        Relevant Elements: {{"objects": ["vehicle"], "attributes": [], "states": []}}
        Output:
        {{
        "objects": ["building", "tree", "road"],
        "attributes": [],
        "states": []
        }}

        Caption: {caption}
        Relevant Elements: {relevant_objects}

        Output:"""

    return prompt_template


def extract_entities_from_caption(llm, caption: str) -> Dict[str, List[str]]:
    """从描述中提取对象、属性和状态"""
    prompt = generate_extraction_prompt(caption)

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=200,
        stop=["</s>", "\n\n"]
    )

    logger.info(f"正在处理描述: {caption[:50]}...")

    try:
        # 生成响应
        outputs = llm.generate([prompt], sampling_params)

        if not outputs:
            logger.warning("模型未返回任何输出")
            return {"objects": [], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}

        response = outputs[0].outputs[0].text.strip()

        # 尝试解析JSON响应
        try:
            # 清理响应文本，确保是有效的JSON
            response = response.strip()

            result = json.loads(json.dumps(ast.literal_eval(response), ensure_ascii=False))
            logger.info(f"成功提取实体: {result}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 响应内容: {response}")
            return {"objects": [], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}

    except Exception as e:
        logger.error(f"提取过程中出现错误: {e}")
        return {"objects": [], "attributes": [], "states": []}

def process_dataset_with_extraction(input_file: str, output_file: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """处理数据集并提取实体"""
    logger.info("开始加载模型...")
    llm = load_model(model_name, num_gpus=0)

    logger.info(f"开始读取输入文件: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 判断数据结构
        data = []
        if isinstance(raw_data, dict):
            # 扁平化数据：将所有类别下的描述合并成一个列表
            for category_name, items in raw_data.items():
                for item in items:
                    item['category_name'] = category_name # 添加类别名称
                    data.append(item)
        elif isinstance(raw_data, list):
            # 已经是列表结构
            data = raw_data
        else:
            logger.error(f"不支持的数据格式: {type(raw_data)}")
            return

    except Exception as e:
        logger.error(f"读取或处理文件失败: {e}")
        return

    # 移除CSV过滤逻辑，直接使用输入文件中的数据
    logger.info(f"共加载 {len(data)} 条记录")

    # # 随机挑选1000条数据进行处理
    # if len(data) > 1000:
    #     data = random.sample(data, 1000)
    #     logger.info(f"随机选择1000条记录进行处理")
    # else:
    #     logger.info(f"数据总量不足1000条，使用全部{len(data)}条记录")

    processed_data = []
    total_processed = 0
    extraction_count = 0

    for item in data:
        total_processed += 1

        if total_processed % 100 == 0:
            logger.info(f"已处理 {total_processed}/{len(data)} 条记录")

        # 使用raw字段作为描述
        caption = item.get('raw', '').strip()

        processed_item = {
            "filename": item.get('filename', ''),
            "imgid": item.get('imgid', ''),
            "caption": caption,
            "category_name": item.get('category_name', item.get('category', '')) # 兼容category字段
        }

        if caption:
            # 对描述进行实体提取
            entities = extract_entities_from_caption(llm, caption)
            processed_item["caption_extraction"] = entities

            # 统计提取到的实体数量
            total_entities = len(entities.get('objects', [])) + \
                           len(entities.get('attributes', [])) + \
                           len(entities.get('states', []))

            if total_entities > 0:
                extraction_count += 1
                processed_data.append(processed_item)
            else:
                # 不包含实体提取结果的跳过
                continue
        else:
            processed_item["caption_extraction"] = {"objects": [], "attributes": [], "states": [], "objects_with_attributes": [], "objects_with_states": []}
            # 不包含实体提取结果的跳过
            continue

    logger.info(f"处理完成，共处理 {total_processed} 条记录，其中 {extraction_count} 条包含实体提取结果")

    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")

def generate_negative_objects(llm, caption: str, relevant_entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """生成负对象（negative objects）"""
    prompt = generate_negative_objects_prompt(caption, relevant_entities)

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.1,  
        top_p=0.9,
        max_tokens=300,
        stop=["</s>", "\n\n"]
    )

    logger.info(f"正在为描述生成负对象: {caption[:50]}...")

    try:
        # 生成响应
        outputs = llm.generate([prompt], sampling_params)

        if not outputs:
            logger.warning("模型未返回任何输出")
            return {"objects": [], "attributes": [], "states": []}

        response = outputs[0].outputs[0].text.strip()

        # 尝试解析JSON响应
        try:
            # 清理响应文本，确保是有效的JSON
            response = response.strip()

            result = json.loads(json.dumps(ast.literal_eval(response), ensure_ascii=False))
            logger.info(f"成功生成负对象: {result}")
            return result

        except json.JSONDecodeError as e2:
            logger.error(f"原始响应: {response}")
            return {"objects": [], "attributes": [], "states": []}

    except Exception as e:
        logger.error(f"生成负对象过程中出现错误: {e}")
        return {"objects": [], "attributes": [], "states": []}

def process_negative_objects_generation(input_file: str, output_file: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """处理数据集并生成负对象"""
    logger.info("开始加载模型...")
    llm = load_model(model_name, num_gpus=0)

    logger.info(f"开始读取输入文件: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return

    logger.info(f"数据加载完成，共 {len(data)} 条记录")

    processed_data = []
    total_processed = 0
    negative_generation_count = 0

    for item in data:
        total_processed += 1

        if total_processed % 10 == 0:
            logger.info(f"已处理 {total_processed}/{len(data)} 条记录")

        processed_item = item.copy()

        # 获取描述和实体信息
        caption = item.get('caption', '')
        entities = item.get('caption_extraction', {})

        if caption and entities and entities.get('objects'):
            # 生成负对象
            negative_entities = generate_negative_objects(llm, caption, entities)

            # 添加负对象结果到项目中
            processed_item['negative_entities'] = negative_entities

            # 统计生成的负对象数量
            total_negative_entities = len(negative_entities.get('objects', [])) + \
                                   len(negative_entities.get('attributes', [])) + \
                                   len(negative_entities.get('states', []))

            if total_negative_entities > 0:
                negative_generation_count += 1

        processed_data.append(processed_item)

    logger.info(f"处理完成，共处理 {total_processed} 条记录，其中 {negative_generation_count} 条包含负对象生成结果")

    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="从描述中提取对象、属性和状态，或生成负对象")
    parser.add_argument('--input', '-i', required=True, help='输入JSON文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出JSON文件路径')
    parser.add_argument('--model', '-m', default="Qwen/Qwen2.5-7B-Instruct", help='模型名称')
    parser.add_argument('--task', '-t', choices=['extract', 'negative'], default='extract',
                       help='任务类型：extract（实体提取）或 negative（负对象生成）')

    args = parser.parse_args()

    if args.task == 'extract':
        logger.info("开始执行实体提取任务...")
        process_dataset_with_extraction(args.input, args.output, args.model)
    elif args.task == 'negative':
        logger.info("开始执行负对象生成任务...")
        process_negative_objects_generation(args.input, args.output, args.model)

if __name__ == "__main__":
    main()


