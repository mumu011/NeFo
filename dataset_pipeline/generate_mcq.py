import json
import random
import re
import argparse
from vllm import LLM, SamplingParams
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def replace_case_insensitive(text: str, old: str, new: str) -> str:
    """
    大小写不敏感的字符串替换

    Args:
        text: 原始文本
        old: 要替换的字符串
        new: 替换后的字符串

    Returns:
        替换后的文本
    """
    # 使用正则表达式进行大小写不敏感替换
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, text)

def find_attr_phrase(objects_with_attributes: list, attr: str) -> str:
    """
    从objects_with_attributes中找到包含指定属性的短语

    Args:
        objects_with_attributes: 带有属性的对象列表
        attr: 要查找的属性

    Returns:
        包含属性的短语，如果找不到则返回空字符串
    """
    if not objects_with_attributes:
        return ""

    attr_lower = attr.lower()
    for obj_attr in objects_with_attributes:
        obj_attr_lower = obj_attr.lower()
        if attr_lower in obj_attr_lower:
            return obj_attr

    return ""

def generate_prompt(caption: str):
    # 构建提示词
    prompt = f"""
    Task: You will be given a caption that describes the presence of certain objects, the absence of some objects, or both presence and absence. Your task is to rephrase the caption to improve its flow and make it more engaging.

    Instructions:
    1. Do not introduce any new objects.
    2. Keep the captions concise and clear, while preserving the original meaning.
    3. Do not include any additional text.

    Here are some examples:

    Caption: "There is a white plane on the runway . There is no shadow in the image."
    Output: "There is a white plane on the runway, but there is no shadow in sight."

    Caption: "Four airplanes and two cars are parked on the open place . There is bikes in the image."
    Output: "Four airplanes and two cars are parked on the open place, and bikes are present in the image."

    Caption: "Four airplanes and two cars are parked on the no red place ."
    Output: "Four airplanes and two cars are parked on the non-red place ."

    Caption: "Three airplanes of blue were parked on the airport ."
    Output: "Three blue airplanes were parked on the airport ."

    Caption: "Two planes are not flying the blue house ."
    Output: "Two planes are not flying through the blue house ."

    Caption: {caption}
    Output:"""
    return prompt


def process_polished_answer(answer: str) -> str:
    """
    处理润色后的答案，移除额外的引号和转义字符

    Args:
        answer (str): 润色后的答案字符串

    Returns:
        str: 处理后的干净字符串
    """
    if not answer:
        return answer

    # 移除开头的引号
    if answer.startswith('"'):
        answer = answer[1:]

    # 移除结尾的引号
    if answer.endswith('"'):
        answer = answer[:-1]

    # 处理转义字符
    answer = answer.replace('\\"', '"')  # 移除转义的双引号
    answer = answer.replace('\\n', ' ')  # 将换行符替换为空格
    answer = answer.replace('\\t', ' ')  # 将制表符替换为空格
    answer = answer.replace('\\r', '')   # 移除回车符

    # 清理多余的空格
    answer = ' '.join(answer.split())

    return answer

def create_mcq(dataset_json, output_json, seed=42):
    """
    从数据集JSON文件中提取caption、caption_extraction和negative_entities字段

    Args:
        dataset_json (str): 输入数据集JSON文件路径
        output_json (str): 输出JSON文件路径
        seed (int): 随机种子，用于确保结果可重现，默认值为42
    """
    
    # 设置随机种子
    random.seed(seed)

    # 读取数据集JSON文件
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def select_answer_from_right_answers(item_right_answers):
        """从item_right_answers中根据概率选择一个答案，返回(答案, 索引)"""
        if not item_right_answers:
            return None, None

        # 为不同的答案类型分配不同的概率权重
        # 通常第一个答案是原始caption，第二个是negative，第三个是hybrid
        probabilities = []
        for i, answer in enumerate(item_right_answers):
            if i == 0:  # 原始caption
                probabilities.append(0.3)  # 30% 概率
            elif i == 1:  # negative类型
                probabilities.append(0.3)  # 30% 概率
            elif i == 2:  # hybrid类型
                probabilities.append(0.3)  # 30% 概率
            else:
                probabilities.append(0.1)  # 其他答案10%概率

        # 重新标准化概率
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]

        # 按照概率选择答案
        rand_val = random.random()
        cumulative_prob = 0.0
        selected_idx = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if rand_val <= cumulative_prob:
                selected_idx = i
                break

        # fallback到第一个答案
        if selected_idx >= len(item_right_answers):
            selected_idx = 0

        return item_right_answers[selected_idx], selected_idx

    def select_wrong_answers_from_list(item_wrong_answers):
        """从item_wrong_answers中根据概率选择三个错误答案，返回(答案列表, 原始索引列表)"""
        if not item_wrong_answers:
            return [], []

        # 为不同的错误答案类型分配概率权重
        # 通常前3个是objects类型，后面的依次是attributes和states类型
        probabilities = []
        for i, answer in enumerate(item_wrong_answers):
            if i < 3:  # objects类型的错误答案 (通常有3个)
                if i == 0:
                    probabilities.append(0.3)  # 30% 概率
                elif i == 1:
                    probabilities.append(0.3)  # 30% 概率
                elif i == 2:
                    probabilities.append(0.3)  # 30% 概率
            else:
                probabilities.append(0.1)  # 其他错误答案10%概率

        # 使用加权随机选择选择3个错误答案
        selected_answers = []
        selected_original_indices = []  # 记录原始索引
        remaining_answers = item_wrong_answers.copy()
        remaining_probs = probabilities.copy()
        # 创建原始索引映射
        original_indices = list(range(len(item_wrong_answers)))

        for _ in range(3):
            if not remaining_answers:
                break

            # 重新标准化剩余概率
            total_prob = sum(remaining_probs[:len(remaining_answers)])
            if total_prob > 0:
                normalized_probs = [p / total_prob for p in remaining_probs[:len(remaining_answers)]]
            else:
                normalized_probs = [1.0 / len(remaining_answers)] * len(remaining_answers)

            # 按照概率选择一个答案
            rand_val = random.random()
            cumulative_prob = 0.0
            selected_idx = 0
            for i, prob in enumerate(normalized_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_idx = i
                    break

            # 添加选中的答案并从剩余列表中移除
            selected_answers.append(remaining_answers[selected_idx])
            selected_original_indices.append(original_indices[selected_idx])
            remaining_answers.pop(selected_idx)
            remaining_probs.pop(selected_idx)
            original_indices.pop(selected_idx)

        return selected_answers, selected_original_indices

    def safe_get(data_dict, key):
        """安全地获取字典中的值"""
        if not data_dict:
            return []
        value = data_dict.get(key, [])
        if not isinstance(value, list):
            return []
        return value

    # 提取所需字段
    extracted_data = []
    for item in data:
        file_name = item.get('filename')
        imgid = item.get('imgid')
        caption = item.get('caption')
        caption_extraction = item.get('caption_extraction')
        negative_entities = item.get('negative_entities')

        # 安全地选择正例元素，如果列表为空则设为None
        attributes_list = safe_get(caption_extraction, 'attributes')
        random_positive_attr = random.choice(attributes_list) if attributes_list else None

        states_list = safe_get(caption_extraction, 'states')
        random_positive_state = random.choice(states_list) if states_list else None

        objects_list = safe_get(caption_extraction, 'objects')
        random_positive_object = random.choice(objects_list) if objects_list else None

        # 为每个数据项初始化答案列表
        item_right_answers = []
        item_wrong_answers = []
        validation_questions = []

        # 初始化否定实体变量
        negative_object = None
        negative_attr = None
        negative_state = None

        if safe_get(negative_entities, 'objects') and objects_list and random_positive_object is not None:
            objects_list = safe_get(negative_entities, 'objects')
            if objects_list:  # 确保列表不为空
                random_negative_object = random.choice(objects_list)
            
            # positive
            right_answer = caption
            item_right_answers.append(right_answer)
            # negative
            right_answer = f" There is no {random_negative_object} in the image."
            item_right_answers.append(right_answer)
            # hybrid
            right_answer = caption + f" There is no {random_negative_object} in the image."
            item_right_answers.append(right_answer)

            # 构建错误选项
            wrong_answer_objects_1 = caption + f" There is {random_negative_object} in the image."
            wrong_answer_objects_2 = replace_case_insensitive(caption, random_positive_object, random_negative_object)
            wrong_answer_objects_3 = f" There is no {random_positive_object} in the image."

            # 添加到答案列表
            item_wrong_answers.extend([wrong_answer_objects_1, wrong_answer_objects_2, wrong_answer_objects_3])

            # 否定对象
            negative_object = random_negative_object

            # 待验证问题
            question = f"Is there a {random_negative_object} seen in the image?"
            validation_questions.append({
                'question': question,
                'answer': ['Yes', 'No'],
                'type': 'object_negative_validation'
            })
        
        if safe_get(negative_entities, 'attributes') and attributes_list and random_positive_attr is not None:
            attributes_list = safe_get(negative_entities, 'attributes')
            # 确保列表不为空
            if attributes_list:
                random_negative_attr = random.choice(attributes_list)

                # 否定属性 - 在caption中找到包含positive_attr的短语，用negative_attr替换
                objects_with_attributes = safe_get(caption_extraction, 'objects_with_attributes')
                if objects_with_attributes:
                    attr_phrase = find_attr_phrase(objects_with_attributes, random_positive_attr)
                else:
                    attr_phrase = ""
                    
                if attr_phrase:
                    # 将caption中的positive_attr替换为"no negative_attr"
                    right_answer = replace_case_insensitive(caption, random_positive_attr, f"non-{random_negative_attr}")

                    # 构建错误选项
                    wrong_answer_attributes_1 = replace_case_insensitive(caption, random_positive_attr, random_negative_attr)

                    # 添加到答案列表
                    item_right_answers.append(right_answer)
                    item_wrong_answers.append(wrong_answer_attributes_1)

                    negative_attr = attr_phrase.replace(random_positive_attr.lower(), random_negative_attr.lower())

                    # 待验证问题
                    question = f"Is there a {negative_attr} seen in the image?"
                    validation_questions.append({
                        'question': question,
                        'answer': ['Yes', 'No'],
                        'type': 'attribute_negative_validation'
                    })  
        
        
        if safe_get(negative_entities, 'states') and states_list and random_positive_state is not None:
            states_list = safe_get(negative_entities, 'states')
            if states_list:  # 确保列表不为空
                random_negative_state = random.choice(states_list)

                # 将caption中的positive_state替换为"no negative_state"
                right_answer = replace_case_insensitive(caption, random_positive_state, f"not {random_negative_state}")

                # 构建错误选项
                wrong_answer_states_1 = replace_case_insensitive(caption, random_positive_state, random_negative_state)

                # 添加到答案列表
                item_right_answers.append(right_answer)
                item_wrong_answers.append(wrong_answer_states_1)

                # 否定状态
                negative_state = random_negative_state

        if len(item_wrong_answers) < 3:
            continue

        # 从item_right_answers中根据概率选择一个最终的正确答案
        selected_right_answer, right_answer_idx = select_answer_from_right_answers(item_right_answers)

        # 从item_wrong_answers中根据概率选择三个最终的错误答案
        selected_wrong_answers, wrong_answer_indices = select_wrong_answers_from_list(item_wrong_answers)

        # 判断MCQ类型：根据选中的答案索引判断
        # 答案构建顺序：object(0-2), attribute(3), state(4+)
        # 优先级：attribute > state > object
        mcq_type = 'object'  # 默认类型
        
        # 计算各类型答案的起始索引
        object_count = 3 if negative_object is not None else 0
        attr_start_idx = object_count  # attribute答案的起始索引
        state_start_idx = object_count + (1 if negative_attr is not None else 0)  # state答案的起始索引
        
        # 收集所有选中的答案索引
        all_selected_indices = []
        if right_answer_idx is not None:
            all_selected_indices.append(right_answer_idx)
        if wrong_answer_indices:
            all_selected_indices.extend(wrong_answer_indices)
        
        # 检查所有选中的答案，优先检查attribute，然后state
        found_attr = False
        found_state = False
        
        for idx in all_selected_indices:
            if idx is not None:
                # 检查是否是attribute类型
                if negative_attr is not None and idx == attr_start_idx:
                    found_attr = True
                    break
                # 检查是否是state类型
                elif negative_state is not None and idx == state_start_idx:
                    found_state = True
        
        # 根据优先级设置类型：attribute > state > object
        if found_attr:
            mcq_type = 'attribute'
        elif found_state:
            mcq_type = 'state'
        else:
            mcq_type = 'object'

        # 构建提取的数据项
        extracted_item = {
            'filename': file_name,
            'imgid': imgid,
            'caption': caption,
            'caption_extraction': caption_extraction,
            'negative_entities': negative_entities,
            'selected_right_answer': selected_right_answer,  # 根据概率选中的正确答案
            'selected_wrong_answers': selected_wrong_answers,  # 根据概率选中的3个错误答案
            'right_answers': item_right_answers,  # 每个数据项自己的所有正确答案列表
            'wrong_answers': item_wrong_answers,   # 每个数据项自己的所有错误答案列表
            'negative_object': negative_object,  # 每个数据项自己的否定对象
            'negative_attr': negative_attr,  # 每个数据项自己的否定属性
            'negative_state': negative_state,  # 每个数据项自己的否定状态
            'validation_questions': validation_questions,
            'mcq_type': mcq_type  # MCQ类型：object, attribute, 或 state
        }
        extracted_data.append(extracted_item)

    # 保存提取的数据到输出文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"成功提取 {len(extracted_data)} 条数据并保存到 {output_json}")

    return extracted_data


def polish_answers_with_vllm(input_json, output_json, model_path="/mnt/nvme0/wj/Model/Llama-3.1-8B-Instruct", limit: int = None, dtype: str = None):
    """
    使用vLLM和Llama模型对extracted_mcq_data.json中的答案进行润色

    Args:
        input_json (str): 输入JSON文件路径
        output_json (str): 输出JSON文件路径
        model_path (str): Llama模型路径
        limit (int): 仅处理前N条数据
        dtype (str): 模型数据类型
    """
    try:
        # 读取数据
        print("正在读取数据文件...")
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"共读取到 {len(data)} 条数据")

        # 如果提供了 limit 参数，则只处理前 limit 条数据（临时限制）
        if limit is not None and isinstance(limit, int) and limit > 0:
            total_to_process = min(limit, len(data))
            print(f"将只润色前 {total_to_process} 条数据（临时限制）")
        else:
            total_to_process = len(data)

        # 初始化vLLM模型
        print(f"正在加载模型: {model_path}")
        def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct", num_gpus: int = 1, dtype: str = None):
            logger.info(f"正在加载模型: {model_name}")
            
            if num_gpus <= 0: # num_gpus <= 0 表示使用所有可用GPU
                tp_size = torch.cuda.device_count()
                logger.info(f"检测到 {tp_size} 个GPU，将使用所有GPU进行张量并行处理。")
            else:
                tp_size = num_gpus
                logger.info(f"将使用 {tp_size} 个GPU进行张量并行处理。")

            llm_kwargs = {
                'model': model_name,
                'trust_remote_code': True,
                'tensor_parallel_size': tp_size,
                'gpu_memory_utilization': 0.8
            }

            # 如果用户指定 dtype（例如 'half'），将其传递给 LLM（vLLM 支持 'bf16'/'fp16' 等符号）
            if dtype:
                llm_kwargs['dtype'] = dtype

            llm = LLM(**llm_kwargs)
            
            logger.info("模型加载完成")

            return llm

        llm = load_model(model_name=model_path, num_gpus=0, dtype=dtype)

        sampling_params = SamplingParams(
            temperature=0.1,  
            top_p=0.9,
            max_tokens=300,
            stop=["</s>", "\n\n"]
        )
        processed_count = 0

        for item_index, item in enumerate(data[:total_to_process]):
            print(f"处理数据项 {item_index + 1}/{len(data)}: {item.get('filename', 'unknown')}")

            # 为当前数据项收集需要润色的答案
            current_captions = []

            # 添加正确答案
            if 'selected_right_answer' in item and item['selected_right_answer']:
                current_captions.append(item['selected_right_answer'])

            # 添加错误答案
            if 'selected_wrong_answers' in item and item['selected_wrong_answers']:
                for i, wrong_answer in enumerate(item['selected_wrong_answers']):
                    current_captions.append(wrong_answer)

            if not current_captions:
                print(f"  跳过 - 没有需要润色的答案")
                continue

            # 生成提示词
            prompts = []
            for caption_info in current_captions:
                prompts.append(generate_prompt(caption_info))

            # 推理当前数据项的答案
            outputs = llm.generate(prompts, sampling_params)

            # 处理推理结果
            polished_results = []
            for output in outputs:
                polished_caption = output.outputs[0].text.strip()
                # 处理润色后的答案，移除额外的引号和转义字符
                cleaned_caption = process_polished_answer(polished_caption)
                polished_results.append(cleaned_caption)

            # 更新当前数据项
            result_index = 0

            # 更新正确答案
            if 'selected_right_answer' in item and item['selected_right_answer']:
                item['polished_right_answer'] = polished_results[result_index]
                result_index += 1

            # 更新错误答案
            if 'selected_wrong_answers' in item and item['selected_wrong_answers']:
                item['polished_wrong_answers'] = []
                for _ in item['selected_wrong_answers']:
                    item['polished_wrong_answers'].append(polished_results[result_index])
                    result_index += 1

            processed_count += 1

        print(f"完成处理，共处理了 {processed_count} 个数据项")

        # 保存结果
        print(f"正在保存结果到 {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"润色完成！共处理了 {processed_count} 条数据（已更新到原始数据列表中）")
        return data

    except Exception as e:
        print(f"润色过程中出现错误: {e}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成MCQ数据或进行润色')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入JSON文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='输出JSON文件路径')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--model', '-m', type=str, default="/mnt/nvme0/wj/Model/Llama-3.1-8B-Instruct",
                       help='润色任务使用的模型路径')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='仅润色前N条数据（临时限制），例如 --limit 100')
    parser.add_argument('--dtype', '-d', type=str, default=None,
                       help='模型数据类型，例如 bf16 或 half（示例: --dtype=half 使用 float16）')
    parser.add_argument('--task', '-t', choices=['generate', 'polish'], default='generate',
                       help='任务类型：generate（生成MCQ数据）或 polish（润色问答对）')

    args = parser.parse_args()

    if args.task == 'generate':
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"随机种子: {args.seed}")

        # 调用函数提取数据
        print("\n开始生成MCQ数据...")
        extracted_data = create_mcq(args.input, args.output, seed=args.seed)

    elif args.task == 'polish':
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"模型路径: {args.model}")

        # 润色问答对
        print("\n开始润色答案...")
        polished_data = polish_answers_with_vllm(
            args.input,
            args.output,
            args.model,
            args.limit,
            args.dtype
        )