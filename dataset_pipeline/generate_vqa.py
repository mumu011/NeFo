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

def find_state_phrase(objects_with_states: list, state: str) -> str:
    """
    从objects_with_states中找到包含指定状态的短语

    Args:
        objects_with_states: 带有状态的对象列表
        state: 要查找的状态

    Returns:
        包含状态的短语，如果找不到则返回空字符串
    """
    if not objects_with_states:
        return ""

    state_lower = state.lower()
    for obj_state in objects_with_states:
        obj_state_lower = obj_state.lower()
        if state_lower in obj_state_lower:
            return obj_state

    return ""

def generate_prompt(question: str):
    """
    构建VQA润色的提示词

    Args:
        question: 问题
        answer: 答案

    Returns:
        润色提示词
    """
    # TODO: 根据VQA任务特点设计合适的润色提示词
    prompt = f"""
    Task: You will be given a question about an image. Your task is to rephrase the question to improve their flow and make them more engaging.

    Instructions:
    1. Keep the question concise and clear.
    2. Preserve the original meaning.
    3. Do not introduce any new information.
    4. Do not include any additional text.
    5. Preserve negation forms in questions - keep negative words such as "no", "not", "without", or "non-" in their original form and context.

    Here are some examples:
    Question: "Is there a residential not away from the road seen in the image?"
    Output: "In the image, is there a residence visible that is not away from the road?"

    Question: "Is there a cars without road seen in the image?"
    Output: "In the image, are there any cars without road seen?"

    Question: "Is there a non-red cars seen in the image?"
    Output: "In the image, is there a non-red car seen?"

    Question: {question}
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

def create_vqa(dataset_json, output_json, seed=42):
    """
    从数据集JSON文件中提取caption、caption_extraction和negative_entities字段，生成VQA数据

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

    def safe_get(data_dict, key):
        """安全地获取字典中的值"""
        if not data_dict:
            return []
        value = data_dict.get(key, [])
        if not isinstance(value, list):
            return []
        return value

    # 提取所需字段并生成VQA数据
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

        # 为每个数据项初始化VQA问答对列表
        item_qa_pairs = []

        # 初始化否定实体变量
        negative_object = None
        negative_attr = None
        negative_state = None

        # TODO: 根据objects生成VQA问答对
        if safe_get(negative_entities, 'objects') and objects_list and random_positive_object is not None:
            objects_list = safe_get(negative_entities, 'objects')
            if objects_list:
                random_negative_object = random.choice(objects_list)

                # TODO: 设计基于object的问题和答案
                # positive
                question = f"Is there a {random_positive_object} without {random_negative_object} seen in the image?"
                answer = f"Yes"
                item_qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'object_negative'
                })

                # negative
                question = f"Is there a {random_negative_object} without {random_positive_object} seen in the image?"
                answer = f"No"
                item_qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'object_negative'
                })

                # original
                # question = f"Is there a {random_positive_object} with {random_negative_object} seen in the image?"
                question = f"Is there a {random_positive_object} seen in the image?"
                # answer = f"No"
                answer = f"Yes"
                item_qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'object_positive'
                })

                # question = f"Is there a {random_negative_object} with {random_positive_object} seen in the image?"
                question = f"Is there a {random_negative_object} seen in the image?"
                answer = f"No"
                item_qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'type': 'object_positive'
                })

                # # 待验证问题
                # question = f"Is there a {random_negative_object} seen in the image?"
                # item_qa_pairs.append({
                #     'question': question,
                #     'answer': ['Yes', 'No'],
                #     'type': 'object_negative_validation'
                # })

                negative_object = random_negative_object

        
        # TODO: 根据attributes生成VQA问答对
        if safe_get(negative_entities, 'attributes') and attributes_list and random_positive_attr is not None:
            attributes_list = safe_get(negative_entities, 'attributes')
            if attributes_list:
                random_negative_attr = random.choice(attributes_list)

                # TODO: 设计基于attribute的问题和答案
                # 否定属性 - 从objects_with_attributes中找到包含positive_attr的短语，用negative_attr替换
                objects_with_attributes = safe_get(caption_extraction, 'objects_with_attributes')
                if objects_with_attributes:
                    attr_phrase = find_attr_phrase(objects_with_attributes, random_positive_attr)
                else:
                    attr_phrase = ""
                if attr_phrase:
                    negative_attr = attr_phrase.replace(random_positive_attr.lower(), random_negative_attr.lower())
                    # positive
                    question = f"Is there a non-{negative_attr} seen in the image?"
                    answer = f"Yes"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'attribute_negative'
                    })
                    # negative
                    question = f"Is there a non-{attr_phrase} seen in the image?"
                    answer = f"No"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'attribute_negative'
                    })

                    # original
                    question = f"Is there a {attr_phrase} seen in the image?"
                    answer = f"Yes"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'attribute_positive'
                    })

                    question = f"Is there a {negative_attr} seen in the image?"
                    answer = f"No"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'attribute_positive'
                    })

                    # # 待验证问题
                    # question = f"Is there a {negative_attr} seen in the image?"
                    # item_qa_pairs.append({
                    #     'question': question,
                    #     'answer': ['Yes', 'No'],
                    #     'type': 'attribute_negative_validation'
                    # })

                negative_attr = random_negative_attr

        # TODO: 根据states生成VQA问答对
        if safe_get(negative_entities, 'states') and states_list and random_positive_state is not None:
            states_list = safe_get(negative_entities, 'states')
            if states_list:
                random_negative_state = random.choice(states_list)

                # 设计基于state的问题和答案
                # positive - 询问是否存在处于positive_state状态的对象
                objects_with_states = safe_get(caption_extraction, 'objects_with_states')
                if objects_with_states:
                    state_phrase = find_state_phrase(objects_with_states, random_positive_state)
                else:
                    state_phrase = ""
                if state_phrase:
                    positive_state_attr = state_phrase.replace(random_positive_state.lower(), "not " + random_negative_state.lower())
                    negative_state_attr = state_phrase.replace(random_positive_state.lower(), "not " + random_positive_state.lower())
                    # positive
                    question = f"Is there a {positive_state_attr} seen in the image?"
                    answer = f"Yes"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'state_negative'
                    })
                    # negative
                    question = f"Is there a {negative_state_attr} seen in the image?"
                    answer = f"No"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'state_negative'
                    })

                    # original
                    question = f"Is there a {state_phrase} seen in the image?"  
                    answer = f"Yes"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'state_positive'
                    })

                    negative_state_attr = state_phrase.replace(random_positive_state.lower(), random_negative_state.lower())
                    question = f"Is there a {negative_state_attr} seen in the image?"
                    answer = f"No"
                    item_qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'state_positive'
                    })

                negative_state = random_negative_state

        # TODO: 根据需求决定是否跳过没有足够问答对的数据项
        # if len(item_qa_pairs) < 1:
        #     continue

        # 构建提取的数据项
        extracted_item = {
            'filename': file_name,
            'imgid': imgid,
            'caption': caption,
            'caption_extraction': caption_extraction,
            'negative_entities': negative_entities,
            'qa_pairs': item_qa_pairs,  # VQA问答对列表
            'negative_object': negative_object,
            'negative_attr': negative_attr,
            'negative_state': negative_state
        }
        extracted_data.append(extracted_item)

    # 保存提取的数据到输出文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"成功提取 {len(extracted_data)} 条数据并保存到 {output_json}")

    return extracted_data


def polish_vqa_with_vllm(input_json, output_json, model_path="/mnt/nvme0/wj/Model/Llama-3.1-8B-Instruct", limit: int = None, dtype: str = None):
    """
    使用vLLM和Llama模型对VQA问答对进行润色

    Args:
        input_json (str): 输入JSON文件路径
        output_json (str): 输出JSON文件路径
        model_path (str): Llama模型路径
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
            print(f"处理数据项 {item_index + 1}/{total_to_process}: {item.get('filename', 'unknown')}")

            # 为当前数据项收集需要润色的问答对
            qa_pairs = item.get('qa_pairs', [])

            if not qa_pairs:
                print(f"  跳过 - 没有需要润色的问答对")
                continue

            # 生成提示词并进行润色
            prompts = []
            for qa_pair in qa_pairs:
                question = qa_pair.get('question', '')
                prompts.append(generate_prompt(question))

            # 推理当前数据项的问答对
            outputs = llm.generate(prompts, sampling_params)

            # 处理推理结果
            polished_qa_pairs = []
            for idx, output in enumerate(outputs):
                polished_question = output.outputs[0].text.strip()
                # 处理润色后的答案，移除额外的引号和转义字符
                clean_question = process_polished_answer(polished_question)

                # 复制原有的问答对，只更新question
                polished_qa_pair = qa_pairs[idx].copy()
                polished_qa_pair['question'] = clean_question

                polished_qa_pairs.append(polished_qa_pair)

            # 更新当前数据项
            item['qa_pairs'] = polished_qa_pairs

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


def main():
    parser = argparse.ArgumentParser(description='生成VQA数据或进行润色')
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
                       help='任务类型：generate（生成VQA数据）或 polish（润色问答对）')

    args = parser.parse_args()

    if args.task == 'generate':
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"随机种子: {args.seed}")

        # 调用函数提取数据
        print("\n开始生成VQA数据...")
        extracted_data = create_vqa(args.input, args.output, seed=args.seed)

    elif args.task == 'polish':
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"模型路径: {args.model}")

        # 润色问答对
        print("\n开始润色问答对...")
        polished_data = polish_vqa_with_vllm(
            args.input,
            args.output,
            args.model,
            args.limit,
            args.dtype
        )

if __name__ == "__main__":
    main()

