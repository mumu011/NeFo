import json
import random
import re
import argparse
import logging
import torch
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def replace_case_insensitive(text: str, old: str, new: str) -> str:
    """
    大小写不敏感的字符串替换
    """
    pattern = re.compile(re.escape(old), re.IGNORECASE)
    return pattern.sub(new, text)

def find_attr_phrase(objects_with_attributes: list, attr: str) -> str:
    """
    从objects_with_attributes中找到包含指定属性的短语
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
    """
    if not objects_with_states:
        return ""

    state_lower = state.lower()
    for obj_state in objects_with_states:
        obj_state_lower = obj_state.lower()
        if state_lower in obj_state_lower:
            return obj_state

    return ""

def create_validation_questions(dataset_json, output_json, seed=42):
    """
    从数据集JSON文件中提取negative_entities字段，生成验证问题
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

    # 提取所需字段并生成验证数据
    extracted_data = []
    for item in data:
        file_name = item.get('filename')
        imgid = item.get('imgid')
        caption = item.get('caption')
        caption_extraction = item.get('caption_extraction')
        negative_entities = item.get('negative_entities')

        if not negative_entities:
            continue

        # 安全地选择正例元素
        attributes_list = safe_get(caption_extraction, 'attributes')
        random_positive_attr = random.choice(attributes_list) if attributes_list else None

        states_list = safe_get(caption_extraction, 'states')
        random_positive_state = random.choice(states_list) if states_list else None

        # 收集该条目的所有验证问题
        validation_questions = []

        # 1. 处理所有 Negative Objects
        neg_objects_list = safe_get(negative_entities, 'objects')
        for neg_obj in neg_objects_list:
            question = f"Is there a {neg_obj} seen in the image?"
            validation_questions.append({
                'question': question,
                'answer': ['Yes', 'No'],
                'type': 'object_negative_validation',
                'target': neg_obj,
                'entity_value': neg_obj
            })

        # 2. 处理所有 Negative Attributes
        neg_attributes_list = safe_get(negative_entities, 'attributes')
        if neg_attributes_list and attributes_list and random_positive_attr is not None:
            objects_with_attributes = safe_get(caption_extraction, 'objects_with_attributes')
            if objects_with_attributes:
                attr_phrase = find_attr_phrase(objects_with_attributes, random_positive_attr)
            else:
                attr_phrase = ""
            
            if attr_phrase:
                for neg_attr in neg_attributes_list:
                    negative_attr_phrase = attr_phrase.replace(random_positive_attr.lower(), neg_attr.lower())
                    question = f"Is there a {negative_attr_phrase} seen in the image?"
                    validation_questions.append({
                        'question': question,
                        'answer': ['Yes', 'No'],
                        'type': 'attribute_negative_validation',
                        'target': negative_attr_phrase,
                        'entity_value': neg_attr
                    })

        # 3. 处理所有 Negative States
        neg_states_list = safe_get(negative_entities, 'states')
        if neg_states_list and states_list and random_positive_state is not None:
            objects_with_states = safe_get(caption_extraction, 'objects_with_states')
            if objects_with_states:
                state_phrase = find_state_phrase(objects_with_states, random_positive_state)
            else:
                state_phrase = ""
            
            if state_phrase:
                for neg_state in neg_states_list:
                    # 注意：这里逻辑参考generate_vqa，但generate_vqa中state的处理比较复杂
                    # generate_vqa中: negative_state_attr = state_phrase.replace(random_positive_state.lower(), random_negative_state.lower())
                    # 这里的命名 negative_state_attr 其实是 "phrase with negative state"
                    
                    negative_state_phrase = state_phrase.replace(random_positive_state.lower(), neg_state.lower())
                    question = f"Is there a {negative_state_phrase} seen in the image?"
                    validation_questions.append({
                        'question': question,
                        'answer': ['Yes', 'No'],
                        'type': 'state_negative_validation',
                        'target': negative_state_phrase,
                        'entity_value': neg_state
                    })

        if not validation_questions:
            continue

        # 构建提取的数据项
        extracted_item = {
            'filename': file_name,
            'imgid': imgid,
            'caption': caption,
            'validation_questions': validation_questions
        }
        extracted_data.append(extracted_item)

    # 保存提取的数据到输出文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    print(f"成功生成 {len(extracted_data)} 条数据的验证问题并保存到 {output_json}")

    return extracted_data

def generate_prompt(question: str):
    """
    构建VQA润色的提示词

    Args:
        question: 问题

    Returns:
        润色提示词
    """
    prompt = f"""
    Task: You will be given a question about an image. Your task is to:
    1. Rephrase the question to improve its flow and make it more engaging.
    2. Extract the main objects mentioned in the question.

    Instructions:
    1. Keep the question concise and clear.
    2. Preserve the original meaning.
    3. Do not introduce any new information.
    4. Output the result in JSON format with keys "question" and "objects".

    Here are some examples:
    Question: "Is there a trees seen in the image?"
    Output: {{"question": "In the image, are there any trees seen?", "objects": ["trees"]}}

    Question: "Is there a aircraft of small seen in the image?"
    Output: {{"question": "In the image, is there a small aircraft seen?", "objects": ["aircraft"]}}

    Question: "Is the child standing on a lawn?"
    Output: {{"question": "Is the child standing on a lawn?", "objects": ["child", "lawn"]}}

    Question: "Is the elephant wearing a saddle?"
    Output: {{"question": "Is the elephant wearing a saddle?", "objects": ["elephant", "saddle"]}}

    Question: "Is the garage located in front of the building?"
    Output: {{"question": "Is the garage located in front of the building?", "objects": ["garage"]}}

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

def polish_validation_questions(input_json, output_json, model_path="/mnt/nvme0/wj/Model/Llama-3.1-8B-Instruct", limit: int = None, dtype: str = None):
    """
    使用vLLM和Llama模型对验证问题进行润色
    """
    try:
        # 读取数据
        print("正在读取数据文件...")
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"共读取到 {len(data)} 条数据")

        # 如果提供了 limit 参数，则只处理前 limit 条数据
        if limit is not None and isinstance(limit, int) and limit > 0:
            total_to_process = min(limit, len(data))
            print(f"将只润色前 {total_to_process} 条数据")
        else:
            total_to_process = len(data)

        # 初始化vLLM模型
        print(f"正在加载模型: {model_path}")
        def load_model(model_name: str, num_gpus: int = 1, dtype: str = None):
            logger.info(f"正在加载模型: {model_name}")
            
            if num_gpus <= 0:
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

            validation_questions = item.get('validation_questions', [])
            if not validation_questions:
                continue

            # 生成提示词
            prompts = []
            for q in validation_questions:
                question = q.get('question', '')
                prompts.append(generate_prompt(question))

            # 推理
            outputs = llm.generate(prompts, sampling_params)

            # 处理结果
            for idx, output in enumerate(outputs):
                generated_text = output.outputs[0].text.strip()
                
                try:
                    # 尝试查找JSON块
                    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                    if json_match:
                        result = json.loads(json_match.group(0))
                        polished_question = result.get('question', '')
                        objects = result.get('objects', [])
                        
                        if polished_question:
                            clean_question = process_polished_answer(polished_question)
                            validation_questions[idx]['question'] = clean_question
                        
                        if objects:
                            validation_questions[idx]['object'] = objects
                    else:
                        # 如果解析失败，回退到原始处理方式
                        clean_question = process_polished_answer(generated_text)
                        validation_questions[idx]['question'] = clean_question
                        
                except Exception as e:
                    print(f"处理输出时出错: {e}")

            processed_count += 1

        print(f"完成处理，共处理了 {processed_count} 个数据项")

        # 保存结果
        print(f"正在保存结果到 {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    except Exception as e:
        print(f"润色过程中出现错误: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='生成或润色验证问题')
    parser.add_argument('--task', '-t', choices=['generate', 'polish'], default='generate',
                       help='任务类型：generate（生成验证问题）或 polish（润色验证问题）')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入JSON文件路径')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='输出JSON文件路径')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--model', '-m', type=str, default="/mnt/nvme0/wj/Model/Llama-3.1-8B-Instruct",
                       help='润色任务使用的模型路径')
    parser.add_argument('--limit', '-l', type=int, default=None,
                       help='仅润色前N条数据')
    parser.add_argument('--dtype', '-d', type=str, default=None,
                       help='模型数据类型')

    args = parser.parse_args()

    if args.task == 'generate':
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"随机种子: {args.seed}")
        create_validation_questions(args.input, args.output, seed=args.seed)
    
    elif args.task == 'polish':
        print(f"输入文件: {args.input}")
        print(f"输出文件: {args.output}")
        print(f"模型路径: {args.model}")
        polish_validation_questions(
            args.input,
            args.output,
            args.model,
            args.limit,
            args.dtype
        )

if __name__ == "__main__":
    main()

