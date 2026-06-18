import json
import random
import re
import argparse
# 尝试导入 vllm，但不强制依赖，以便在没有 vllm 的环境中运行 rule-based 生成
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None
import logging
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_get(data_dict, key):
    """安全地获取字典中的值"""
    if not data_dict:
        return []
    value = data_dict.get(key, [])
    if not isinstance(value, list):
        return []
    return value

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

def create_referring(referring_json, negative_json, output_json, seed=42):
    """
    结合 Referring 数据和 Negative Entities 数据，生成带有否定描述的 Referring Expression
    """
    random.seed(seed)

    print(f"Loading Referring data from {referring_json}...")
    with open(referring_json, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    
    print(f"Loading Negative data from {negative_json}...")
    with open(negative_json, 'r', encoding='utf-8') as f:
        neg_data = json.load(f)
        
    neg_map = {}
    for item in neg_data:
        key = (item.get('filename'), item.get('caption'))
        neg_map[key] = item
    
    extracted_data = []
    match_count = 0
    has_neg_count = 0
    
    for item in tqdm(ref_data, desc="Processing"):
        filename = item.get('image_id')
        caption = item.get('question')
        
        neg_item = neg_map.get((filename, caption))
        
        if not neg_item:
            extracted_data.append(item)
            continue
            
        match_count += 1
        negative_entities = neg_item.get('negative_entities')
        caption_extraction = neg_item.get('caption_extraction')
        
        # 收集所有的否定项 (分开生成多条数据)
        generated_any = False

        # 1. Object Negation
        neg_objects = safe_get(negative_entities, 'objects')
        if neg_objects:
            neg_obj = random.choice(neg_objects)
            new_item_obj = item.copy()
            new_item_obj['original_question'] = caption
            # VQA style: "without {object}"
            new_item_obj['question'] = f"{caption} without {neg_obj} seen."
            new_item_obj['negation_type'] = 'object'
            new_item_obj['negations'] = {'object': neg_obj}
            extracted_data.append(new_item_obj)
            generated_any = True

        # 2. Attribute Negation
        # Refer to generate_mcq.py: replace positive attribute with "non-{negative_attribute}"
        pos_attrs = safe_get(caption_extraction, 'attributes')
        neg_attrs = safe_get(negative_entities, 'attributes')
        objects_with_attributes = safe_get(caption_extraction, 'objects_with_attributes')
        
        if pos_attrs and neg_attrs:
            random_positive_attr = random.choice(pos_attrs)
            random_negative_attr = random.choice(neg_attrs)
            
            # Check if positive attribute is associated with an object
            attr_phrase = ""
            if objects_with_attributes:
                attr_phrase = find_attr_phrase(objects_with_attributes, random_positive_attr)
            
            # If found, prepare for replacement
            if attr_phrase and replace_case_insensitive(caption, random_positive_attr, "") != caption:
                new_item_attr = item.copy()
                new_item_attr['original_question'] = caption
                # MCQ style: replace with "non-{negative_attribute}"
                new_item_attr['question'] = replace_case_insensitive(caption, random_positive_attr, f"non-{random_negative_attr}")
                new_item_attr['negation_type'] = 'attribute'
                new_item_attr['negations'] = {'attribute': random_negative_attr}
                extracted_data.append(new_item_attr)
                generated_any = True

        # 3. State Negation
        # Refer to generate_mcq.py: replace positive state with "not {negative_state}"
        pos_states = safe_get(caption_extraction, 'states')
        neg_states = safe_get(negative_entities, 'states')
        if pos_states and neg_states:
            random_positive_state = random.choice(pos_states)
            random_negative_state = random.choice(neg_states)

            if replace_case_insensitive(caption, random_positive_state, "") != caption:
                new_item_state = item.copy()
                new_item_state['original_question'] = caption
                # MCQ style: replace with "not {negative_state}"
                new_item_state['question'] = replace_case_insensitive(caption, random_positive_state, f"not {random_negative_state}")
                new_item_state['negation_type'] = 'state'
                new_item_state['negations'] = {'state': random_negative_state}
                extracted_data.append(new_item_state)
                generated_any = True

        if generated_any:
            has_neg_count += 1
        # else:
            # extracted_data.append(item) # Skip original item if no negation generated
        
    print(f"Matched {match_count} items. {has_neg_count} items have at least one negation.")
    print(f"Saving merged data to {output_json}...")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
    return extracted_data

def generate_rewrite_prompt(caption, negations):
    """
    生成 LLM 润色 Prompt，支持多种否定类型
    """
    neg_desc = []
    if 'object' in negations:
        neg_desc.append(f"Object '{negations['object']}' is NOT present.")
    if 'attribute' in negations:
        neg_desc.append(f"Target does NOT have attribute '{negations['attribute']}'.")
    if 'state' in negations:
        neg_desc.append(f"Target is NOT in state '{negations['state']}'.")
        
    neg_instruction = "\n".join(neg_desc)

    prompt = f"""
Task: Rewrite the referring expression to naturally include the valid negations. Keep the core object description unchanged.
Output: A single concise sentence.

Examples:
Original: "The large vehicle is situated towards the middle-left side."
Constraints:
Object 'road' is NOT present.
Output: "The large vehicle is situated towards the middle-left side, with no road seen nearby."

Original: "The small vehicle located near the top-right corner."
Constraints:
Target does NOT have attribute 'black'.
Output: "The small, non-black vehicle located near the top-right corner."

Original: "A white airplane parked on the tarmac."
Constraints:
Target is NOT in state 'flying'.
Output: "A white airplane parked on the tarmac, not flying."

Original: "{caption}"
Constraints:
{neg_instruction}

Output:"""
    return prompt

def polish_referring_with_vllm(input_json, output_json, model_path, limit=None, dtype=None, batch_size=1000):
    if LLM is None:
        print("Error: vllm module not found. Cannot perform polishing.")
        return

    print("Reading data for polishing...")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if limit:
        data = data[:limit]
        
    # Filter items that have negations
    items_to_process = [item for item in data if item.get('negations')]
    print(f"Found {len(items_to_process)} items eligible for polishing.")
    
    if not items_to_process:
        print("No items to polish (missing 'negations' field needed from create step).")
        return

    print(f"Loading model: {model_path}")
    
    try:
        tp_size = torch.cuda.device_count()
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.8,
            dtype=dtype if dtype else 'auto'
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=150,
        stop=["\n\n", "<|im_end|>", "<|eot_id|>"]
    )
    
    total_items = len(items_to_process)
    print(f"Starting generation for {total_items} items in batches of {batch_size}...")

    import math
    num_batches = math.ceil(total_items / batch_size)

    for i in range(0, total_items, batch_size):
        batch_items = items_to_process[i : i + batch_size]
        batch_prompts = []
        for item in batch_items:
            # Use the modified question (with 'without ...') as the source for rewriting
            original = item['question']
            negations = item['negations']
            batch_prompts.append(generate_rewrite_prompt(original, negations))
        
        print(f"Processing batch {i // batch_size + 1}/{num_batches} (items {i+1} to {min(i+batch_size, total_items)})...")
        batch_outputs = llm.generate(batch_prompts, sampling_params)
        
        for item, output in zip(batch_items, batch_outputs):
            new_caption = output.outputs[0].text.strip()

            # Robust cleanup for artifacts (comments, multiple lines)
            # 1. Take first line only
            if '\n' in new_caption:
                new_caption = new_caption.split('\n')[0]
            
            # 2. Remove python-style comments often generated by models
            if '#' in new_caption:
                new_caption = new_caption.split('#')[0]
                
            new_caption = new_caption.strip()

            # 3. Basic cleanup: remove quotes and potential form-filling underscores
            new_caption = new_caption.replace("_", "")
            
            # 4. Remove surrounding quotes (iterative to handle multiple layers like '"text"')
            while (len(new_caption) >= 2 and 
                ((new_caption.startswith('"') and new_caption.endswith('"')) or 
                    (new_caption.startswith("'") and new_caption.endswith("'")))):
                new_caption = new_caption[1:-1].strip()

            item['question'] = new_caption
            
            # # Add negation_type based on negations field
            # negations = item.get('negations', {})
            # if 'object' in negations:
            #     item['negation_type'] = 'object_negative'
            # elif 'attribute' in negations:
            #     item['negation_type'] = 'attribute_negative'
            # elif 'state' in negations:
            #     item['negation_type'] = 'state_negative'
        
    print(f"Saving polished data to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Generate or Polish Referring Expressions with Negation')
    parser.add_argument('--task', choices=['generate', 'polish'], default='generate')
    # Generate args
    parser.add_argument('--input_ref', type=str, help='Input Referring Dataset JSON')
    parser.add_argument('--input_neg', type=str, help='Input Negative Entities JSON')
    # Polish args (uses output from generate as input)
    parser.add_argument('--input', type=str, help='Input JSON for polishing')
    # Common args
    parser.add_argument('--output', type=str, required=True, help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default="/mnt/nvme0/wj/Model/Llama-3.1-8B-Instruct")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for generation')

    args = parser.parse_args()

    if args.task == 'generate':
        if not args.input_ref or not args.input_neg:
            print("Error: --input_ref and --input_neg are required for generate task")
            return
        create_referring(args.input_ref, args.input_neg, args.output, args.seed)
        
    elif args.task == 'polish':
        if not args.input:
            print("Error: --input is required for polish task")
            return
        polish_referring_with_vllm(args.input, args.output, args.model, args.limit, args.dtype, args.batch_size)

if __name__ == "__main__":
    main()
