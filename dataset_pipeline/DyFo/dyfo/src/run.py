import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import asyncio
import random
import base64
from PIL import Image
import io

from utils import get_options, get_chunk

from policies import policy_map  # Use full import path for policy_map

def save_encoded_image(encoded_image_str, save_path):
    """
    Save an encoded image string to file.
    
    Args:
        encoded_image_str: Base64 encoded image string or other encoded format
        save_path: Path to save the image
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        if encoded_image_str.startswith('data:'):
            encoded_image_str = encoded_image_str.split(',', 1)[1]
        
        # Decode base64 string
        image_data = base64.b64decode(encoded_image_str)
        
        # Create PIL image from binary data
        image = Image.open(io.BytesIO(image_data))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save image
        image.save(save_path)
        
    except Exception as e:
        print(f"Error saving encoded image: {e}")
        # Fallback: save as text file for debugging
        with open(save_path + ".txt", "w") as f:
            f.write(str(encoded_image_str))

def png_to_base64(image_path):
    """
    Convert a PNG file to base64 string.
    
    Args:
        image_path: Path to the PNG image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    try:
        # Open and read the image file
        with Image.open(image_path) as image:
            # Convert to RGB if necessary (PNG might have alpha channel)
            if image.mode in ('RGBA', 'LA'):
                # Create a white background
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                else:  # LA mode
                    background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Save to BytesIO buffer
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=95)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
    except Exception as e:
        print(f"Error converting PNG to base64: {e}")
        return None
    
async def create_sample(args):
    row, method_args, round_idx = args
    QuestionSample = policy_map[method_args.method_name]
    return QuestionSample(row, method_args, round_idx)

async def process_sample(sample):
    return await sample.process()

async def eval_model(args):
    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # Prepare sample arguments
    sample_args = []
    rows_as_dicts = questions.to_dict(orient="records")
    for row in rows_as_dicts:
        if args.all_rounds:
            # do not use this 
            num_rounds = len(get_options(row, ['A', 'B', 'C', 'D']))
        else:
            num_rounds = 1
            
        for round_idx in range(num_rounds):
            sample_args.append((row, args, round_idx))
    
    # Generate samples using coroutines
    samples = []
    if args.debug: 
        # sample_args = [random.choice(sample_args[115:])]
        test_row = {
            "index": "remote_sense",
            "image": png_to_base64("/mnt/wj/gen_neg/NWPU-Caption/NWPU_images/sparse_residential/sparse_residential_667.jpg"),
            "question": "In the image, is there a green woods seen?",
            "A": "yes",
            "B": "no",
            "C": "none",
            "D": "none",
            "answer": "B",
            "hint": "none"
        }
        sample_args = [(test_row, args, 0)]
    else:
        random.seed(42)  # Fix random seed
        # sample_args = [random.choice(sample_args) for _ in range(100)]
        
    # Clear output file
    with open(answers_file, "w") as f:
        pass

    batch_size = 100
    total_samples = len(sample_args)
    
    # Limit concurrency to avoid overloading the server
    # Adjust this number based on your GPU memory and server capacity
    concurrency_limit = 4
    sem = asyncio.Semaphore(concurrency_limit)

    async def process_sample_with_sem(sample):
        async with sem:
            return await process_sample(sample)
    
    for i in range(0, total_samples, batch_size):
        batch_args = sample_args[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_samples + batch_size - 1)//batch_size}")

        # Generate samples for batch
        batch_samples = []
        tasks = [create_sample(args) for args in batch_args]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Creating samples"):
            sample = await coro
            batch_samples.append(sample)
        
        # Process samples for batch
        batch_results = []
        tasks = [process_sample_with_sem(sample) for sample in batch_samples]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing samples"):
            process_result = await coro
            result, final_image, original_image = process_result
                
            batch_results.append(result)
            if args.debug:
                save_encoded_image(final_image, f"debug_images/{result['question_id']}.png")
                save_encoded_image(original_image, f"debug_images/{result['question_id']}_original.png")
                # 同时显示confirmed objects信息
                if 'confirmed_objects' in result and result['confirmed_objects']:
                    print(f"Question {result['question_id']} confirmed objects: {result['confirmed_objects']}")
        
        # Append batch results to file
        with open(answers_file, "a") as f:
            for result in batch_results:
                f.write(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--method_name", type=str, default="common")
    parser.add_argument("--image-size", type=int, default=336)
    
    def str2bool(v):
        return v.lower() == 'true'
    
    parser.add_argument("--debug", type=str2bool, help="debug mode", default=False)
    args = parser.parse_args()
    
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugpy connection...")
        debugpy.wait_for_client()
        print("Breakpoint stopped here, ready for debugging...")
        debugpy.breakpoint()
    
    import time
    start_time = time.time()
    asyncio.run(eval_model(args))
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")