import json
import os
import sys
import torch
from PIL import Image
from tqdm import tqdm
import argparse

# Add lang-segment-anything to path
current_dir = os.path.dirname(os.path.abspath(__file__))
lang_sam_path = os.path.join(current_dir, "../../../lang-segment-anything")
sys.path.insert(0, lang_sam_path)

from lang_sam import LangSAM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="/mnt/wj/gen_neg/NWPU-Caption/validation_questions_polished.json")
    parser.add_argument("--image_dir", type=str, default="/mnt/wj/gen_neg/NWPU-Caption/NWPU_images_all")
    parser.add_argument("--output_path", type=str, default="/mnt/wj/gen_neg/NWPU-Caption/validation_questions_polished_object_results.json")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box threshold for detection")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for detection")
    args = parser.parse_args()

    print("Loading model...")
    model = LangSAM(
        gdino_model_ckpt_path="/mnt/wj/Model/grounding-dino-base", 
        gdino_processor_ckpt_path="/mnt/wj/Model/grounding-dino-base"
    )
    print("Model loaded.")

    print(f"Loading data from {args.json_path}...")
    with open(args.json_path, 'r') as f:
        data = json.load(f)
    
    results = []
    
    # Iterate through data
    for item in tqdm(data):
        filename = item['filename']
        image_path = os.path.join(args.image_dir, filename)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Process validation questions
        if 'validation_questions' in item:
            for q in item['validation_questions']:
                if q.get('type') == 'object_negative_validation':
                    target_object = q.get('target')
                    if not target_object:
                        continue
                        
                    # Predict
                    try:
                        # predict returns list of dicts, we pass one image
                        prediction = model.predict([image], [target_object], box_threshold=args.box_threshold, text_threshold=args.text_threshold)
                        
                        # Check if boxes exist
                        has_boxes = False
                        if prediction and len(prediction) > 0:
                            boxes = prediction[0].get('boxes', [])
                            if len(boxes) > 0:
                                has_boxes = True
                        
                        # Save result in the question object (or separate structure)
                        q['langsam_detection'] = has_boxes
                        
                    except Exception as e:
                        print(f"Error predicting for {filename}, target {target_object}: {e}")
                        q['langsam_detection_error'] = str(e)

    print(f"Saving results to {args.output_path}...")
    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
