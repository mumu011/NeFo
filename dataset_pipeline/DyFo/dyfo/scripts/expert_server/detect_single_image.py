import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch

# Add lang-segment-anything to path
current_dir = os.path.dirname(os.path.abspath(__file__))
lang_sam_path = os.path.join(current_dir, "../../../lang-segment-anything")
sys.path.insert(0, lang_sam_path)

from lang_sam import LangSAM
from lang_sam.utils import draw_image

def main():
    parser = argparse.ArgumentParser(description="Detect objects in a single image using LangSAM")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for detection")
    parser.add_argument("--output_path", type=str, default="output.jpg", help="Path to save the output image")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    print("Loading model...")
    model = LangSAM(
        gdino_model_ckpt_path="/mnt/wj/Model/grounding-dino-base", 
        gdino_processor_ckpt_path="/mnt/wj/Model/grounding-dino-base"
    )
    print("Model loaded.")

    print(f"Loading image from {args.image_path}...")
    image_pil = Image.open(args.image_path).convert("RGB")

    print(f"Predicting with prompt: '{args.text_prompt}'...")
    results = model.predict([image_pil], [args.text_prompt], box_threshold=args.box_threshold, text_threshold=args.text_threshold)
    
    if not results:
        print("No detections found.")
        return

    # results is a list of dicts, we only processed one image
    result = results[0]
    
    boxes = result["boxes"]
    scores = result["scores"]
    masks = result["masks"]
    # labels usually come from the prompt, but let's see what predict returns.
    # In lang_sam.py, predict returns "boxes", "scores", "masks", "mask_scores".
    # It doesn't seem to return labels explicitly in the result dict in the code I read earlier, 
    # but let's check model_service.py again or lang_sam.py.
    
    # Checking lang_sam.py content from previous turn:
    # result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
    # processed_result = { **result, "masks": [], "mask_scores": [] }
    # ...
    # all_results.append(processed_result)
    
    # GDINO predict returns labels?
    # Let's assume we construct labels from the prompt and scores for visualization if not present.
    # Or we can check what keys are in result.
    
    # Wait, in model_service.py:
    # "labels": result["labels"] if len(result["labels"]) else [],
    
    # So GDINO result has "labels".
    
    labels = result.get("labels", [])
    if len(labels) == 0:
        # If no labels returned (maybe empty detection), but boxes exist?
        # If boxes exist, labels should exist.
        # If labels are just class names, we might want to append scores.
        labels = [f"{args.text_prompt} {score:.2f}" for score in scores]
    else:
        # Append score to label if not already there
        labels = [f"{label} {score:.2f}" for label, score in zip(labels, scores)]

    print(f"Found {len(boxes)} detections.")

    if len(boxes) > 0:
        # Convert image to numpy array for supervision
        image_np = np.array(image_pil)
        
        # draw_image expects: image_rgb, masks, xyxy, probs, labels
        annotated_image = draw_image(
            image_np, 
            masks, 
            boxes, 
            scores, 
            labels
        )
        
        # Save image
        Image.fromarray(annotated_image).save(args.output_path)
        print(f"Saved annotated image to {args.output_path}")
    else:
        print("No objects detected to draw.")

if __name__ == "__main__":
    main()
