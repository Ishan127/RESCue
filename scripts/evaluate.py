import argparse
import sys
import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rescue_pipeline import RESCuePipeline
from src.utils import calculate_iou

def evaluate(fraction=0.1, N=4):
    print(f"Loading ReasonSeg dataset (Validation)...")
    try:
        ds = load_dataset("Ricky06662/ReasonSeg_val", split="validation")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Subsample
    total_samples = len(ds)
    num_samples = int(total_samples * fraction)
    print(f"Total samples: {total_samples}. Evaluating on {fraction*100}% = {num_samples} samples.")
    
    # Shuffle and select
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    # Initialize Pipeline
    pipeline = RESCuePipeline()
    
    ious = []
    
    print("Starting evaluation...")
    for i, sample in tqdm(enumerate(ds), total=num_samples):
        # Inspect columns if first sample
        if i == 0:
            print("Dataset columns:", sample.keys())
            
        # Extract fields
        # Adapting to likely column names
        image = sample.get('image')
        query = sample.get('text') or sample.get('query') or sample.get('sentence')
        gt_mask = sample.get('mask') or sample.get('label')
        
        if image is None or query is None:
            print(f"Skipping sample {i}: Missing image or query.")
            continue
            
        # Convert GT mask to numpy if needed
        if gt_mask:
            gt_mask = np.array(gt_mask) > 0
        
        # Run Pipeline
        # We need to save image to temp path for the pipeline currently (as it takes path)
        # OR update pipeline to accept PIL image.
        # Pipeline `run` takes `image_path`.
        # Let's verify `rescue_pipeline.py`.
        # It calls `load_image(image_path)`.
        # I should probably update `rescue_pipeline.py` to accept Image object OR path.
        # For now, I'll save to temp.
        
        temp_img_path = f"temp_eval_{i}.jpg"
        image.save(temp_img_path)
        
        try:
            result = pipeline.run(temp_img_path, query, N=N)
            
            if result and gt_mask is not None:
                # Calculate IoU
                pred_mask = result['best_mask']
                # Resize gt_mask if needed to match pred_mask?
                # Usually they should match if image size is same.
                
                # Ensure binary
                pred_mask_bin = pred_mask > 0
                
                iou = calculate_iou(pred_mask_bin, gt_mask)
                ious.append(iou)
                print(f"Sample {i} | IoU: {iou:.4f}")
            else:
                print(f"Sample {i} | No result or No GT.")
                ious.append(0.0)
                
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    if ious:
        print(f"\nMean gIoU (approx as IoU): {np.mean(ious):.4f}")
    else:
        print("No evaluations performed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of dataset to use (0.0 to 1.0)")
    parser.add_argument("--N", type=int, default=4, help="Number of reasoning paths")
    args = parser.parse_args()
    
    evaluate(fraction=args.fraction, N=args.N)
