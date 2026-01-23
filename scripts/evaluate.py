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

def evaluate(fraction=0.1, N=4, dtype="auto", quantization=None, planner_url="http://localhost:8000/v1", executor_url="http://localhost:8001"):
    print(f"Loading ReasonSeg dataset (Validation)...")
    try:
        ds = load_dataset("Ricky06662/ReasonSeg_val", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_samples = len(ds)
    num_samples = int(total_samples * fraction)
    print(f"Total samples: {total_samples}. Evaluating on {fraction*100}% = {num_samples} samples.")
    
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    pipeline = RESCuePipeline(
        dtype=dtype, 
        quantization=quantization,
        planner_api_base=planner_url, 
        executor_api_base=executor_url
    )
    
    ious = []
    
    print("Starting evaluation...")
    for i, sample in tqdm(enumerate(ds), total=num_samples):
        if i == 0:
            print("Dataset columns:", sample.keys())
            
        image = sample.get('image')
        query = sample.get('text') or sample.get('query') or sample.get('sentence')
        gt_mask = sample.get('mask') or sample.get('label')
        
        if image is None or query is None:
            print(f"Skipping sample {i}: Missing image or query.")
            continue
            
        if gt_mask:
            gt_mask = np.array(gt_mask) > 0
        
        temp_img_path = f"temp_eval_{i}.jpg"
        image.save(temp_img_path)
        
        try:
            # Pass gt_mask to pipeline.run is not standard but we can calculate IoU per candidate here if we refactor run or do it after.
            # Easiest way is to modify RESCuePipeline.run to return all candidates and we compute IoU here for debugging.
            result = pipeline.run(temp_img_path, query, N=N)
            
            if result and gt_mask is not None:
                # --- Debug: Print IoU for each candidate ---
                print("--- Candidate Analysis ---")
                best_cand_iou = 0.0
                for cand in result.get('all_candidates', []):
                    c_mask = cand['mask'] > 0
                    c_score = cand['score']
                    c_iou = calculate_iou(c_mask, gt_mask)
                    print(f"  > {cand['id']}: Score={c_score:.1f}, IoU={c_iou:.4f} | {cand['noun_phrase']}")
                    if c_score == result['best_score'] and cand.get('best_flag', False): # Or just match by ID if 'best_mask' logic is consistent
                        pass

                pred_mask = result['best_mask']
                pred_mask_bin = pred_mask > 0
                
                iou = calculate_iou(pred_mask_bin, gt_mask)
                ious.append(iou)
                print(f"Sample {i} | Final Selected IoU: {iou:.4f}")
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
    parser.add_argument("--dtype", default="auto", help="Model data type (auto, float16, etc.)")
    parser.add_argument("--quantization", default=None, help="Model quantization (awq, gptq, int8, etc.)")
    parser.add_argument("--planner_url", default="http://localhost:8000/v1", help="API URL for Planner LLM")
    parser.add_argument("--executor_url", default="http://localhost:8001", help="API URL for Executor SAM")
    args = parser.parse_args()
    
    evaluate(
        fraction=args.fraction, 
        N=args.N, 
        dtype=args.dtype, 
        quantization=args.quantization,
        planner_url=args.planner_url,
        executor_url=args.executor_url
    )
