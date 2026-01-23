import argparse
import sys
import os
import random
import numpy as np
import json
import logging
from datasets import load_dataset
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rescue_pipeline import RESCuePipeline
from src.utils import apply_red_alpha_overlay, calculate_iou

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def debug_sample(sample_index=None, pipeline=None):
    print("\n" + "="*80)
    print(" DEBUG SINGLE SAMPLE EXECUTION ")
    print("="*80 + "\n")

    # 1. Load Dataset
    print("[1] Loading Dataset...")
    ds = load_dataset("Ricky06662/ReasonSeg_val", split="test")
    
    if sample_index is None:
        sample_index = random.randint(0, len(ds) - 1)
        
    sample = ds[sample_index]
    print(f"    Selected Sample Index: {sample_index}")
    
    image = sample['image']
    query = sample.get('text') or sample.get('query')
    gt_mask_pil = sample.get('mask') or sample.get('label')
    gt_mask = np.array(gt_mask_pil) > 0 if gt_mask_pil else None

    print(f"    Query: '{query}'")
    print(f"    Image Size: {image.size}")
    if gt_mask is not None:
        print(f"    GT Mask Shape: {gt_mask.shape}")
    else:
        print("    No Ground Truth Mask provided.")

    # Save temp image for API
    temp_img_path = f"debug_sample_{sample_index}.jpg"
    image.save(temp_img_path)
    
    try:
        # 2. Planning Phase
        print("\n" + "-"*50)
        print("[2] Running PLANNER (LLM)")
        print("-"*50)
        
        # We manually call pipeline.planner.generate_hypotheses to inspect raw output
        # pipeline.run does this internally but we want to see the objects
        hypotheses = pipeline.planner.generate_hypotheses(temp_img_path, query, N=4)
        
        print(f"\n    Generated {len(hypotheses)} Hypotheses:")
        for idx, hyp in enumerate(hypotheses):
            print(f"\n    [Hypothesis {idx}]")
            print(f"      Strategy:    {hyp.get('strategy', 'N/A')}")
            print(f"      Noun Phrase: '{hyp.get('noun_phrase')}'")
            print(f"      Reasoning:   '{hyp.get('reasoning')}'")
            print(f"      Box:         {hyp.get('box')}")
            print(f"      Confidence:  {hyp.get('confidence')}")
            print(f"      RAW LLM RESPONSE (Snippet): {hyp.get('raw_text', '')[:200]}...") # Truncate if too long

        # 3. Execution Phase
        print("\n" + "-"*50)
        print("[3] Running EXECUTOR (SAM)")
        print("-"*50)
        
        image_np = np.array(image)
        candidates = []

        for i, hyp in enumerate(hypotheses):
            box = hyp['box']
            noun_phrase = hyp['noun_phrase']
            
            print(f"\n    Extracting masks for H{i}: '{noun_phrase}' with Box {box}")
            masks = pipeline.executor.execute(image_np, box, noun_phrase)
            
            print(f"      SAM returned {len(masks)} masks.")
            
            for j, mask in enumerate(masks):
                print(f"        Mask M{j} Shape: {mask.shape}, Unique Values: {np.unique(mask)}")
                
                # Check match with raw image size
                if mask.shape[:2] != image_np.shape[:2]:
                     print(f"        [WARNING] Shape Mismatch! Image: {image_np.shape[:2]} vs Mask: {mask.shape[:2]}")

                # 4. Verification
                print(f"        Running Verifier...")
                score = pipeline.verifier.verify(image, mask, query)
                print(f"        -> Score: {score}")

                cand_id = f"H{i}_M{j}"
                
                # Calculate IoU if GT exists
                iou_val = 0.0
                if gt_mask is not None:
                     iou_val = calculate_iou(mask, gt_mask)
                     print(f"        -> IoU vs GT: {iou_val:.4f}")

                candidates.append({
                    'id': cand_id,
                    'mask': mask,
                    'score': score,
                    'iou': iou_val
                })
                
                # Save debugging overlay
                overlay_dir = "debug_raw_output"
                os.makedirs(overlay_dir, exist_ok=True)
                ov_path = os.path.join(overlay_dir, f"sample_{sample_index}_{cand_id}_score{score}.jpg")
                try:
                    ov_img = apply_red_alpha_overlay(image, mask, alpha=0.5)
                    ov_img.save(ov_path)
                except Exception as e:
                    print(f"Failed to save overlay: {e}")

        # 5. Final Result
        print("\n" + "-"*50)
        print("[5] FINAL SELECTION")
        print("-"*50)
        
        if candidates:
            best_cand = max(candidates, key=lambda x: x['score'])
            print(f"    Best Candidate: {best_cand['id']}")
            print(f"    Verifier Score: {best_cand['score']}")
            print(f"    IoU:            {best_cand['iou']:.4f}")
        else:
            print("    No candidates generated.")

    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=None, help="Index of sample to debug (random if not set)")
    args = parser.parse_args()
    
    # Initialize Pipeline
    pipeline = RESCuePipeline(
        planner_api_base="http://localhost:8000/v1",
        executor_api_base="http://localhost:8001"
    )
    
    debug_sample(args.index, pipeline)
