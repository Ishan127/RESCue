import argparse
import sys
import os
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rescue_pipeline import RESCuePipeline
from src.utils import calculate_iou, apply_red_alpha_overlay

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
    
    ious_comparative = []
    ious_individual = []
    ious_oracle = []
    
    print("\n" + "="*60)
    print("Starting evaluation with BOTH verification methods...")
    print("="*60 + "\n")
    
    for i, sample in tqdm(enumerate(ds), total=num_samples):
        if i == 0:
            print("Dataset columns:", sample.keys())
            
        image = sample.get('image')
        query = sample.get('text') or sample.get('query') or sample.get('sentence')
        gt_mask = sample.get('mask') or sample.get('label')
        
        if image is None or query is None:
            print(f"Skipping sample {i}: Missing image or query.")
            continue
            
        if gt_mask is not None:
            gt_mask = np.array(gt_mask) > 0
        
        temp_img_path = f"temp_eval_{i}.jpg"
        image.save(temp_img_path)
        
        try:
            result = pipeline.run(temp_img_path, query, N=N, gt_mask=gt_mask, verification_mode="both")
            
            if result and gt_mask is not None:
                iou_comp = calculate_iou(result['best_mask_comparative'], gt_mask)
                iou_indiv = calculate_iou(result['best_mask_individual'], gt_mask)
                iou_oracle = result['oracle_best'].get('iou', 0) if result.get('oracle_best') else 0
                
                ious_comparative.append(iou_comp)
                ious_individual.append(iou_indiv)
                ious_oracle.append(iou_oracle)
                
                comp_win = "✓" if iou_comp >= iou_indiv else " "
                indiv_win = "✓" if iou_indiv >= iou_comp else " "
                
                print(f"\nSample {i} Summary:")
                print(f"  Comparative [{comp_win}]: IoU {iou_comp:.4f} ({result['best_candidate_comparative']['id']})")
                print(f"  Individual  [{indiv_win}]: IoU {iou_indiv:.4f} ({result['best_candidate_individual']['id']})")
                print(f"  Oracle (GT):     IoU {iou_oracle:.4f} ({result['oracle_best']['id']})")
                
                if i < 5:
                    debug_dir = "debug_output"
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    img_pil = sample.get('image')
                    try:
                        overlay_comp = apply_red_alpha_overlay(img_pil, result['best_mask_comparative'], alpha=0.5)
                        overlay_comp.save(os.path.join(debug_dir, f"sample_{i}_comparative_iou{iou_comp:.3f}.png"))
                        
                        overlay_indiv = apply_red_alpha_overlay(img_pil, result['best_mask_individual'], alpha=0.5)
                        overlay_indiv.save(os.path.join(debug_dir, f"sample_{i}_individual_iou{iou_indiv:.3f}.png"))
                        
                        overlay_gt = apply_red_alpha_overlay(img_pil, gt_mask, alpha=0.5)
                        overlay_gt.save(os.path.join(debug_dir, f"sample_{i}_gt.png"))
                    except Exception as e:
                        print(f"    - Failed to save debug images: {e}")
            else:
                print(f"Sample {i} | No result or No GT.")
                ious_comparative.append(0.0)
                ious_individual.append(0.0)
                ious_oracle.append(0.0)
                
        except Exception as e:
            print(f"Sample {i} | Error: {e}")
            ious_comparative.append(0.0)
            ious_individual.append(0.0)
            ious_oracle.append(0.0)
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if ious_comparative:
        mean_comp = np.mean(ious_comparative)
        mean_indiv = np.mean(ious_individual)
        mean_oracle = np.mean(ious_oracle)
        
        print(f"\n{'Method':<25} | {'Mean gIoU':>10} | {'vs Oracle':>10}")
        print("-"*50)
        print(f"{'Comparative Verification':<25} | {mean_comp:>10.4f} | {mean_comp/mean_oracle*100:>9.1f}%")
        print(f"{'Individual Verification':<25} | {mean_indiv:>10.4f} | {mean_indiv/mean_oracle*100:>9.1f}%")
        print(f"{'Oracle (Best Possible)':<25} | {mean_oracle:>10.4f} | {'100.0':>9}%")
        
        comp_wins = sum(1 for c, i in zip(ious_comparative, ious_individual) if c > i)
        indiv_wins = sum(1 for c, i in zip(ious_comparative, ious_individual) if i > c)
        ties = sum(1 for c, i in zip(ious_comparative, ious_individual) if c == i)
        
        print(f"\nHead-to-head (n={len(ious_comparative)}):")
        print(f"  Comparative wins: {comp_wins} ({comp_wins/len(ious_comparative)*100:.1f}%)")
        print(f"  Individual wins:  {indiv_wins} ({indiv_wins/len(ious_comparative)*100:.1f}%)")
        print(f"  Ties:             {ties} ({ties/len(ious_comparative)*100:.1f}%)")
        
        winner = "COMPARATIVE" if mean_comp > mean_indiv else "INDIVIDUAL" if mean_indiv > mean_comp else "TIE"
        print(f"\n>>> WINNER: {winner} <<<")
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
