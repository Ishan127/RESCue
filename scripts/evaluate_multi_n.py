"""
Optimized Multi-N Evaluation Script

Generates N=64 candidates ONCE, then evaluates verification performance
for N=1,2,4,8,16,32,64 using subsets of the generated candidates.

This is much faster than running the full pipeline 7 times.
"""
import argparse
import sys
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rescue_pipeline_fast import RESCuePipelineFast
from src.utils import calculate_iou

N_VALUES = [1, 2, 4, 8, 16, 32, 64]

def evaluate_multi_n(fraction=0.1, max_n=64, planner_url="http://localhost:8000/v1", 
                     executor_url="http://localhost:8001", verification_mode="comparative"):
    print(f"Loading ReasonSeg dataset...")
    try:
        ds = load_dataset("Ricky06662/ReasonSeg_val", split="test")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_samples = len(ds)
    num_samples = max(1, int(total_samples * fraction))
    print(f"Samples: {num_samples}/{total_samples} ({fraction*100:.0f}%)")
    
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    pipeline = RESCuePipelineFast(
        planner_api_base=planner_url, 
        executor_api_base=executor_url,
        verbose=False
    )
    
    results_by_n = {n: {'ious': [], 'oracle_ious': [], 'times': []} for n in N_VALUES if n <= max_n}
    
    print(f"\n{'='*70}")
    print(f"Evaluating N={list(results_by_n.keys())} | Mode: {verification_mode}")
    print(f"{'='*70}\n")
    
    for sample_idx, sample in tqdm(enumerate(ds), total=num_samples, desc="Samples"):
        image = sample.get('image')
        query = sample.get('text') or sample.get('query') or sample.get('sentence')
        gt_mask = sample.get('mask') or sample.get('label')
        
        if image is None or query is None:
            continue
            
        if gt_mask is not None:
            gt_mask = np.array(gt_mask) > 0
        
        temp_img_path = f"temp_eval_{sample_idx}.jpg"
        image.save(temp_img_path)
        
        try:
            t0 = time.time()
            all_candidates = pipeline.generate_all_candidates(temp_img_path, query, N=max_n, gt_mask=gt_mask)
            gen_time = time.time() - t0
            
            if not all_candidates:
                continue
            
            for n in results_by_n.keys():
                candidates_subset = all_candidates[:n]
                
                t0 = time.time()
                best_idx = pipeline.verify_and_select(
                    image, candidates_subset, query, 
                    mode=verification_mode
                )
                verify_time = time.time() - t0
                
                if best_idx is not None and gt_mask is not None:
                    pred_mask = candidates_subset[best_idx]['mask']
                    iou = calculate_iou(pred_mask, gt_mask)
                    oracle_iou = max(c.get('iou', 0) for c in candidates_subset)
                    
                    results_by_n[n]['ious'].append(iou)
                    results_by_n[n]['oracle_ious'].append(oracle_iou)
                    results_by_n[n]['times'].append(gen_time / max_n * n + verify_time)
                    
        except Exception as e:
            tqdm.write(f"Sample {sample_idx} error: {e}")
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
    
    print_results(results_by_n, verification_mode)
    
    output_file = f"results_{verification_mode}_{int(fraction*100)}pct.json"
    save_results(results_by_n, output_file)

def print_results(results_by_n, mode):
    print(f"\n{'='*70}")
    print(f"RESULTS ({mode.upper()})")
    print(f"{'='*70}")
    
    print(f"\n{'N':>4} | {'gIoU':>8} | {'Oracle':>8} | {'%Oracle':>8} | {'Time(s)':>8} | {'Samples':>7}")
    print("-" * 60)
    
    for n in sorted(results_by_n.keys()):
        data = results_by_n[n]
        if not data['ious']:
            continue
        
        mean_iou = np.mean(data['ious'])
        mean_oracle = np.mean(data['oracle_ious'])
        pct_oracle = mean_iou / mean_oracle * 100 if mean_oracle > 0 else 0
        mean_time = np.mean(data['times'])
        
        print(f"{n:>4} | {mean_iou:>8.4f} | {mean_oracle:>8.4f} | {pct_oracle:>7.1f}% | {mean_time:>8.2f} | {len(data['ious']):>7}")
    
    print(f"\nScaling Analysis:")
    ns = sorted(results_by_n.keys())
    ious = [np.mean(results_by_n[n]['ious']) for n in ns if results_by_n[n]['ious']]
    
    if len(ious) >= 2:
        improvement_1_to_max = (ious[-1] - ious[0]) / ious[0] * 100 if ious[0] > 0 else 0
        print(f"  N=1 → N={ns[-1]}: {improvement_1_to_max:+.1f}% gIoU improvement")
        
        for i in range(1, len(ns)):
            if results_by_n[ns[i]]['ious'] and results_by_n[ns[i-1]]['ious']:
                delta = np.mean(results_by_n[ns[i]]['ious']) - np.mean(results_by_n[ns[i-1]]['ious'])
                print(f"  N={ns[i-1]} → N={ns[i]}: {delta:+.4f} gIoU")

def save_results(results_by_n, output_file):
    output = {}
    for n, data in results_by_n.items():
        output[str(n)] = {
            'mean_iou': float(np.mean(data['ious'])) if data['ious'] else 0,
            'mean_oracle': float(np.mean(data['oracle_ious'])) if data['oracle_ious'] else 0,
            'mean_time': float(np.mean(data['times'])) if data['times'] else 0,
            'num_samples': len(data['ious'])
        }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-N Evaluation for RESCue Pipeline")
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--max_n", type=int, default=64)
    parser.add_argument("--planner_url", default="http://localhost:8002/v1", help="Planner API (7B model)")
    parser.add_argument("--verifier_url", default="http://localhost:8000/v1", help="Verifier API (32B model)")
    parser.add_argument("--executor_url", default="http://localhost:8001", help="SAM3 executor API")
    parser.add_argument("--mode", choices=["comparative", "individual", "heuristic"], default="comparative")
    args = parser.parse_args()
    
    # Update environment variables for dual-model setup
    os.environ["PLANNER_API_BASE"] = args.planner_url
    os.environ["VERIFIER_API_BASE"] = args.verifier_url
    
    evaluate_multi_n(
        fraction=args.fraction,
        max_n=args.max_n,
        planner_url=args.planner_url,
        executor_url=args.executor_url,
        verification_mode=args.mode
    )
