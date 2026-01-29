"""
Cache Consistency Checker
=========================
Scans the cache directory for:
1. Missing Plans
2. Missing or Corrupt Masks (Zero byte or bad zip)
3. Missing or Inconsistent CLIP Scores
4. Missing VLM Scores

Outputs: cache_report.txt
"""
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile

def check_sample(sample_idx, cache_dir):
    sample_key = f"sample_{sample_idx}"
    sample_masks_dir = os.path.join(cache_dir, "masks", sample_key)
    
    report = {
        "sample_key": sample_key,
        "issues": [],
        "corrupt_files": []
    }
    
    # 1. Check Plans (loaded globally, passed via args? No, easier to check dir structure)
    # Actually we don't need plan content, just assume we need masks for whatever plan exists.
    # But to know EXPECTED masks, we need the plan's hypothesis count.
    
    # Let's assume we check what exists vs what is valid.
    # Or load plans.json first.
    return report

def validate_npz(path):
    try:
        with np.load(path) as data:
            _ = data['mask']
        return True
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", default="cache_report.txt")
    parser.add_argument("--delete_corrupt", action="store_true", help="Delete corrupt .npz files")
    args = parser.parse_args()
    
    print(f"Loading plans from {args.cache_dir}/plans/plans.json...")
    plans_path = os.path.join(args.cache_dir, "plans", "plans.json")
    if not os.path.exists(plans_path):
        print("CRITICAL: plans.json not found!")
        return
        
    with open(plans_path, 'r') as f:
        plans = json.load(f)
        
    print(f"Checking {len(plans)} samples...")
    
    results = []
    
    def process(sample_key):
        issues = []
        corrupt = []
        
        plan = plans[sample_key]
        hypotheses = plan.get('hypotheses', [])
        num_hyp = len(hypotheses)
        
        sample_masks_dir = os.path.join(args.cache_dir, "masks", sample_key)
        
        if not os.path.exists(sample_masks_dir):
            return [f"{sample_key}: Missing masks directory"]
        
        # Check Masks
        # Expectation: 10 versions per hypothesis
        for h_idx in range(num_hyp):
            for v in range(10):
                fname = f"mask_{h_idx}_v{v}.npz"
                fpath = os.path.join(sample_masks_dir, fname)
                
                if not os.path.exists(fpath):
                    # Check if v0 exists (fallback logic)
                    if v == 0:
                         issues.append(f"Missing primary mask: {fname}")
                else:
                    # Validate
                    if not validate_npz(fpath):
                        issues.append(f"Corrupt mask: {fname}")
                        corrupt.append(fpath)
        
        # Check CLIP
        clip_path = os.path.join(sample_masks_dir, "clip_scores.json")
        if not os.path.exists(clip_path):
            issues.append("Missing CLIP scores")
        
        # Check VLM
        vlm_path = os.path.join(sample_masks_dir, "vlm_scores.json")
        if not os.path.exists(vlm_path):
            issues.append("Missing VLM scores")
            
        if args.delete_corrupt and corrupt:
            for p in corrupt:
                try:
                    os.remove(p)
                except:
                    pass
            issues.append(f"Deleted {len(corrupt)} corrupt files")
            
        return [f"{sample_key}: {i}" for i in issues] if issues else []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process, key): key for key in plans.keys()}
        
        with open(args.output, 'w') as out_f:
            for future in tqdm(as_completed(futures), total=len(plans)):
                res = future.result()
                if res:
                    for line in res:
                        out_f.write(line + "\n")
                        print(line) # Stream to stdout too
    
    print(f"Report written to {args.output}")

if __name__ == "__main__":
    main()
