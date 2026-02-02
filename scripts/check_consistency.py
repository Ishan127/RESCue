"""
Cache Consistency Checker
=========================
Scans the cache directory for:
1. Missing Plans
2. Missing or Corrupt Masks (Zero byte or bad zip)
3. Missing or Inconsistent CLIP Scores
4. Missing or Inconsistent VLM Scores

Outputs: cache_report.txt
"""
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile

def validate_npz(path):
    try:
        with np.load(path) as data:
            _ = data['mask']
        return True
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        return False

def validate_json_file(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data is not None
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output", default="cache_report.txt")
    parser.add_argument("--delete_corrupt", action="store_true", help="Delete corrupt .npz files")
    args = parser.parse_args()
    
    print(f"Loading plans from {args.cache_dir}/plans/plans.json...")
    plans_path = os.path.join(args.cache_dir, "plans", "plans.json")
    if not os.path.exists(plans_path):
        print("CRITICAL: plans.json not found!")
        return
        
    try:
        with open(plans_path, 'r') as f:
            plans = json.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load plans.json: {e}")
        return
        
    print(f"Checking {len(plans)} samples...")
    
    def process(sample_key):
        issues = []
        corrupt = []
        
        plan = plans[sample_key]
        hypotheses = plan.get('hypotheses', [])
        
        # 1. Check Plans
        if not hypotheses:
            return [f"{sample_key}: No hypotheses in plan"]
            
        num_hyp = len(hypotheses)
        sample_masks_dir = os.path.join(args.cache_dir, "masks", sample_key)
        
        if not os.path.exists(sample_masks_dir):
            return [f"{sample_key}: Missing masks directory"]
        
        # 2. Check Masks (10 versions)
        masks_exist = 0
        for h_idx in range(num_hyp):
            for v in range(10):
                fname = f"mask_{h_idx}_v{v}.npz"
                fpath = os.path.join(sample_masks_dir, fname)
                
                if not os.path.exists(fpath):
                    if v == 0:
                        issues.append(f"Missing mask {h_idx}_v{v}")
                else:
                    masks_exist += 1
                    # Only random sampling check for speed? Or check all?
                    # Checking all is safer for "consistency check"
                    if not validate_npz(fpath):
                        issues.append(f"Corrupt mask: {fname}")
                        corrupt.append(fpath)
        
        if masks_exist == 0:
            issues.append("No masks found at all")

        # 3. Check CLIP
        clip_path = os.path.join(sample_masks_dir, "clip_scores.json")
        if not os.path.exists(clip_path):
            issues.append("Missing CLIP scores")
        elif not validate_json_file(clip_path):
            issues.append("Corrupt CLIP scores JSON")
        
        # 4. Check VLM
        vlm_path = os.path.join(sample_masks_dir, "vlm_scores.json")
        if not os.path.exists(vlm_path):
            issues.append("Missing VLM scores")
        else:
            try:
                with open(vlm_path, 'r') as f:
                    vlm_data = json.load(f)
                
                # Check coverage
                if not vlm_data:
                    issues.append("Empty VLM scores")
                else:
                    # Optional: Check if all hypotheses have scores
                    # (soft check, maybe just warn if coverage is low)
                    scored_hypotheses = len(vlm_data)
                    if scored_hypotheses < num_hyp * 0.5: # Arbitrary threshold
                        issues.append(f"Low VLM coverage: {scored_hypotheses}/{num_hyp} hypotheses scored")
            except:
                issues.append("Corrupt VLM scores JSON")
            
        # Handle corruption deletion
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
        
        total_samples = len(plans)
        samples_with_issues = 0
        
        with open(args.output, 'w') as out_f:
            for future in tqdm(as_completed(futures), total=total_samples, desc="Checking Cache"):
                res = future.result()
                if res:
                    samples_with_issues += 1
                    for line in res:
                        out_f.write(line + "\n")
                        # Don't print everything to stdout if it's too much, just critical errors
                        if "Corrupt" in line or "Missing masks directory" in line:
                            print(line)

    print("-" * 60)
    print(f"Consistency Check Complete")
    print(f"Total Samples: {total_samples}")
    print(f"Samples with Issues: {samples_with_issues}")
    print(f"Perfect Samples: {total_samples - samples_with_issues}")
    print(f"Detailed report: {args.output}")
    print("-" * 60)

if __name__ == "__main__":
    main()
