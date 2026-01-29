"""
Repair Cache Script
===================
Reads cache_report.txt (or scans) and deletes corrupt files.
Also deletes partial CLIP/VLM results for affected samples to force re-computation.
"""
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="cache_report.txt")
    parser.add_argument("--cache_dir", default="cache")
    args = parser.parse_args()
    
    if not os.path.exists(args.report):
        print(f"Report {args.report} not found. Run check_consistency.py first.")
        return

    print(f"Reading {args.report}...")
    
    affected_samples = set()
    files_to_delete = []
    
    with open(args.report, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Format: sample_61: Corrupt mask: mask_41_v0.npz
            parts = line.split(": ")
            if len(parts) >= 2:
                sample_name = parts[0]
                msg = parts[1]
                
                if "Corrupt mask" in msg:
                    mask_name = parts[2]
                    full_path = os.path.join(args.cache_dir, "masks", sample_name, mask_name)
                    files_to_delete.append(full_path)
                    affected_samples.add(sample_name)
                    
                if "Missing CLIP scores" in msg:
                    affected_samples.add(sample_name)
                    
    print(f"Found {len(files_to_delete)} corrupt files in {len(affected_samples)} samples.")
    
    # Delete corrupt masks
    for p in files_to_delete:
        if os.path.exists(p):
            print(f"Deleting corrupt: {p}")
            os.remove(p)
            
    # For affected samples, we might want to delete clip_scores.json / vlm_scores.json 
    # to ensure they are re-computed with the fixed masks.
    print(f"Resetting scores for {len(affected_samples)} samples...")
    for s in affected_samples:
        s_dir = os.path.join(args.cache_dir, "masks", s)
        
        # Delete CLIP scores
        clip_p = os.path.join(s_dir, "clip_scores.json")
        if os.path.exists(clip_p):
            print(f"  Resetting CLIP: {s}")
            os.remove(clip_p)
            
        # Delete VLM scores
        vlm_p = os.path.join(s_dir, "vlm_scores.json")
        if os.path.exists(vlm_p):
            print(f"  Resetting VLM: {s}")
            os.remove(vlm_p)

    print("\nRepair complete.")
    print("Now run the following to refill gaps:")
    print(f"  python scripts/precompute_all.py --phase masks --cache_dir {args.cache_dir}")
    print(f"  python scripts/precompute_all.py --phase clip --cache_dir {args.cache_dir}")
    print(f"  ./scripts/phase4_vlm.sh")

if __name__ == "__main__":
    main()
