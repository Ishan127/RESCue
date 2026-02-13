
import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def calculate_iou(mask1, mask2):
    mask1 = np.asarray(mask1) > 0
    mask2 = np.asarray(mask2) > 0
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0: return 0.0
    return intersection / union

def apply_red_alpha_overlay(image, mask, alpha=0.5):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    mask = mask > 0
    if image.size != mask.shape[::-1]:
        mask = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST) > 0
        
    overlay = np.array(image)
    overlay[mask, 0] = (1 - alpha) * overlay[mask, 0] + alpha * 255
    return Image.fromarray(overlay)

from datasets import load_dataset

# Use precise same shuffle logic as evaluate_pipeline.py
# Default fraction=0.1 means only top 10% are in cache.
# But "sample_0" in cache corresponds to index 0 of the shuffled dataset.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="paper_samples")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--num_samples", type=int, default=25, help="Number of samples to extract")
    parser.add_argument("--min_iou", type=float, default=0.95, help="Minimum IoU for the 'Best' candidate")
    # parser.add_argument("--split", type=str, default="test") # Hardcoded to test for ReasonSeg
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ReasonSeg dataset (HF)...")
    try:
        ds = load_dataset("Ricky06662/ReasonSeg_test", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Shuffle to match pipeline order
    ds = ds.shuffle(seed=42)
    
    print(f"Dataset loaded: {len(ds)} samples (checking cache for matches)")

    extracted_count = 0
    
    # Iterate through dataset
    # We check if sample_idx exists in cache.
    # Note: sample_idx used in pipeline is the index after shuffle.
    
    for idx, sample in tqdm(enumerate(ds), total=len(ds)):
        if extracted_count >= args.num_samples:
            break
            
        # sample_idx is simply idx in this loop
        
        image = sample['image'] # PIL Image
        query = sample['text']
        gt_mask = np.array(sample['mask']) > 0
        
        raw_img = image.convert("RGB")


        # Load Cached Candidates
        # Structure: cache/masks/sample_{idx}/mask_{cand_idx}_v{version}.npz
        sample_cache_dir = os.path.join(args.cache_dir, "masks", f"sample_{idx}")
        if not os.path.exists(sample_cache_dir):
            continue
            
        candidates = []
        
        # 1. Load Planner Output (Reconstruct generic candidates list from cache content)
        # We don't have the original 'candidates' list from Planner here easily unless we load plans.json
        # Let's verify against plans if possible, or just scan the directory.
        
        # Scan directory for all masks
        files = os.listdir(sample_cache_dir)
        
        # We need to group by hypothesis index (mask_0, mask_1, etc.)
        # and pick the best version for each hypothesis (Oracle logic)
        
        hyp_indices = set()
        for f in files:
            if f.startswith("mask_") and f.endswith(".npz"):
                parts = f.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                     hyp_indices.add(int(parts[1]))
        
        if not hyp_indices:
            continue

        valid_hypotheses = []

        for h_idx in hyp_indices:
            # Find best version for this hypothesis
            curr_best_iou = -1.0
            curr_best_mask = None
            curr_best_ver = -1
            
            # Check v0-v9
            found_any = False
            for v in range(10):
                fname = f"mask_{h_idx}_v{v}.npz"
                if fname not in files:
                     # fallback logic check
                     if v == 0 and f"mask_{h_idx}.npz" in files:
                         fname = f"mask_{h_idx}.npz"
                     else:
                         continue
                
                path = os.path.join(sample_cache_dir, fname)
                try:
                    with np.load(path) as data:
                        mask = data['mask']
                        iou = calculate_iou(mask, gt_mask)
                        if iou > curr_best_iou:
                            curr_best_iou = iou
                            curr_best_mask = mask
                            curr_best_ver = v
                        found_any = True
                except:
                    continue
            
            if found_any:
                valid_hypotheses.append({
                    "id": h_idx,
                    "iou": curr_best_iou,
                    "mask": curr_best_mask,
                    "version": curr_best_ver
                })

        if not valid_hypotheses:
            continue

        # Sort by IoU
        valid_hypotheses.sort(key=lambda x: x["iou"], reverse=True)
        
        best_cand = valid_hypotheses[0]
        
        # Check if Sample qualifies
        if best_cand["iou"] < args.min_iou:
            continue
            
        # Select Representative Candidates
        worst_cand = valid_hypotheses[-1]
        
        # Median
        mid_idx = len(valid_hypotheses) // 2
        avg_cand = valid_hypotheses[mid_idx]
        
        # Deduplicate if very few candidates
        selected = {
            "Best": best_cand,
            "Average": avg_cand,
            "Poor": worst_cand
        }
        
        print(f"Found Sample {idx}: Best IoU={best_cand['iou']:.4f}")
        
        # Save Artifacts
        sample_out_dir = os.path.join(args.output_dir, f"sample_{idx}")
        os.makedirs(sample_out_dir, exist_ok=True)
        
        # 1. Save Raw
        raw_img.save(os.path.join(sample_out_dir, "raw_image.jpg"))
        
        # 2. Save GT Overlay
        gt_overlay = apply_red_alpha_overlay(raw_img, gt_mask, alpha=0.6)
        gt_overlay.save(os.path.join(sample_out_dir, "gt_overlay.jpg"))
        
        # 3. Retrieve Scores (Optional - simplified from cache/scores if exists)
        # Try to load scores.json
        scores_file = os.path.join(args.cache_dir, "scores", f"sample_{idx}", "scores.json")
        scores_data = {}
        if os.path.exists(scores_file):
            try:
                with open(scores_file, 'r') as f:
                    scores_data = json.load(f)
            except:
                pass
                
        summary_info = {
            "sample_idx": idx,
            "query": query,
            "candidates": []
        }
        
        for name, cand in selected.items():
            # Generate Overlay
            overlay = apply_red_alpha_overlay(raw_img, cand["mask"], alpha=0.5)
            overlay.save(os.path.join(sample_out_dir, f"{name.lower()}_{cand['iou']:.2f}.jpg"))
            
            # Get Score
            score_val = "N/A"
            reasoning = "N/A"
            
            # Lookup in scores data
            # Key format: indices are strings "0", "1"
            h_key = str(cand["id"])
            v_key = f"v{cand['version']}"
            
            if h_key in scores_data:
                if v_key in scores_data[h_key]:
                     s_entry = scores_data[h_key][v_key]
                     score_val = s_entry.get("total_score", "N/A")
                     reasoning = s_entry.get("breakdown", {}).get("reasoning", "N/A")
                elif "v0" in scores_data[h_key]:
                     # Display fallback info if that's what we have
                     score_val = f"{scores_data[h_key]['v0'].get('total_score', 'N/A')} (v0 fallback)"
            
            cand_info = {
                "type": name,
                "iou": float(cand["iou"]),
                "version": cand["version"],
                "planner_id": cand["id"],
                "vlm_score": score_val,
                "vlm_reasoning": reasoning
            }
            summary_info["candidates"].append(cand_info)
            
        with open(os.path.join(sample_out_dir, "info.json"), "w") as f:
            json.dump(summary_info, f, indent=2)
            
        extracted_count += 1

    print(f"\nExtraction Complete! Check: {args.output_dir}")

if __name__ == "__main__":
    main()
