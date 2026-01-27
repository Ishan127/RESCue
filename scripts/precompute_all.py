"""
MASTER PRECOMPUTE SCRIPT
========================
Runs all preprocessing phases sequentially on 8 GPUs.

Phases:
1. Deploy Planner (8 GPUs) → Generate 512 plans per sample
2. Deploy SAM (8 GPUs) → Generate masks for all hypotheses  
3. Deploy CLIP (8 GPUs) → Get CLIP scores for all masks
4. Deploy Verifier (8 GPUs) → Precompute VLM pointwise scores

After this completes, use evaluate_tournament.py to run quick experiments.

Usage:
    python scripts/precompute_all.py --phase all
    python scripts/precompute_all.py --phase plans
    python scripts/precompute_all.py --phase masks
    python scripts/precompute_all.py --phase clip
    python scripts/precompute_all.py --phase vlm
"""
import argparse
import json
import os
import sys
import subprocess
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser(description="Master precompute script")
parser.add_argument("--phase", choices=["all", "plans", "masks", "clip", "vlm"], default="all")
parser.add_argument("--cache_dir", default="cache", help="Cache directory")
parser.add_argument("--max_n", type=int, default=512, help="Hypotheses per sample")
parser.add_argument("--workers", type=int, default=64, help="Parallel workers")
parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction")
parser.add_argument("--planner_url", default="http://localhost:8002/v1")
parser.add_argument("--sam_url", default="http://localhost:8001")
parser.add_argument("--clip_url", default="http://localhost:8003/verify")
parser.add_argument("--verifier_url", default="http://localhost:8000/v1")
parser.add_argument("--planner_strategy", default=None, help="Filter planner strategy")
args = parser.parse_args()

# Set environment
os.environ["PLANNER_API_BASE"] = args.planner_url
os.environ["VERIFIER_API_BASE"] = args.verifier_url

from datasets import load_dataset
from PIL import Image


def load_dataset_samples(fraction):
    """Load ReasonSeg dataset."""
    print("Loading ReasonSeg dataset...")
    ds = load_dataset("Ricky06662/ReasonSeg_test", split="test")
    num_samples = int(len(ds) * fraction)
    print(f"Samples: {num_samples}/{len(ds)}")
    return ds.shuffle(seed=42).select(range(num_samples))


# ============================================================================
# PHASE 1: GENERATE PLANS
# ============================================================================
def run_phase_plans(ds, cache_dir, max_n, workers):
    """Generate hypotheses for all samples."""
    from src.planner import Planner
    
    plans_dir = os.path.join(cache_dir, "plans")
    os.makedirs(plans_dir, exist_ok=True)
    
    plans_path = os.path.join(plans_dir, "plans.json")
    
    # Load existing plans if any
    existing_plans = {}
    if os.path.exists(plans_path):
        with open(plans_path, 'r') as f:
            existing_plans = json.load(f)
        print(f"Loaded {len(existing_plans)} existing plans")
    
    planner = Planner(api_base=args.planner_url)
    
    def process_sample(item):
        sample_idx, sample = item
        sample_key = f"sample_{sample_idx}"
        
        # Check if already have enough hypotheses
        if sample_key in existing_plans:
            existing_hyps = existing_plans[sample_key].get('hypotheses', [])
            if len(existing_hyps) >= max_n:
                return sample_key, existing_plans[sample_key]
        
        image = sample.get('image')
        query = sample.get('text') or sample.get('query') or sample.get('sentence')
        
        if image is None or query is None:
            return sample_key, None
        
        try:
            hypotheses = planner.generate_hypotheses(image, query, N=max_n, strategy_filter=args.planner_strategy)
            
            # FALLBACK: If not enough, retry with higher temperature
            retry_count = 0
            while len(hypotheses) < max_n and retry_count < 3:
                retry_count += 1
                print(f"[Retry {retry_count}] Sample {sample_idx}: {len(hypotheses)}/{max_n}")
                more = planner.generate_hypotheses(image, query, N=max_n - len(hypotheses), 
                                                   temperature=0.9 + retry_count * 0.1)
                hypotheses.extend(more)
            
            return sample_key, {"query": query, "hypotheses": hypotheses[:max_n]}
        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            return sample_key, None
    
    work_items = list(enumerate(ds))
    results = dict(existing_plans)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 1: Plans"):
            key, result = future.result()
            if result is not None:
                results[key] = result
    
    # Save
    with open(plans_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved {len(results)} plans to {plans_path}")
    return results


# ============================================================================
# PHASE 2: GENERATE MASKS
# ============================================================================
def run_phase_masks(ds, plans, cache_dir, workers):
    """Generate masks for all hypotheses using SAM."""
    from src.executor import Executor
    
    masks_dir = os.path.join(cache_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    executor = Executor(remote_url=args.sam_url, timeout=300)
    
    def process_sample(item):
        sample_idx, sample = item
        sample_key = f"sample_{sample_idx}"
        
        if sample_key not in plans:
            return sample_key, 0
        
        image = sample.get('image')
        if image is None:
            return sample_key, 0
        
        sample_dir = os.path.join(masks_dir, sample_key)
        os.makedirs(sample_dir, exist_ok=True)
        
        hypotheses = plans[sample_key].get('hypotheses', [])
        masks_generated = 0
        
        for hyp_idx, hyp in enumerate(hypotheses):
            mask_path = os.path.join(sample_dir, f"mask_{hyp_idx}.npz")
            
            if os.path.exists(mask_path):
                masks_generated += 1
                continue
            
            try:
                bbox = hyp.get('box') or hyp.get('bbox')
                
                # Use prompts_list format like original pipeline
                prompts_list = [{
                    "type": "box",
                    "box": bbox,
                    "label": True
                }]
                masks = executor.segment(image, prompts_list=prompts_list)
                
                if masks and len(masks) > 0:
                    np.savez_compressed(mask_path, mask=masks[0])
                    masks_generated += 1
            except Exception as e:
                pass
        
        return sample_key, masks_generated
    
    work_items = list(enumerate(ds))
    total_masks = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor_pool:
        futures = {executor_pool.submit(process_sample, item): item[0] for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2: Masks"):
            key, count = future.result()
            total_masks += count
    
    print(f"Generated {total_masks} masks in {masks_dir}")
    return total_masks


# ============================================================================
# PHASE 3: GET CLIP SCORES
# ============================================================================
def run_phase_clip(ds, plans, cache_dir, workers):
    """Get CLIP scores for all masks."""
    from src.clip_verifier import ClipVerifier
    
    masks_dir = os.path.join(cache_dir, "masks")
    clip_verifier = ClipVerifier(server_url=args.clip_url)
    
    def process_sample(item):
        sample_idx, sample = item
        sample_key = f"sample_{sample_idx}"
        
        if sample_key not in plans:
            return sample_key, None
        
        sample_masks_dir = os.path.join(masks_dir, sample_key)
        clip_path = os.path.join(sample_masks_dir, "clip_scores.json")
        
        # Skip if already computed
        if os.path.exists(clip_path):
            return sample_key, "cached"
        
        image = sample.get('image')
        query = plans[sample_key].get('query', '')
        hypotheses = plans[sample_key].get('hypotheses', [])
        
        if image is None:
            return sample_key, None
        
        # Load masks
        masks = []
        for hyp_idx in range(len(hypotheses)):
            mask_path = os.path.join(sample_masks_dir, f"mask_{hyp_idx}.npz")
            if os.path.exists(mask_path):
                masks.append(np.load(mask_path)['mask'])
            else:
                masks.append(None)
        
        valid_masks = [m for m in masks if m is not None]
        if not valid_masks:
            return sample_key, None
        
        try:
            scores = clip_verifier.verify_batch(image, valid_masks, query)
            
            # Map scores back to hypothesis indices
            clip_scores = {}
            valid_idx = 0
            for hyp_idx, mask in enumerate(masks):
                if mask is not None:
                    clip_scores[hyp_idx] = scores[valid_idx]
                    valid_idx += 1
                else:
                    clip_scores[hyp_idx] = 0.0
            
            with open(clip_path, 'w') as f:
                json.dump(clip_scores, f)
            
            return sample_key, len(scores)
        except Exception as e:
            print(f"CLIP error on {sample_key}: {e}")
            return sample_key, None
    
    work_items = list(enumerate(ds))
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 3: CLIP"):
            future.result()
    
    print("CLIP scoring complete")


# ============================================================================
# PHASE 4: VLM POINTWISE SCORING
# ============================================================================
def run_phase_vlm(ds, plans, cache_dir, workers):
    """Precompute VLM pointwise scores for all masks."""
    from src.verifier import Verifier
    from src.api_utils import encode_pil_image
    
    masks_dir = os.path.join(cache_dir, "masks")
    verifier = Verifier()
    
    def process_sample(item):
        sample_idx, sample = item
        sample_key = f"sample_{sample_idx}"
        
        if sample_key not in plans:
            return sample_key, None
        
        sample_masks_dir = os.path.join(masks_dir, sample_key)
        vlm_path = os.path.join(sample_masks_dir, "vlm_scores.json")
        
        # Skip if already computed
        if os.path.exists(vlm_path):
            return sample_key, "cached"
        
        image = sample.get('image')
        query = plans[sample_key].get('query', '')
        hypotheses = plans[sample_key].get('hypotheses', [])
        
        if image is None:
            return sample_key, None
        
        # Load masks
        masks = []
        valid_indices = []
        for hyp_idx in range(len(hypotheses)):
            mask_path = os.path.join(sample_masks_dir, f"mask_{hyp_idx}.npz")
            if os.path.exists(mask_path):
                masks.append(np.load(mask_path)['mask'])
                valid_indices.append(hyp_idx)
        
        if not masks:
            return sample_key, None
        
        try:
            # Pre-encode image
            base64_img = encode_pil_image(image)
            w, h = image.size
            
            # Run pointwise scoring
            results = verifier._pointwise_score_batch(base64_img, w, h, masks, query, max_workers=16)
            
            # Map scores back
            vlm_scores = {}
            for res in results:
                mask_idx = res.get('mask_idx', 0)
                hyp_idx = valid_indices[mask_idx]
                vlm_scores[hyp_idx] = {
                    'total_score': res.get('total_score', 0),
                    'breakdown': res.get('breakdown', {})
                }
            
            with open(vlm_path, 'w') as f:
                json.dump(vlm_scores, f)
            
            return sample_key, len(results)
        except Exception as e:
            print(f"VLM error on {sample_key}: {e}")
            return sample_key, None
    
    work_items = list(enumerate(ds))
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 4: VLM"):
            future.result()
    
    print("VLM pointwise scoring complete")


# ============================================================================
# MAIN
# ============================================================================
def main():
    os.makedirs(args.cache_dir, exist_ok=True)
    
    ds = load_dataset_samples(args.fraction)
    
    plans = {}
    plans_path = os.path.join(args.cache_dir, "plans", "plans.json")
    if os.path.exists(plans_path):
        with open(plans_path, 'r') as f:
            plans = json.load(f)
    
    if args.phase in ["all", "plans"]:
        print("\n" + "="*60)
        print("PHASE 1: GENERATING PLANS")
        print("="*60)
        plans = run_phase_plans(ds, args.cache_dir, args.max_n, args.workers)
    
    if args.phase in ["all", "masks"]:
        print("\n" + "="*60)
        print("PHASE 2: GENERATING MASKS")
        print("="*60)
        run_phase_masks(ds, plans, args.cache_dir, min(16, args.workers))
    
    if args.phase in ["all", "clip"]:
        print("\n" + "="*60)
        print("PHASE 3: CLIP SCORING")
        print("="*60)
        run_phase_clip(ds, plans, args.cache_dir, args.workers)
    
    if args.phase in ["all", "vlm"]:
        print("\n" + "="*60)
        print("PHASE 4: VLM POINTWISE SCORING")
        print("="*60)
        run_phase_vlm(ds, plans, args.cache_dir, args.workers)
    
    print("\n" + "="*60)
    print("PRECOMPUTE COMPLETE!")
    print("="*60)
    print(f"Cache directory: {args.cache_dir}")
    print("Run evaluation with: python scripts/evaluate_tournament.py")


if __name__ == "__main__":
    main()
