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
    
    def process_sample_by_idx(sample_idx):
        sample_key = f"sample_{sample_idx}"
        
        # Check if already have enough hypotheses -> SKIP loading image
        if sample_key in existing_plans:
            existing_hyps = existing_plans[sample_key].get('hypotheses', [])
            if len(existing_hyps) >= max_n:
                return sample_key, existing_plans[sample_key]
        
        # Load sample only if needed
        try:
            sample = ds[sample_idx]
            image = sample.get('image')
            query = sample.get('text') or sample.get('query') or sample.get('sentence')
            
            if image is None or query is None:
                return sample_key, None
            
            hypotheses = planner.generate_hypotheses(image, query, N=max_n, strategy_filter=args.planner_strategy)
            
            # FALLBACK: If not enough, retry with higher temperature
            retry_count = 0
            while len(hypotheses) < max_n and retry_count < 1:
                retry_count += 1
                print(f"[Retry {retry_count}] Sample {sample_idx}: {len(hypotheses)}/{max_n}")
                more = planner.generate_hypotheses(image, query, N=max_n - len(hypotheses), 
                                                   temperature=0.9 + retry_count * 0.1)
                hypotheses.extend(more)
            
            return sample_key, {"query": query, "hypotheses": hypotheses[:max_n]}
        except Exception as e:
            print(f"Error on sample {sample_idx}: {e}")
            return sample_key, None
    
    # Use indices list instead of loading full dataset
    num_samples = len(ds)
    all_indices = list(range(num_samples))
    results = dict(existing_plans)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit indices
        futures = {executor.submit(process_sample_by_idx, idx): idx for idx in all_indices}
        
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
# ============================================================================
# PHASE 2: GENERATE MASKS
# ============================================================================
def run_phase_masks(ds, plans, cache_dir, workers):
    """Generate masks for all hypotheses using SAM."""
    from src.executor import Executor
    
    masks_dir = os.path.join(cache_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    # Multi-GPU Scaling: Routing based on sample_idx
    # We don't verify connection here because we will connect dynamically inside threads
    # BUT we do need to verify at least one to ensure we don't start blindly
    # (Actually phase2_masks.sh already verifies all 8, so we can skip verification here to be faster)
    
    # We will instantiate Executor inside the thread to target specific ports
    base_url = args.sam_url # e.g. http://localhost:8001
    base_port = 8001
    try:
        if base_url:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            if parsed.port:
                base_port = parsed.port
    except:
        pass

    import traceback
    
    def process_sample_by_idx(sample_idx):
        try:
            return _process_sample_unsafe(sample_idx)
        except Exception as e:
            print(f"CRASH on sample {sample_idx}: {e}")
            traceback.print_exc()
            return f"sample_{sample_idx}", 0

    import threading
    thread_local = threading.local()

    def _process_sample_unsafe(sample_idx):
        sample_key = f"sample_{sample_idx}"
        
        # Route to specific server (0-7)
        worker_id = sample_idx % 8 
        target_port = base_port + worker_id
        target_url = f"http://localhost:{target_port}"
        
        # Get or create thread-local executor for this specific target URL
        if not hasattr(thread_local, "executors"):
            thread_local.executors = {}
        
        if target_url not in thread_local.executors:
            # First time this thread talks to this port (or first time init)
            # Force CPU device to avoid initializing CUDA context in threads on client side
            thread_local.executors[target_url] = Executor(remote_url=target_url, timeout=600, device="cpu")
            
        local_executor = thread_local.executors[target_url]
        
        if sample_key not in plans:
            return sample_key, 0
            
        # Optimization: Check if all mask versions exist BEFORE loading image
        sample_dir = os.path.join(masks_dir, sample_key)
        all_exist = True
        hypotheses = plans[sample_key].get('hypotheses', [])
        for hyp_idx in range(len(hypotheses)):
            for ver in range(10):
                if not os.path.exists(os.path.join(sample_dir, f"mask_{hyp_idx}_v{ver}.npz")):
                    all_exist = False
                    break
            if not all_exist:
                break
        
        if all_exist and len(hypotheses) > 0:
            return sample_key, len(hypotheses) * 10
        
        # Load sample only if needed
        try:
            sample = ds[sample_idx]
            image = sample.get('image')
            if image is None:
                return sample_key, 0
                
            os.makedirs(sample_dir, exist_ok=True)
            masks_generated = 0
            
            # Loop 10 times for the 10 versions
            for ver in range(10):
                # Identify which hypotheses need this version generated
                prompts_list = []
                hyp_indices_to_process = []
                
                for hyp_idx, hyp in enumerate(hypotheses):
                    mask_path = os.path.join(sample_dir, f"mask_{hyp_idx}_v{ver}.npz")
                    if os.path.exists(mask_path):
                        continue
                        
                    bbox = hyp.get('box') or hyp.get('bbox')
                    prompts_list.append({
                        "type": "box",
                        "box": bbox,
                        "label": True
                    })
                    hyp_indices_to_process.append(hyp_idx)
                
                if not prompts_list:
                    masks_generated += len(hypotheses) # Count as done
                    continue
                
                try:
                    # Single batched call for ALL hypotheses for this version
                    masks = local_executor.segment(image, prompts_list=prompts_list)
                    
                    if masks:
                        # Save results mapping back to hypothesis index
                        for local_idx, mask in enumerate(masks):
                            if local_idx < len(hyp_indices_to_process):
                                true_hyp_idx = hyp_indices_to_process[local_idx]
                                mask_path = os.path.join(sample_dir, f"mask_{true_hyp_idx}_v{ver}.npz")
                                np.savez_compressed(mask_path, mask=mask)
                                masks_generated += 1
                except Exception:
                    # traceback.print_exc() # detailed remote error 
                    pass
            
            return sample_key, masks_generated
        except Exception:
            # traceback.print_exc()
            return sample_key, 0
    
    num_samples = len(ds)
    all_indices = list(range(num_samples))
    total_masks = 0

    with ThreadPoolExecutor(max_workers=workers) as executor_pool:
        futures = {executor_pool.submit(process_sample_by_idx, idx): idx for idx in all_indices}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 2: Masks"):
            key, count = future.result()
            total_masks += count
    
    print(f"Generated {total_masks} masks in {masks_dir}")
    return total_masks


# ============================================================================
# PHASE 3: GET CLIP SCORES
# ============================================================================
# ============================================================================
# PHASE 3: GET CLIP SCORES
# ============================================================================
def run_phase_clip(ds, plans, cache_dir, workers):
    """Get CLIP scores for all masks."""
    from src.clip_verifier import ClipVerifier
    
    masks_dir = os.path.join(cache_dir, "masks")
    
    # We will instantiate ClipVerifier inside threads
    base_url = args.clip_url
    base_port = 8011
    try:
        from urllib.parse import urlparse
        if base_url:
            parsed = urlparse(base_url)
            if parsed.port:
                base_port = parsed.port
    except:
        pass

    import threading
    thread_local = threading.local()

    def process_sample_by_idx(sample_idx):
        sample_key = f"sample_{sample_idx}"
        
        # Route to specific server (0-7)
        worker_id = sample_idx % 8 
        target_port = base_port + worker_id
        target_url = f"http://localhost:{target_port}/verify"
        
        # Get or create thread-local verifier
        if not hasattr(thread_local, "verifiers"):
            thread_local.verifiers = {}
            
        if target_url not in thread_local.verifiers:
            thread_local.verifiers[target_url] = ClipVerifier(server_url=target_url)
            
        local_verifier = thread_local.verifiers[target_url]
        
        if sample_key not in plans:
            return sample_key, None
        
        sample_masks_dir = os.path.join(masks_dir, sample_key)
        clip_path = os.path.join(sample_masks_dir, "clip_scores.json")
        
        # Skip if already computed -> Check file ON DISK, no image load logic
        if os.path.exists(clip_path):
            return sample_key, "cached"
            
        try:
            sample = ds[sample_idx]
            image = sample.get('image')
            query = plans[sample_key].get('query', '')
            hypotheses = plans[sample_key].get('hypotheses', [])
            
            if image is None:
                return sample_key, None
            
            # Load masks (all versions)
            masks = []
            mask_meta = [] # (hyp_idx, version)
            
            for hyp_idx in range(len(hypotheses)):
                # Look for versions
                for ver in range(10):
                    mask_path = os.path.join(sample_masks_dir, f"mask_{hyp_idx}_v{ver}.npz")
                    if os.path.exists(mask_path):
                        masks.append(np.load(mask_path)['mask'])
                        mask_meta.append((hyp_idx, ver))
                    elif ver == 0:
                        # Fallback for old single-version style
                        old_path = os.path.join(sample_masks_dir, f"mask_{hyp_idx}.npz")
                        if os.path.exists(old_path):
                             masks.append(np.load(old_path)['mask'])
                             mask_meta.append((hyp_idx, 0))
            
            if not masks:
                return sample_key, None
            
            scores = local_verifier.verify_batch(image, masks, query)
            
            # Map scores back to hypothesis indices structure
            clip_scores = {}
            for i, (hyp_idx, ver) in enumerate(mask_meta):
                if str(hyp_idx) not in clip_scores:
                    clip_scores[str(hyp_idx)] = {}
                clip_scores[str(hyp_idx)][f"v{ver}"] = scores[i]
            
            with open(clip_path, 'w') as f:
                json.dump(clip_scores, f)
            
            return sample_key, len(scores)
        except Exception as e:
            print(f"CLIP error on {sample_key}: {e}")
            return sample_key, None
    
    num_samples = len(ds)
    all_indices = list(range(num_samples))
    
    print(f"Starting Phase 3 with {workers} workers...", flush=True)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sample_by_idx, idx): idx for idx in all_indices}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Phase 3: CLIP"):
            future.result()
    
    print("CLIP scoring complete")


# ============================================================================
# PHASE 4: VLM POINTWISE SCORING
# ============================================================================
# ============================================================================
# PHASE 4: VLM POINTWISE SCORING
# ============================================================================
def run_phase_vlm(ds, plans, cache_dir, workers):
    """Compute VLM pointwise scores using vLLM endpoint."""
    from src.verifier import Verifier
    from src.utils import calculate_iou # Ensure imported
    
    masks_dir = os.path.join(cache_dir, "masks")
    
    # Single endpoint for vLLM (TP=8) OR Multiple (TP=4x2)
    verifier_urls = args.verifier_url.split(",")
    verifier_urls = [u.strip() for u in verifier_urls if u.strip()]
    
    import threading
    thread_local = threading.local()

    def process_sample_by_idx(sample_idx):
        sample_key = f"sample_{sample_idx}"
        
        # Round Robin Load Balancing
        target_url_idx = sample_idx % len(verifier_urls)
        target_url = verifier_urls[target_url_idx]
        
        # Get or create thread-local verifier for THIS url
        if not hasattr(thread_local, "verifiers"):
             thread_local.verifiers = {}
             
        if target_url not in thread_local.verifiers:
            thread_local.verifiers[target_url] = Verifier(
                model_path="Qwen/Qwen3-VL-30B-A3B-Thinking",
                api_base=target_url 
            )
            
        local_verifier = thread_local.verifiers[target_url]
        
        if sample_key not in plans:
            return sample_key, None
        
        sample_masks_dir = os.path.join(masks_dir, sample_key)
        vlm_path = os.path.join(sample_masks_dir, "vlm_scores.json")
        
        # Load existing scores to support partial resumption
        vlm_scores = {}
        if os.path.exists(vlm_path):
            try:
                with open(vlm_path, 'r') as f:
                    vlm_scores = json.load(f)
            except:
                pass
            
        try:
            sample = ds[sample_idx]
            image = sample.get('image')
            query = plans[sample_key].get('query', '')
            hypotheses = plans[sample_key].get('hypotheses', [])
            
            if image is None:
                return sample_key, None
            
            # 1. Collect Valid Mask Paths
            valid_mask_entries = [] # (path, hyp_idx, ver)
            completed_count = 0
            
            for hyp_idx in range(len(hypotheses)):
                hyp_str = str(hyp_idx)
                # check versions 0..9
                for ver in range(10):
                    # Check if already done
                    if hyp_str in vlm_scores and f"v{ver}" in vlm_scores[hyp_str]:
                        completed_count += 1
                        continue

                    # Check file existence
                    mask_path = os.path.join(sample_masks_dir, f"mask_{hyp_idx}_v{ver}.npz")
                    if os.path.exists(mask_path):
                        valid_mask_entries.append((mask_path, hyp_idx, ver))
                        
                # Fallback check for old non-versioned masks (v0)
                old_path = os.path.join(sample_masks_dir, f"mask_{hyp_idx}.npz")
                if os.path.exists(old_path):
                     # If v0 is somehow missing from VLM scores but v0 file exists (as mask_idx.npz)
                     if not (hyp_str in vlm_scores and "v0" in vlm_scores[hyp_str]):
                        valid_mask_entries.append((old_path, hyp_idx, 0))

             # If everything we could possibly score is already scored (or no masks exist to score)
            if not valid_mask_entries:
                return sample_key, "cached" if completed_count > 0 else None
            
            if not valid_mask_entries:
                return sample_key, None

            # 2. Process in Chunks
            chunk_size = 64
            vlm_scores = {}
            from src.utils import calculate_iou 
            
            for i in range(0, len(valid_mask_entries), chunk_size):
                chunk_entries = valid_mask_entries[i : i + chunk_size]
                chunk_masks = []
                chunk_meta = []
                
                # Load Chunk
                for path, h_idx, v in chunk_entries:
                    try:
                        with np.load(path) as data:
                            chunk_masks.append(data['mask'])
                        chunk_meta.append((h_idx, v))
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                
                if not chunk_masks: continue
                
                # Pruning Logic
                unique_masks = []
                unique_indices = []
                duplicate_map = {} 
                
                for m_idx in range(len(chunk_masks)):
                    current_mask = chunk_masks[m_idx]
                    current_h_idx, current_ver = chunk_meta[m_idx]
                    
                    is_duplicate = False
                    
                    if m_idx > 0:
                        prev_h_idx, prev_ver = chunk_meta[m_idx-1]
                        # Only prune if it's the SAME hypothesis (different version/jitter)
                        if prev_h_idx == current_h_idx:
                            iou = calculate_iou(current_mask, chunk_masks[m_idx-1])
                            if iou > 0.97:
                                is_duplicate = True
                                duplicate_map[m_idx] = m_idx - 1
                    
                    if not is_duplicate:
                        unique_masks.append(current_mask)
                        unique_indices.append(m_idx)
                
                if unique_masks:
                    results = local_verifier.verify_batch_pointwise(
                        image, 
                        unique_masks, 
                        query,
                        skip_clip=True,
                        skip_consistency=True,
                        max_workers=32 
                    )
                else:
                    results = []
                
                unique_results_map = {} 
                for r_idx, res in enumerate(results):
                     u_original_idx = unique_indices[res.get('mask_idx', r_idx)]
                     unique_results_map[u_original_idx] = res

                for m_idx in range(len(chunk_masks)):
                    source_idx = m_idx
                    while source_idx in duplicate_map:
                        source_idx = duplicate_map[source_idx]
                    
                    if source_idx in unique_results_map:
                        res = unique_results_map[source_idx]
                        hyp_idx, ver = chunk_meta[m_idx]
                        if str(hyp_idx) not in vlm_scores:
                            vlm_scores[str(hyp_idx)] = {}
                            
                        vlm_scores[str(hyp_idx)][f"v{ver}"] = {
                            'total_score': res.get('score', 0),
                            'breakdown': res.get('pointwise_details', {})
                        }
                
                del chunk_masks
                del unique_masks
            
            with open(vlm_path, 'w') as f:
                json.dump(vlm_scores, f)
            
            import gc
            gc.collect()
            
            return sample_key, len(valid_mask_entries)
        except Exception as e:
            print(f"VLM error on {sample_key}: {e}")
            import traceback
            traceback.print_exc()
            return sample_key, None
    
    num_samples = len(ds)
    all_indices = list(range(num_samples))
    
    print(f"Starting Phase 4 with {workers} workers (Targeting vLLM at {args.verifier_url})...", flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_sample_by_idx, idx): idx for idx in all_indices}
        
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
            print(f"DEBUG: Loaded {len(plans)} plans from {plans_path}")
            if len(plans) > 0:
                first_key = list(plans.keys())[0]
                print(f"DEBUG: First plan key example: '{first_key}'")
    else:
        print(f"DEBUG: Plans file not found at {plans_path}")
    
    if args.phase in ["all", "plans"]:
        print("\n" + "="*60)
        print("PHASE 1: GENERATING PLANS")
        print("="*60)
        plans = run_phase_plans(ds, args.cache_dir, args.max_n, args.workers)
    
    if args.phase in ["all", "masks"]:
        print("\n" + "="*60)
        print("PHASE 2: GENERATING MASKS")
        print("="*60)
        print(f"DEBUG: Starting Phase 2 with {len(plans)} plans available.")
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
