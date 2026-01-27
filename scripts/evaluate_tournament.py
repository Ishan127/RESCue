"""
TOURNAMENT-ONLY EVALUATION
==========================
Runs only the tournament phase using precomputed scores.
All plans, masks, CLIP scores, and VLM pointwise scores are loaded from cache.

Usage:
    python scripts/evaluate_tournament.py --max_n 16
    python scripts/evaluate_tournament.py --max_n 32 --strategy spatial
    python scripts/evaluate_tournament.py --max_n 128
"""
import argparse
import json
import os
import sys
import random
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser(description="Tournament-only evaluation with precomputed scores")
parser.add_argument("--max_n", type=int, default=16, help="Number of hypotheses to use")
parser.add_argument("--cache_dir", default="cache", help="Cache directory")
parser.add_argument("--strategy", default=None, help="Filter by strategy (e.g., spatial)")
parser.add_argument("--verifier_url", default="http://localhost:8000/v1")
parser.add_argument("--workers", type=int, default=32, help="Parallel workers for tournament")
parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset")
parser.add_argument("--skip_tournament", action="store_true", help="Skip tournament, use pointwise scores only")
args = parser.parse_args()

os.environ["VERIFIER_API_BASE"] = args.verifier_url

from datasets import load_dataset
from PIL import Image
from src.utils import calculate_iou


@dataclass  
class PrecomputedSample:
    sample_idx: int
    image: Any
    query: str
    gt_mask: Optional[np.ndarray]
    hypotheses: List[Dict]
    masks: List[np.ndarray]
    clip_scores: Dict[int, float]
    vlm_scores: Dict[int, Dict]


def load_precomputed_sample(sample_idx: int, sample: Dict, plans: Dict, 
                           masks_dir: str, max_n: int, strategy: str = None) -> Optional[PrecomputedSample]:
    """Load all precomputed data for a sample."""
    sample_key = f"sample_{sample_idx}"
    
    if sample_key not in plans:
        return None
    
    image = sample.get('image')
    query = sample.get('text') or sample.get('query') or sample.get('sentence')
    gt_mask = sample.get('mask') or sample.get('label')
    
    if image is None or query is None:
        return None
    
    if gt_mask is not None:
        gt_mask = np.array(gt_mask) > 0
    
    sample_masks_dir = os.path.join(masks_dir, sample_key)
    
    # Load CLIP scores
    clip_path = os.path.join(sample_masks_dir, "clip_scores.json")
    clip_scores = {}
    if os.path.exists(clip_path):
        with open(clip_path, 'r') as f:
            clip_scores = {int(k): v for k, v in json.load(f).items()}
    
    # Load VLM scores
    vlm_path = os.path.join(sample_masks_dir, "vlm_scores.json")
    vlm_scores = {}
    if os.path.exists(vlm_path):
        with open(vlm_path, 'r') as f:
            vlm_scores = {int(k): v for k, v in json.load(f).items()}
    
    # Load hypotheses and filter by strategy
    all_hypotheses = plans[sample_key].get('hypotheses', [])
    
    if strategy:
        all_hypotheses = [(i, h) for i, h in enumerate(all_hypotheses) 
                          if h.get('strategy', '').lower() == strategy.lower()]
    else:
        all_hypotheses = list(enumerate(all_hypotheses))
    
    # Filter to those with precomputed scores
    valid_candidates = []
    for orig_idx, hyp in all_hypotheses:
        mask_path = os.path.join(sample_masks_dir, f"mask_{orig_idx}.npz")
        if os.path.exists(mask_path) and orig_idx in vlm_scores:
            valid_candidates.append((orig_idx, hyp))
    
    # Sample up to max_n
    if len(valid_candidates) > max_n:
        valid_candidates = random.sample(valid_candidates, max_n)
    
    if not valid_candidates:
        return None
    
    # Load masks and scores for selected candidates
    hypotheses = []
    masks = []
    selected_clip = {}
    selected_vlm = {}
    
    for new_idx, (orig_idx, hyp) in enumerate(valid_candidates):
        mask_path = os.path.join(sample_masks_dir, f"mask_{orig_idx}.npz")
        mask = np.load(mask_path)['mask']
        
        hypotheses.append(hyp)
        masks.append(mask)
        selected_clip[new_idx] = clip_scores.get(orig_idx, 0.0)
        selected_vlm[new_idx] = vlm_scores.get(orig_idx, {'total_score': 0})
    
    return PrecomputedSample(
        sample_idx=sample_idx,
        image=image,
        query=query,
        gt_mask=gt_mask,
        hypotheses=hypotheses,
        masks=masks,
        clip_scores=selected_clip,
        vlm_scores=selected_vlm
    )


def build_ranking(sample: PrecomputedSample) -> List[Dict]:
    """Build initial ranking from precomputed scores."""
    ranking = []
    
    for idx in range(len(sample.masks)):
        vlm_data = sample.vlm_scores.get(idx, {})
        clip_score = sample.clip_scores.get(idx, 0.0)
        
        total_score = vlm_data.get('total_score', 0)
        
        ranking.append({
            'mask_idx': idx,
            'score': total_score,
            'vlm_score': total_score,
            'clip_score': clip_score,
            'breakdown': vlm_data.get('breakdown', {})
        })
    
    # Sort by score descending
    ranking.sort(key=lambda x: x['score'], reverse=True)
    
    for i, r in enumerate(ranking):
        r['rank'] = i + 1
    
    return ranking


def run_tournament(sample: PrecomputedSample, ranking: List[Dict], verifier) -> List[Dict]:
    """Run only the tournament phase using precomputed pointwise scores."""
    from src.parallel_tournament import parallel_pyramid_tournament
    
    n = len(ranking)
    if n <= 1:
        return ranking
    
    # Run parallel tournament
    try:
        ranking = parallel_pyramid_tournament(verifier, sample.image, sample.masks, sample.query, ranking)
    except Exception as e:
        print(f"Tournament error: {e}")
    
    return ranking


def evaluate_sample(sample: PrecomputedSample, verifier, skip_tournament: bool) -> Dict:
    """Evaluate a single sample using precomputed scores + tournament."""
    t_start = time.time()
    
    # Build ranking from precomputed scores
    ranking = build_ranking(sample)
    
    # Run tournament (only online computation)
    if not skip_tournament and len(ranking) > 1:
        ranking = run_tournament(sample, ranking, verifier)
    
    t_end = time.time()
    
    # Calculate IoU
    iou = 0.0
    if sample.gt_mask is not None and ranking:
        best_mask = sample.masks[ranking[0]['mask_idx']]
        iou = calculate_iou(best_mask, sample.gt_mask)
    
    return {
        "sample_idx": sample.sample_idx,
        "iou": iou,
        "num_masks": len(sample.masks),
        "time": t_end - t_start,
        "best_mask_idx": ranking[0]['mask_idx'] if ranking else -1,
        "top_score": ranking[0]['score'] if ranking else 0
    }


def main():
    plans_dir = os.path.join(args.cache_dir, "plans")
    masks_dir = os.path.join(args.cache_dir, "masks")
    
    # Load plans
    plans_path = os.path.join(plans_dir, "plans.json")
    print(f"Loading plans from {plans_path}...")
    with open(plans_path, 'r') as f:
        plans = json.load(f)
    
    print(f"Loaded {len(plans)} sample plans")
    
    # Load dataset
    print("Loading ReasonSeg dataset...")
    ds = load_dataset("Ricky06662/ReasonSeg_test", split="test")
    num_samples = int(len(ds) * args.fraction)
    ds = ds.shuffle(seed=42).select(range(num_samples))
    
    # Initialize verifier (only needed for tournament)
    verifier = None
    if not args.skip_tournament:
        from src.verifier import Verifier
        verifier = Verifier()
    
    # Load precomputed samples
    print(f"Loading precomputed data (max_n={args.max_n}, strategy={args.strategy})...")
    samples = []
    for i, sample in enumerate(tqdm(ds, desc="Loading cache")):
        precomputed = load_precomputed_sample(i, sample, plans, masks_dir, args.max_n, args.strategy)
        if precomputed:
            samples.append(precomputed)
    
    print(f"Loaded {len(samples)} samples with precomputed scores")
    
    # Run evaluation
    print(f"\nRunning tournament evaluation...")
    results = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_sample, s, verifier, args.skip_tournament): s.sample_idx 
                   for s in samples}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tournament"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
    
    # Compute metrics
    ious = [r['iou'] for r in results]
    times = [r['time'] for r in results]
    
    mean_iou = np.mean(ious) if ious else 0
    std_iou = np.std(ious) if ious else 0
    mean_time = np.mean(times) if times else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(results)} samples | N={args.max_n} | Strategy={args.strategy or 'all'}")
    print(f"{'='*60}")
    print(f"  Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"  Mean Time: {mean_time:.3f}s per sample (tournament only)")
    print(f"  Total Time: {sum(times):.1f}s")
    print(f"{'='*60}")
    
    # Save results
    results_dir = os.path.join(args.cache_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_path = os.path.join(results_dir, f"tournament_n{args.max_n}_{args.strategy or 'all'}.json")
    with open(output_path, 'w') as f:
        json.dump({
            "config": vars(args),
            "metrics": {
                "mean_iou": mean_iou,
                "std_iou": std_iou,
                "mean_time": mean_time,
                "num_samples": len(results)
            },
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
