"""
Optimized RESCue Pipeline for speed.

Key optimizations:
1. Minimal logging (verbose=False by default)
2. Separate generation and verification phases
3. Batched operations where possible
4. Heuristic-based selection option (no VLM calls)
"""
from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .utils import load_image, get_device, calculate_iou
import numpy as np


class RESCuePipelineFast:
    def __init__(self, 
                 planner_model="Qwen/Qwen3-VL-32B-Thinking",
                 verifier_model="Qwen/Qwen3-VL-32B-Thinking",
                 executor_model="facebook/sam3",
                 planner_api_base="http://localhost:8000/v1",
                 executor_api_base="http://localhost:8001",
                 verbose=False):
        
        self.verbose = verbose
        self.device = get_device()
        
        if verbose:
            print("Initializing RESCue Pipeline (Fast Mode)...")
        
        self.planner = Planner(
            model_path=planner_model, 
            api_base=planner_api_base,
            device=self.device
        )
        
        self.executor = Executor(
            model_path=executor_model, 
            device=self.device,
            remote_url=executor_api_base
        )
        
        self.verifier = Verifier(
            client=self.planner.client,
            model_path=verifier_model,
            api_base=planner_api_base
        )
        
        self._cached_image_path = None
        
        if verbose:
            print("Pipeline ready.")

    def generate_all_candidates(self, image_path, query, N=64, gt_mask=None):
        """
        Generate all N candidates (hypotheses + masks).
        Returns list of candidate dicts with masks and metadata.
        """
        image = load_image(image_path)
        
        hypotheses = self.planner.generate_hypotheses(image_path, query, N=N)
        
        if self.verbose:
            print(f"Generated {len(hypotheses)} hypotheses")
        
        self.executor.encode_image(np.array(image))
        self._cached_image_path = image_path
        
        candidates = []
        
        for i, hyp in enumerate(hypotheses):
            box = hyp['box']
            noun_phrase = hyp['noun_phrase']
            
            try:
                masks = self.executor.predict_masks(box, noun_phrase)
                
                for j, mask in enumerate(masks):
                    cand = {
                        'id': f"H{i}_M{j}",
                        'idx': len(candidates),
                        'mask': mask,
                        'box': box,
                        'noun_phrase': noun_phrase,
                        'quality': self._compute_mask_quality(mask)
                    }
                    
                    if gt_mask is not None:
                        cand['iou'] = calculate_iou(mask, gt_mask)
                    
                    candidates.append(cand)
            except Exception as e:
                if self.verbose:
                    print(f"Mask generation failed for H{i}: {e}")
        
        if self.verbose:
            print(f"Total candidates: {len(candidates)}")
        
        return candidates

    def verify_and_select(self, image, candidates, query, mode="comparative"):
        """
        Verify candidates and return index of best one.
        
        Modes:
        - "comparative": VLM compares all masks in grid
        - "individual": VLM scores each mask separately
        - "heuristic": Use mask quality score (no VLM, fastest)
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return 0
        
        if mode == "heuristic":
            return self._select_by_heuristic(candidates)
        
        masks_list = [c['mask'] for c in candidates]
        
        similarities = self._compute_mask_similarities(masks_list)
        avg_sim = np.mean(similarities) if similarities else 0
        
        if avg_sim > 0.85:
            if self.verbose:
                print(f"Masks too similar ({avg_sim:.2f}), using heuristic")
            return self._select_by_heuristic(candidates)
        
        if mode == "comparative":
            return self._select_comparative(image, candidates, query)
        else:
            return self._select_individual(image, candidates, query)

    def _select_by_heuristic(self, candidates):
        """Select by mask quality score (no VLM calls)."""
        best_idx = 0
        best_quality = -1
        
        for i, cand in enumerate(candidates):
            quality = cand.get('quality', self._compute_mask_quality(cand['mask']))
            if quality > best_quality:
                best_quality = quality
                best_idx = i
        
        return best_idx

    def _select_comparative(self, image, candidates, query):
        """Use VLM comparative ranking."""
        masks_list = [c['mask'] for c in candidates]
        
        try:
            results = self.verifier.verify_batch_comparative(image, masks_list, query)
            
            best_rank = 999
            best_idx = 0
            
            for r in results:
                if r['rank'] < best_rank:
                    best_rank = r['rank']
                    best_idx = r['mask_idx']
            
            return best_idx
        except Exception as e:
            if self.verbose:
                print(f"Comparative verification failed: {e}")
            return self._select_by_heuristic(candidates)

    def _select_individual(self, image, candidates, query):
        """Use VLM individual scoring."""
        best_score = -1
        best_idx = 0
        
        for i, cand in enumerate(candidates):
            try:
                result = self.verifier.verify(image, cand['mask'], query)
                score = result.get('total', result.get('score', 50))
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            except:
                pass
        
        return best_idx

    def _compute_mask_similarities(self, masks):
        """Compute pairwise IoU (for first 10 masks to save time)."""
        masks = masks[:10]
        similarities = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                iou = calculate_iou(masks[i], masks[j])
                similarities.append(iou)
        return similarities

    def _compute_mask_quality(self, mask):
        """Heuristic quality score (0-100)."""
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        if mask_np.sum() == 0:
            return 0.0
        
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y_idx = np.where(rows)[0]
        x_idx = np.where(cols)[0]
        
        if len(y_idx) == 0 or len(x_idx) == 0:
            return 0.0
        
        bbox_area = (y_idx[-1] - y_idx[0] + 1) * (x_idx[-1] - x_idx[0] + 1)
        mask_area = mask_np.sum()
        solidity = mask_area / bbox_area if bbox_area > 0 else 0
        
        h, w = mask_np.shape
        ratio = mask_area / (h * w)
        
        size_score = 1.0
        if ratio < 0.005:
            size_score = 0.3
        elif ratio < 0.01:
            size_score = 0.6
        elif ratio > 0.9:
            size_score = 0.5
        elif ratio > 0.7:
            size_score = 0.8
        
        return min(100, solidity * 60 + size_score * 40)

    def run(self, image_path, query, N=4, gt_mask=None, verification_mode="comparative"):
        """Convenience method for single-run evaluation."""
        image = load_image(image_path)
        candidates = self.generate_all_candidates(image_path, query, N=N, gt_mask=gt_mask)
        
        if not candidates:
            return None
        
        best_idx = self.verify_and_select(image, candidates, query, mode=verification_mode)
        
        return {
            'best_mask': candidates[best_idx]['mask'],
            'best_candidate': candidates[best_idx],
            'all_candidates': candidates,
            'image': image
        }
