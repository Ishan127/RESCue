from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .utils import load_image, get_device, calculate_iou
import numpy as np

class RESCuePipeline:
    def __init__(self, 
                 planner_model="Qwen/Qwen3-VL-8B-Instruct",
                 verifier_model="Qwen/Qwen3-VL-32B-Thinking",
                 executor_model="facebook/sam3",
                 planner_api_base="http://localhost:8002/v1",
                 verifier_api_base="http://localhost:8000/v1",
                 executor_api_base="http://localhost:8001",
                 device=None,
                 dtype="auto",
                 quantization=None):
        print("Initializing RESCue Pipeline...")
        self.device = device or get_device()
        print(f"Pipeline using device: {self.device}")
        
        self.planner = Planner(
            model_path=planner_model, 
            api_base=planner_api_base,
            device=self.device, 
            dtype=dtype, 
            quantization=quantization
        )
        
        self.executor = Executor(
            model_path=executor_model, 
            device=self.device,
            remote_url=executor_api_base
        )
        
        
        from .api_utils import get_openai_client
        self.verifier = Verifier(
            client=get_openai_client(base_url=verifier_api_base),
            model_path=verifier_model,
            api_base=verifier_api_base
        )
        
        print(f"Planner: {planner_model}")
        print(f"Verifier: {verifier_model}")
        print("Pipeline Initialized.")

    def _compute_mask_similarities(self, masks):
        """Compute pairwise IoU between all masks to detect redundancy."""
        similarities = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                iou = calculate_iou(masks[i], masks[j])
                similarities.append(iou)
        return similarities
    
    def _compute_mask_quality(self, mask):
        """Compute heuristic quality score for a mask (0-100)."""
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        if mask_np.sum() == 0:
            return 0.0
        
        rows = np.any(mask_np, axis=1)
        cols = np.any(mask_np, axis=0)
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            return 0.0
            
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        
        bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
        mask_area = mask_np.sum()
        
        solidity = mask_area / bbox_area if bbox_area > 0 else 0
        
        h, w = mask_np.shape
        mask_ratio = mask_area / (h * w)
        
        size_score = 1.0
        if mask_ratio < 0.005:
            size_score = 0.3
        elif mask_ratio < 0.01:
            size_score = 0.6
        elif mask_ratio > 0.9:
            size_score = 0.5
        elif mask_ratio > 0.7:
            size_score = 0.8
        
        quality = (solidity * 60 + size_score * 40)
        return max(0, min(100, quality))

    def run(self, image_path, query, N=4, gt_mask=None, verification_mode="both"):
        """
        Run the pipeline with specified verification mode.
        
        Args:
            verification_mode: "comparative", "individual", or "both"
        
        Returns dict with keys depending on mode:
            - 'best_mask_comparative', 'best_mask_individual' if mode="both"
            - 'best_mask' for single mode
        """
        image = load_image(image_path)
        
        print(f"--- Step 1: Planning (N={N}) ---")
        hypotheses = self.planner.generate_hypotheses(image_path, query, N=N)
        print(f"Generated {len(hypotheses)} hypotheses.")
        
        candidates = []
        
        print(f"--- Step 2: Execution ---")
        # Stateless execution for load balancer compatibility
        # Each request sends the image to allow round-robin routing
        
        # Batch Execution
        prompt_list = []
        for i, hyp in enumerate(hypotheses):
            prompt_list.append({
                "type": "box",
                "box": hyp['box'],
                "label": True
            })
            
        print(f"Sending batch request with {len(prompt_list)} prompts...")
        all_masks = self.executor.segment(image, prompts_list=prompt_list)
        
        # Determine how many masks per hypothesis (usually 1, but server might return mulitple)
        # Server loops prompts and extends 'all_masks'. 
        # Since we use 1 box -> 1 mask logic in server loop, mask count should equal hypothesis count
        
        if len(all_masks) != len(hypotheses):
             # Handle mismatch (fallback or assignment)
             print(f"Warning: Sent {len(hypotheses)} prompts, got {len(all_masks)} masks")
             
        for i, (hyp, mask) in enumerate(zip(hypotheses, all_masks)):
            candidates.append({
                'id': f"H{i}",
                'mask': mask,
                'score_comparative': 0.0,
                'score_individual': 0.0,
                'rank': 999,
                'box': hyp['box'],
                'reasoning': hyp['reasoning'],
                'noun_phrase': hyp['noun_phrase'],
                'iou': calculate_iou(mask, gt_mask) if gt_mask is not None else 0.0
            })
            
        iou_str = "" # Placeholder

        
        if not candidates:
            print("WARNING: No candidates generated!")
            return None
        
        results = {
            'all_candidates': candidates,
            'image': image
        }
        
        masks_list = [c['mask'] for c in candidates]
        
        mask_similarities = self._compute_mask_similarities(masks_list)
        avg_similarity = np.mean(mask_similarities) if mask_similarities else 0
        print(f"  Mask similarity (avg pairwise IoU): {avg_similarity:.3f}")
        
        skip_vlm_verification = avg_similarity > 0.85
        if skip_vlm_verification:
            print(f"  ⚠ Masks are too similar (>{0.85:.0%} overlap) - using heuristic selection")
        
        if verification_mode in ["comparative", "both"]:
            print(f"--- Step 3a: Verification (Comparative) ---")
            
            if skip_vlm_verification:
                for i, cand in enumerate(candidates):
                    quality = self._compute_mask_quality(cand['mask'])
                    cand['score_comparative'] = quality
                    cand['rank'] = 0
                    cand['label'] = chr(65 + i)
                
                sorted_by_quality = sorted(candidates, key=lambda x: x['score_comparative'], reverse=True)
                for rank, cand in enumerate(sorted_by_quality):
                    cand['rank'] = rank + 1
                    iou_info = f" | IoU: {cand.get('iou', 0):.4f}" if gt_mask is not None else ""
                    print(f"  {cand['id']} (#{cand['rank']}): Quality {cand['score_comparative']:.1f}{iou_info}")
            else:
                comparison_results = self.verifier.verify_batch_comparative(image, masks_list, query)
                
                for result in comparison_results:
                    idx = result['mask_idx']
                    candidates[idx]['score_comparative'] = result['score']
                    candidates[idx]['rank'] = result['rank']
                    candidates[idx]['label'] = result['label']
                    
                    iou_info = f" | IoU: {candidates[idx].get('iou', 0):.4f}" if gt_mask is not None else ""
                    print(f"  {candidates[idx]['id']} (#{result['rank']}): Score {result['score']}{iou_info}")
            
            best_comparative = min(candidates, key=lambda x: x.get('rank', 999))
            results['best_mask_comparative'] = best_comparative['mask']
            results['best_candidate_comparative'] = best_comparative
            print(f"  -> Comparative Best: {best_comparative['id']} (Rank #{best_comparative.get('rank')})")
        
        if verification_mode in ["individual", "both"]:
            print(f"--- Step 3b: Verification (Individual) ---")
            
            if skip_vlm_verification:
                for cand in candidates:
                    quality = self._compute_mask_quality(cand['mask'])
                    cand['score_individual'] = quality
                    iou_info = f" | IoU: {cand.get('iou', 0):.4f}" if gt_mask is not None else ""
                    print(f"  {cand['id']}: Quality {cand['score_individual']:.1f}{iou_info} (heuristic)")
            else:
                for cand in candidates:
                    result = self.verifier.verify(image, cand['mask'], query)
                    cand['score_individual'] = result.get('total', result.get('score', 50))
                    
                    iou_info = f" | IoU: {cand.get('iou', 0):.4f}" if gt_mask is not None else ""
                    print(f"  {cand['id']}: Score {cand['score_individual']:.1f}{iou_info}")
            
            best_individual = max(candidates, key=lambda x: x.get('score_individual', 0))
            results['best_mask_individual'] = best_individual['mask']
            results['best_candidate_individual'] = best_individual
            print(f"  -> Individual Best: {best_individual['id']} (Score {best_individual['score_individual']:.1f})")
        
        if verification_mode == "comparative":
            results['best_mask'] = results['best_mask_comparative']
            results['best_candidate'] = results['best_candidate_comparative']
        elif verification_mode == "individual":
            results['best_mask'] = results['best_mask_individual']
            results['best_candidate'] = results['best_candidate_individual']
        
        if gt_mask is not None:
            best_iou_cand = max(candidates, key=lambda x: x.get('iou', 0))
            results['oracle_best'] = best_iou_cand
            print(f"  -> Oracle Best (GT): {best_iou_cand['id']} (IoU {best_iou_cand.get('iou', 0):.4f})")
        
        print(f"--- Result ---")
        if verification_mode == "both":
            print(f"Comparative: {results['best_candidate_comparative']['id']}")
            print(f"Individual:  {results['best_candidate_individual']['id']}")
        
        return results
