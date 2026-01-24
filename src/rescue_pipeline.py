from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .utils import load_image, get_device, calculate_iou
import numpy as np

class RESCuePipeline:
    def __init__(self, 
                 planner_model="Qwen/Qwen3-VL-32B-Thinking",
                 verifier_model="Qwen/Qwen3-VL-32B-Thinking",
                 executor_model="facebook/sam3",
                 planner_api_base="http://localhost:8000/v1",
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
        
        self.verifier = Verifier(
            client=self.planner.client,
            model_path=verifier_model,
            api_base=planner_api_base
        )
        
        print(f"Planner: {planner_model}")
        print(f"Verifier: {verifier_model}")
        print("Pipeline Initialized.")

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
        print("  - Encoding Image on Executor...")
        self.executor.encode_image(np.array(image))
        
        for i, hyp in enumerate(hypotheses):
            box = hyp['box']
            noun_phrase = hyp['noun_phrase']
            reasoning = hyp['reasoning']
            
            masks = self.executor.predict_masks(box, noun_phrase)
            
            for j, mask in enumerate(masks):
                cand = {
                    'id': f"H{i}_M{j}",
                    'mask': mask,
                    'score_comparative': 0.0,
                    'score_individual': 0.0,
                    'rank': 999,
                    'box': box,
                    'reasoning': reasoning,
                    'noun_phrase': noun_phrase
                }
                if gt_mask is not None:
                    cand['iou'] = calculate_iou(mask, gt_mask)
                candidates.append(cand)
                iou_str = f" | IoU: {cand.get('iou', 0):.4f}" if gt_mask is not None else ""
                print(f"  Generated candidate H{i}_M{j}{iou_str} | {noun_phrase}")
        
        if not candidates:
            print("WARNING: No candidates generated!")
            return None
        
        results = {
            'all_candidates': candidates,
            'image': image
        }
        
        masks_list = [c['mask'] for c in candidates]
        
        if verification_mode in ["comparative", "both"]:
            print(f"--- Step 3a: Verification (Comparative) ---")
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
