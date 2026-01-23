from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .utils import load_image, plot_results, get_device
import numpy as np

class RESCuePipeline:
    def __init__(self, 
                 planner_model="Qwen/Qwen3-VL-30B-A3B-Instruct",
                 executor_model="facebook/sam3",
                 planner_api_base="http://localhost:8000/v1",
                 executor_api_base="http://localhost:8001",
                 device=None,
                 dtype="auto",
                 quantization=None,
                 verification_mode="comparative"):
        """
        Initialize RESCue Pipeline.
        
        Args:
            verification_mode: "simple", "detailed", or "comparative"
                - "simple": Basic scoring (fast, less accurate differentiation)
                - "detailed": Multi-criteria scoring (more LLM calls, better breakdown)
                - "comparative": Compare all masks at once (best differentiation, recommended)
        """
        print("Initializing RESCue Pipeline...")
        self.device = device or get_device()
        self.verification_mode = verification_mode
        print(f"Pipeline using device: {self.device}")
        print(f"Verification mode: {verification_mode}")
        
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
            model_path=planner_model,
            api_base=planner_api_base,
            mode="detailed" if verification_mode == "detailed" else "simple"
        )
        
        print("Pipeline Initialized.")

    def run(self, image_path, query, N=4, gt_mask=None):
        image = load_image(image_path)
        
        print(f"--- Step 1: Planning (N={N}) ---")
        hypotheses = self.planner.generate_hypotheses(image_path, query, N=N)
        print(f"Generated {len(hypotheses)} hypotheses.")
        
        candidates = []
        all_masks = []
        
        print(f"--- Step 2: Execution ---")
        
        # Optimization: Send image to Executor ONCE
        print("  - Encoding Image on Executor...")
        self.executor.encode_image(np.array(image))
        
        for i, hyp in enumerate(hypotheses):
            box = hyp['box']
            noun_phrase = hyp['noun_phrase']
            reasoning = hyp['reasoning']
            
            # Use cached image, just predict prompt
            masks = self.executor.predict_masks(box, noun_phrase)
            
            for j, mask in enumerate(masks):
                all_masks.append(mask)
                candidates.append({
                    'id': f"H{i}_M{j}",
                    'mask': mask,
                    'score': 0.0,  # Will be filled in verification
                    'box': box,
                    'reasoning': reasoning,
                    'noun_phrase': noun_phrase
                })
                print(f"  Generated candidate H{i}_M{j} | {noun_phrase}")
        
        if not candidates:
            return None
        
        print(f"--- Step 3: Verification ({self.verification_mode} mode) ---")
        
        if self.verification_mode == "comparative" and len(candidates) > 1:
            # Comparative verification: score all masks together
            print(f"  - Comparing {len(all_masks)} candidates simultaneously...")
            scores = self.verifier.verify_comparative(image, all_masks, query)
            
            for i, cand in enumerate(candidates):
                cand['score'] = scores[i] if i < len(scores) else 0.0
                
                # IoU for debugging
                iou_info = ""
                if gt_mask is not None:
                    from .utils import calculate_iou
                    iou = calculate_iou(cand['mask'], gt_mask)
                    iou_info = f" | IoU: {iou:.4f}"
                
                print(f"  {cand['id']}: Score {cand['score']:.1f}{iou_info}")
        
        else:
            # Individual verification (simple or detailed mode)
            self.verifier.start_batch()
            
            for cand in candidates:
                result = self.verifier.verify(image, cand['mask'], query, 
                                             return_details=(self.verification_mode == "detailed"))
                
                if isinstance(result, dict):
                    cand['score'] = result['total']
                    cand['score_breakdown'] = result
                else:
                    cand['score'] = result
                
                # IoU for debugging
                iou_info = ""
                if gt_mask is not None:
                    from .utils import calculate_iou
                    iou = calculate_iou(cand['mask'], gt_mask)
                    iou_info = f" | IoU: {iou:.4f}"
                
                breakdown = ""
                if 'score_breakdown' in cand:
                    sb = cand['score_breakdown']
                    breakdown = f" [I:{sb['identity']:.0f} S:{sb['spatial']:.0f} C:{sb['completeness']:.0f} B:{sb['boundary']:.0f}]"
                
                print(f"  {cand['id']}: Score {cand['score']:.1f}{breakdown}{iou_info}")
            
            # Normalize scores within batch for better differentiation
            normalized_scores = self.verifier.end_batch()
            if normalized_scores and len(normalized_scores) == len(candidates):
                print("  - Applying score normalization for better spread...")
                for i, cand in enumerate(candidates):
                    cand['raw_score'] = cand['score']
                    cand['score'] = normalized_scores[i]
                    print(f"    {cand['id']}: {cand['raw_score']:.1f} -> {cand['score']:.1f}")
        
        best_candidate = max(candidates, key=lambda x: x['score'])
        print(f"--- Result ---")
        print(f"Best Candidate: {best_candidate['id']} with Score {best_candidate['score']:.1f}")
        
        return {
            'best_mask': best_candidate['mask'],
            'best_score': best_candidate['score'],
            'all_candidates': candidates,
            'image': image
        }
