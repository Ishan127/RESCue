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
            model_path=planner_model,
            api_base=planner_api_base
        )
        
        print("Pipeline Initialized.")

    def run(self, image_path, query, N=4, gt_mask=None):
        image = load_image(image_path)
        
        print(f"--- Step 1: Planning (N={N}) ---")
        hypotheses = self.planner.generate_hypotheses(image_path, query, N=N)
        print(f"Generated {len(hypotheses)} hypotheses.")
        
        candidates = []
        
        print(f"--- Step 2: Execution & Verification ---")
        
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
                score = self.verifier.verify(image, mask, query)
                
                # IoU Calculation for debugging if Ground Truth is provided
                iou_info = ""
                if gt_mask is not None:
                    from .utils import calculate_iou
                    iou = calculate_iou(mask, gt_mask)
                    iou_info = f" | IoU: {iou:.4f}"
                
                candidates.append({
                    'id': f"H{i}_M{j}",
                    'mask': mask,
                    'score': score,
                    'box': box,
                    'reasoning': reasoning,
                    'noun_phrase': noun_phrase
                })
                print(f"  Candidate H{i}_M{j}: Score {score}{iou_info} | {noun_phrase}")
                
        if not candidates:
            return None
            
        best_candidate = max(candidates, key=lambda x: x['score'])
        print(f"--- Result ---")
        print(f"Best Candidate: {best_candidate['id']} with Score {best_candidate['score']}")
        
        return {
            'best_mask': best_candidate['mask'],
            'best_score': best_candidate['score'],
            'all_candidates': candidates,
            'image': image
        }
