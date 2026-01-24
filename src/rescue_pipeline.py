from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .utils import load_image, get_device
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
        
        print(f"--- Step 2: Execution ---")
        print("  - Encoding Image on Executor...")
        self.executor.encode_image(np.array(image))
        
        for i, hyp in enumerate(hypotheses):
            box = hyp['box']
            noun_phrase = hyp['noun_phrase']
            reasoning = hyp['reasoning']
            
            masks = self.executor.predict_masks(box, noun_phrase)
            
            for j, mask in enumerate(masks):
                candidates.append({
                    'id': f"H{i}_M{j}",
                    'mask': mask,
                    'score': 0.0,
                    'box': box,
                    'reasoning': reasoning,
                    'noun_phrase': noun_phrase
                })
                print(f"  Generated candidate H{i}_M{j} | {noun_phrase}")
        
        if not candidates:
            print("WARNING: No candidates generated!")
            return None
        
        print(f"--- Step 3: Verification ---")
        
        for cand in candidates:
            result = self.verifier.verify(image, cand['mask'], query)
            cand['score'] = result['total']
            cand['score_breakdown'] = result
            
            iou_info = ""
            if gt_mask is not None:
                from .utils import calculate_iou
                iou = calculate_iou(cand['mask'], gt_mask)
                cand['iou'] = iou
                iou_info = f" | IoU: {iou:.4f}"
            
            sb = result
            print(f"  {cand['id']}: Score {cand['score']:.0f} [I:{sb['identity']:.0f} S:{sb['spatial']:.0f} C:{sb['completeness']:.0f} B:{sb['boundary']:.0f}]{iou_info} | {cand['noun_phrase']}")
        
        best_candidate = max(candidates, key=lambda x: x['score'])
        
        if gt_mask is not None:
            best_iou_cand = max(candidates, key=lambda x: x.get('iou', 0))
            if best_iou_cand['id'] != best_candidate['id']:
                print(f"  NOTE: Best IoU is {best_iou_cand['id']} ({best_iou_cand.get('iou', 0):.4f}), but selected {best_candidate['id']}")
        
        print(f"--- Result ---")
        print(f"Best Candidate: {best_candidate['id']} with Score {best_candidate['score']:.0f}")
        
        return {
            'best_mask': best_candidate['mask'],
            'best_score': best_candidate['score'],
            'all_candidates': candidates,
            'image': image
        }
