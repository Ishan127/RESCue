from .planner import Planner
from .executor import Executor
from .verifier import Verifier
from .utils import load_image, plot_results
import numpy as np

class RESCuePipeline:
    def __init__(self, 
                 planner_model="Qwen/Qwen2.5-VL-72B-Instruct",
                 executor_model="facebook/sam3",
                 device="cuda"):
        
        print("Initializing RESCue Pipeline...")
        
        self.planner = Planner(model_path=planner_model, device=device)
        
        self.executor = Executor(model_path=executor_model, device=device)
        
        self.verifier = Verifier(llm_instance=self.planner.llm)
        
        print("Pipeline Initialized.")

    def run(self, image_path, query, N=4):
        image = load_image(image_path)
        
        print(f"--- Step 1: Planning (N={N}) ---")
        hypotheses = self.planner.generate_hypotheses(image_path, query, N=N)
        print(f"Generated {len(hypotheses)} hypotheses.")
        
        candidates = []
        
        print(f"--- Step 2: Execution & Verification ---")
        for i, hyp in enumerate(hypotheses):
            box = hyp['box']
            noun_phrase = hyp['noun_phrase']
            reasoning = hyp['reasoning']
            
            masks = self.executor.execute(np.array(image), box, noun_phrase)
            
            for j, mask in enumerate(masks):
                score = self.verifier.verify(image, mask, query)
                
                candidates.append({
                    'id': f"H{i}_M{j}",
                    'mask': mask,
                    'score': score,
                    'box': box,
                    'reasoning': reasoning,
                    'noun_phrase': noun_phrase
                })
                print(f"  Candidate H{i}_M{j}: Score {score} | {noun_phrase}")
                
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
