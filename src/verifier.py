import re
import json
import json_repair
import numpy as np
import os
import tempfile
from .utils import apply_red_alpha_overlay
from .api_utils import create_vision_message, get_openai_client


class Verifier:
    def __init__(self, client=None, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", 
                 api_base="http://localhost:8000/v1"):
        self.model_path = model_path
        self.client = client if client else get_openai_client(base_url=api_base)

    def verify(self, image_input, mask, query):
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.4)
        
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        mask_coverage = mask_np.sum() / mask_np.size * 100
        
        prompt_text = f"""Evaluate segmentation mask for: "{query}"

        RED region = predicted segmentation ({mask_coverage:.1f}% coverage).

        Score 0-25 each (use ANY integer, not just multiples of 5):

        1. identity: Is the RED region the correct object type described in the query?
        - 23-25: Exact object match
        - 18-22: Correct category, minor ambiguity
        - 8-17: Related but not exact (e.g., "chair" vs "stool")
        - 0-7: Wrong object entirely

        2. spatial: Is the object in the described/expected location?
        - 23-25: Perfectly positioned as described
        - 18-22: Correct general region
        - 8-17: Partially displaced
        - 0-7: Wrong location

        3. completeness: What % of the target object is covered by the mask?
        - Score ≈ (estimated % of object covered) x 0.25
        - Example: 72% covered → score ~18

        4. boundary: How accurate are the mask edges?
        - Score ≈ 25 - (estimated % boundary error x 0.5)
        - Example: 20% over/under-segmentation → score ~15

        Output ONLY JSON: {{"identity": N, "spatial": N, "completeness": N, "boundary": N}}"""

        scores = {"identity": 0, "spatial": 0, "completeness": 0, "boundary": 0, "total": 0}
        
        if not self.client:
            return scores
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            overlay_img.save(tmp.name, quality=95)
            tmp_path = tmp.name
        
        try:
            messages = create_vision_message(prompt_text, tmp_path)
            
            completion = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=0.1,
                max_tokens=100
            )
            
            text = completion.choices[0].message.content.strip()
            print(f"[Verifier]: {text}")
            
            parsed = json_repair.loads(text)
            for key in ["identity", "spatial", "completeness", "boundary"]:
                scores[key] = min(25, max(0, float(parsed.get(key, 0))))
            
            scores["total"] = sum(scores[k] for k in ["identity", "spatial", "completeness", "boundary"])
            
        except Exception as e:
            print(f"Verifier Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return scores
