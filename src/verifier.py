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
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.5)
        
        mask_np = np.array(mask).astype(bool)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        
        mask_coverage = mask_np.sum() / mask_np.size * 100
        
        prompt_text = f"""Look at this image with a RED highlighted region. The task was to segment: "{query}"

        Answer these questions about the RED region:

        Q1: Does the RED region highlight the CORRECT object described in the query? 
        - "yes" = RED covers the right object
        - "partial" = RED covers something related but not exact
        - "no" = RED covers the wrong thing entirely

        Q2: Is the RED region ONLY on the target object, or does it spill onto other things?
        - "precise" = RED stays within the object boundaries
        - "overspill" = RED extends beyond the object onto background/other objects
        - "underspill" = RED misses significant parts of the object
        - "both" = RED both misses parts AND extends beyond

        Q3: Roughly what percentage of the target object is covered by RED? Give a number 0-100.

        Output ONLY this JSON: {{"correct": "yes/partial/no", "precision": "precise/overspill/underspill/both", "coverage_pct": N}}"""

        scores = {"correct": 0, "precision": 0, "coverage": 0, "total": 0}
        raw_response = {}
        
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
            raw_response = parsed
            
            correct_val = str(parsed.get("correct", "no")).lower().strip()
            if correct_val == "yes":
                scores["correct"] = 40
            elif correct_val == "partial":
                scores["correct"] = 20
            else:
                scores["correct"] = 0
            
            precision_val = str(parsed.get("precision", "both")).lower().strip()
            if precision_val == "precise":
                scores["precision"] = 30
            elif precision_val in ["overspill", "underspill"]:
                scores["precision"] = 15
            else:
                scores["precision"] = 0
            
            coverage_pct = float(parsed.get("coverage_pct", 0))
            coverage_pct = min(100, max(0, coverage_pct))
            scores["coverage"] = int(coverage_pct * 0.30)
            
            scores["total"] = scores["correct"] + scores["precision"] + scores["coverage"]
            scores["raw"] = raw_response
            
        except Exception as e:
            print(f"Verifier Error: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return scores
