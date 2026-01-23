import re
import numpy as np
import os
import tempfile
from .utils import apply_red_alpha_overlay
from .api_utils import create_vision_message, get_openai_client

class Verifier:
    def __init__(self, client=None, model_path="Qwen/Qwen3-VL-30B-A3B-Instruct", api_base="http://localhost:8000/v1"):
        self.model_path = model_path
        if client:
            self.client = client
        else:
            self.client = get_openai_client(base_url=api_base)

    def verify(self, image_input, mask, query):
        overlay_img = apply_red_alpha_overlay(image_input, mask, alpha=0.5)
        
        prompt_text = (
            f"Query: {query}\n"
            "Focus on the red-highlighted region. "
            f"Does this region accurately and precisely correspond to the description '{query}'? "
            "Check for spatial correctness, object identity, and boundary precision. "
            "Output a score from 0 to 100."
            "Format: Score: <number>"
        )
        
        if self.client:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                overlay_img.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                messages = create_vision_message(prompt_text, tmp_path)
                
                completion = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=1024
                )
                
                text = completion.choices[0].message.content
                print(f"[Verifier Raw Output]: {text}") 
                
                match = re.search(r"(?:Score|score)[:\s]*is\s*(\d+(?:\.\d+)?)|(?:Score|score)[:\s]*(\d+(?:\.\d+)?)", text)
                if match:
                    # group 1 or group 2 will be non-None
                    score_str = match.group(1) if match.group(1) else match.group(2)
                    score = float(score_str)
                else:
                    # Fallback: look for just a number at the end
                    match_end = re.search(r"\b(\d{1,3})\b$", text.strip())
                    if match_end:
                         score = float(match_end.group(1))
                    else:
                        print(f"Verifier Parsing Failed for: '{text}'")
                        score = 0.0
            except Exception as e:
                print(f"Verifier Error: {e}")
                score = 0.0
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            return score
        else:
            return 0.0
