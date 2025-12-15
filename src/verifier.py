import re
import numpy as np
from .utils import apply_red_alpha_overlay

class Verifier:
    def __init__(self, llm_instance=None, model_path="Qwen/Qwen2.5-VL-72B-Instruct"):
        if llm_instance:
            self.llm = llm_instance
        else:
            try:
                from vllm import LLM
                self.llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1)
            except ImportError:
                print("Verifier: vLLM not available.")
                self.llm = None

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
        
        if self.llm:
            from vllm import SamplingParams
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                overlay_img.save(tmp.name)
                tmp_path = tmp.name
            
            inputs = [{
                "prompt": f"<|image_pad|>{prompt_text}",
                "multi_modal_data": {"image": tmp_path},
            }]
            
            sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
            
            outputs = self.llm.generate(
                [inputs[0]["prompt"]],
                sampling_params=sampling_params
            )
            
            os.remove(tmp_path)
            
            text = outputs[0].outputs[0].text
            
            match = re.search(r"Score:\s*(\d+)", text)
            if match:
                return float(match.group(1))
            else:
                return 0.0
        else:
            return 0.0
