import re
import numpy as np
import torch
from .utils import get_device
from vllm import LLM, SamplingParams

except ImportError:
    LLM = None
    SamplingParams = None
except RuntimeError as e:
    if "Found no NVIDIA driver" in str(e) or "libcuda.so" in str(e):
        print("\n\nCRITICAL ERROR: ROCm/NVIDIA Mismatch Detected!")
        print("it appears you have the NVIDIA version of PyTorch/vLLM installed on an AMD system.")
        print("Please run 'pip uninstall torch torchvision torchaudio vllm' and reinstall the ROCm versions.")
        print("See README.md for details.\n")
        raise e
    else:
        raise e

class Planner:
    def __init__(self, model_path="Qwen/Qwen2.5-VL-72B-Instruct", device=None, dtype="auto", quantization=None):
        self.device = device or get_device()
        self.model_path = model_path
        
        if LLM is not None:
            print(f"Loading Planner (vLLM): {model_path} [dtype={dtype}, quantization={quantization}]")
            self.llm = LLM(model=model_path, trust_remote_code=True, tensor_parallel_size=1, dtype=dtype, quantization=quantization) 
        else:
            raise ImportError("vLLM is not installed. Please install vllm to use the Planner.")

    def generate_hypotheses(self, image_path, query, N=1, temperature=0.7):
        prompt_text = (
            f"Query: {query}\n"
            "Step 1 (See): List all objects visible in the image that are relevant to the query.\n"
            "Step 2 (Think): Analyze the spatial and causal relationships. Resolve ambiguities.\n"
            "Step 3 (Propose): Output the bounding box coordinates [x_1, y_1, x_2, y_2] for the target object.\n"
            "Format your answer as:\n"
            "Reasoning: <your reasoning>\n"
            "Target Concept: <noun phrase>\n"
            "Box: [x1, y1, x2, y2]"
        )
        
        inputs = [
            {
                "prompt": f"<|image_pad|>{prompt_text}",
                "multi_modal_data": {"image": image_path},
            }
            for _ in range(N)
        ]
        
        sampling_params = SamplingParams(temperature=temperature, n=N, max_tokens=512)
        
        outputs = self.llm.generate(
            [inputs[0]["prompt"]],
            sampling_params=sampling_params,
        )
        
        hypotheses = []
        
        from PIL import Image
        with Image.open(image_path) as img:
            real_w, real_h = img.size

        for output in outputs[0].outputs:
            text = output.text
            box_match = re.search(r"Box:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
            concept_match = re.search(r"Target Concept:\s*(.*)", text)
            reasoning_match = re.search(r"Reasoning:\s*(.*)", text, re.DOTALL)
            
            if box_match:
                coords = [int(c) for c in box_match.groups()]
                x1 = int(coords[0] / 1000 * real_w)
                y1 = int(coords[1] / 1000 * real_h)
                x2 = int(coords[2] / 1000 * real_w)
                y2 = int(coords[3] / 1000 * real_h)
                
                hypotheses.append({
                    "box": [x1, y1, x2, y2],
                    "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
                    "noun_phrase": concept_match.group(1).strip() if concept_match else "object",
                    "raw_text": text
                })
            else:
                print(f"Warning: Could not parse box from output: {text[:50]}...")
                
        return hypotheses
