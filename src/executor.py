import torch
import numpy as np
from .utils import get_device

try:
    import sam3
    from sam3 import SAM3Predictor
except ImportError:
    print("Warning: sam3 not found. Executor will fail unless sam3 is installed.")
    SAM3Predictor = None

class Executor:
    def __init__(self, model_path="facebook/sam3", device=None):
        self.device = device or get_device()
        if SAM3Predictor is None:
            raise ImportError("SAM 3 is not installed.")
            
        print(f"Loading Executor (SAM 3): {model_path}")
        self.predictor = SAM3Predictor.from_pretrained(model_path)
        self.predictor.to(self.device)

    def execute(self, image_input, box, text_prompt):
        self.predictor.set_image(image_input)
        
        box_np = np.array(box)[None, :]
        
        masks, scores, logits = self.predictor.predict(
            box=box_np,
            text_prompt=text_prompt,
            multimask_output=True
        )
        
        return [masks[i] for i in range(masks.shape[0])]
