import torch
import numpy as np
from .utils import get_device
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class Executor:
    def __init__(self, model_path="facebook/sam3", device=None):
        self.device = device or get_device()
        print(f"Loading Executor (SAM 3)...")
        
        # model_path arg is kept for compatibility but "facebook/sam3" is the HF Hub ID
        # If user provides a local path, we assume it's a checkpoint
        checkpoint_path = None
        if model_path != "facebook/sam3":
            checkpoint_path = model_path
            
        try:
            self.model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=checkpoint_path
            )
        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            raise e

        self.processor = Sam3Processor(self.model, device=self.device)

    def execute(self, image_input, box, text_prompt):
        # image_input: np array (H, W, 3) or PIL Image
        # box: [x1, y1, x2, y2]
        # text_prompt: str
        
        state = self.processor.set_image(image_input)
        
        if text_prompt:
            state = self.processor.set_text_prompt(text_prompt, state)
            
        if box is not None:
            # Convert [x1, y1, x2, y2] to [cx, cy, w, h] normalized
            if "original_height" in state and "original_width" in state:
                h = state["original_height"]
                w = state["original_width"]
                
                x1, y1, x2, y2 = box
                
                # Center coordinates normalized
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                
                # Width and height normalized
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                
                norm_box = [cx, cy, bw, bh]
                
                # Add box prompt (label=True means foreground/positive)
                state = self.processor.add_geometric_prompt(
                    box=norm_box, 
                    label=True, 
                    state=state
                )
            else:
                print("Warning: Image state missing dimensions, skipping box prompt.")

        # Processor returns masks as tensor (N, H, W)
        masks = state.get("masks", None)
        
        if masks is not None:
            # Check if masks is empty tensor
            if masks.numel() == 0:
                return []
                
            # Convert to numpy list of masks
            masks_np = masks.cpu().numpy()
            return [masks_np[i] for i in range(masks_np.shape[0])]
            
        return []
