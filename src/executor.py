import torch
import numpy as np
import requests
import base64
import io
from PIL import Image
from .utils import get_device
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class Executor:
    def __init__(self, model_path="facebook/sam3", device=None, remote_url=None):
        self.remote_url = remote_url
        self.device = device or get_device()
        
        if self.remote_url:
            print(f"Executor initialized in REMOTE mode. Target: {self.remote_url}")
            self.model = None
            self.processor = None
        else:
            print(f"Loading Executor (SAM 3) locally...")
            
            # model_path arg is kept for compatibility but "facebook/sam3" is the HF Hub ID
            # If user provides a local path, we assume it's a checkpoint
            checkpoint_path = None
            if model_path != "facebook/sam3":
                checkpoint_path = model_path # Fixed bug: was assuming model_path is checkpoint even if passed explicitly
            
            # Fallback for sam3 loading
            try:
                # Try loading assuming 'facebook/sam3' or path
                self.model = build_sam3_image_model(
                    device=self.device,
                    checkpoint_path=checkpoint_path
                )
            except TypeError as te:
                 # Some versions of sam3 build function might differ
                 print(f"Retrying SAM3 load with different signature: {te}")
                 try:
                    self.model = build_sam3_image_model(
                        device=self.device,
                        ckpt_path=checkpoint_path
                    )
                 except Exception as e2:
                     print(f"Failed to load SAM3 model: {e2}")
                     raise e2
            except Exception as e:
                print(f"Error loading SAM3 model: {e}")
                raise e

            self.processor = Sam3Processor(self.model, device=self.device)

    def _image_to_base64(self, image_input):
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype(np.uint8))
        else:
            image = image_input
            
        buff = io.BytesIO()
        image.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    def _base64_to_mask(self, b64_string):
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data))
        # SAM masks are usually boolean or 0/1. 
        # Server sends 0-255 image. 
        return np.array(img) > 128

    def execute(self, image_input, box, text_prompt):
        # image_input: np array (H, W, 3) or PIL Image
        # box: [x1, y1, x2, y2]
        # text_prompt: str
        
        if self.remote_url:
            try:
                payload = {
                    "image_base64": self._image_to_base64(image_input),
                    "box": list(map(float, box)) if box else None,
                    "text_prompt": text_prompt
                }
                response = requests.post(f"{self.remote_url}/segment", json=payload)
                response.raise_for_status()
                data = response.json()
                
                masks = []
                for b64_mask in data.get("masks_base64", []):
                    masks.append(self._base64_to_mask(b64_mask))
                return masks
            except Exception as e:
                print(f"Remote Execution Failed: {e}")
                return []
        
        # Local Execution
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
