import torch
import torchvision
import numpy as np
import requests
import base64
import io
from PIL import Image
from .utils import get_device
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Monkey-Patch torchvision for ROCm compatibility ---
# ROI Align often fails on ROCm if torchvision is not built with HIP support.
# We intercept the call and force it to CPU for that specific operation.
_original_roi_align = torchvision.ops.roi_align

def _roi_align_rocm_safe(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
    if input.is_cuda:
        try:
            return _original_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)
        except RuntimeError as e:
            if "backend" in str(e) or "not available" in str(e):
                # Fallback to CPU
                input_cpu = input.cpu()
                boxes_cpu = boxes.cpu()
                result = _original_roi_align(input_cpu, boxes_cpu, output_size, spatial_scale, sampling_ratio, aligned)
                return result.to(input.device)
            raise e
    return _original_roi_align(input, boxes, output_size, spatial_scale, sampling_ratio, aligned)

torchvision.ops.roi_align = _roi_align_rocm_safe
# -------------------------------------------------------

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
                checkpoint_path = model_path
            
            try:
                self.model = build_sam3_image_model(
                    device=self.device,
                    checkpoint_path=checkpoint_path
                )
            except TypeError:
                 # Checkpoint signature mismatch fallback
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
                try:
                    response.raise_for_status()
                    data = response.json()
                except requests.exceptions.HTTPError as e:
                    print(f"Remote Execution Failed: {e}")
                    print(f"Server Response: {response.text}")
                    return []
                
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
