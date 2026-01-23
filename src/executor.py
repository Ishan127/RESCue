import torch
import torchvision
import numpy as np
import cv2
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
        except (RuntimeError, NotImplementedError) as e:
            if "backend" in str(e) or "not available" in str(e):
                # Fallback to CPU
                input_cpu = input.cpu()
                
                # Handle boxes: can be Tensor or List[Tensor]
                if isinstance(boxes, torch.Tensor):
                    boxes_cpu = boxes.cpu()
                elif isinstance(boxes, (list, tuple)):
                    boxes_cpu = [b.cpu() for b in boxes]
                else:
                    boxes_cpu = boxes # Try as is if unknown type
                    
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
        img = Image.open(io.BytesIO(img_data)).convert("L")
        return np.array(img).astype(bool) # > 128 check done by implicit bool cast or threshold

    def execute(self, image_input, box, text_prompt):
        # Legacy/Convenience method: Encode + Predict
        self.encode_image(image_input)
        return self.predict_masks(box, text_prompt)

    def encode_image(self, image_input):
        """
        encodes the image and stores the state internally.
        """
        if self.remote_url:
            try:
                # Send image to server to be cached
                payload = {"image_base64": self._image_to_base64(image_input)}
                response = requests.post(f"{self.remote_url}/set_image", json=payload)
                response.raise_for_status()
                return True
            except Exception as e:
                print(f"Remote Set Image Failed: {e}")
                return False
        
        # Local Setup
        # Reset state
        self.active_state = None
        
        try:
            # This computes the embedding (expensive)
            self.active_state = self.processor.set_image(image_input)
            
            # Store image dimensions for later normalization
            if hasattr(image_input, 'shape'):
                self.img_h, self.img_w = image_input.shape[:2]
            else:
                self.img_w, self.img_h = image_input.size
                
        except Exception as e:
            print(f"Local Image Encoding Failed: {e}")
            self.active_state = None

    def predict_masks(self, box, text_prompt):
        """
        Uses the cached image state to generate masks for a prompt.
        """
        if self.remote_url:
            try:
                # Server already has image. Just send prompt.
                payload = {
                    "box": list(map(float, box)) if box else None,
                    "text_prompt": text_prompt
                }
                response = requests.post(f"{self.remote_url}/predict", json=payload)
                response.raise_for_status()
                data = response.json()
                
                masks = []
                for b64_mask in data.get("masks_base64", []):
                    masks.append(self._base64_to_mask(b64_mask))
                return masks
            except Exception as e:
                print(f"Remote Prediction Failed: {e}")
                if hasattr(e, 'response') and e.response:
                    print(f"Server Response: {e.response.text}")
                return []

        # Local Execution
        if self.active_state is None:
            print("Error: No image set. Call encode_image() first.")
            return []

        # We must clone the state or use a method that doesn't mutate it permanently if we want to reuse it?
        # Sam3Processor methods usually return a NEW state dict merged with old one.
        # But to be safe, we treat self.active_state as immutable base.
        current_state = self.active_state.copy()
        
        # 1. Apply Text Prompt
        if text_prompt:
            current_state = self.processor.set_text_prompt(text_prompt, current_state)
            
        # 2. Apply Box Prompt
        if box is not None:
             # Convert [x1, y1, x2, y2] to [cx, cy, w, h] normalized
             # We use stored dimensions or dimensions from state
            h = current_state.get("original_height", self.img_h)
            w = current_state.get("original_width", self.img_w)
            
            x1, y1, x2, y2 = box
            
            # Center coordinates normalized
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            norm_box = [cx, cy, bw, bh]
            
            current_state = self.processor.add_geometric_prompt(
                box=norm_box, 
                label=True, 
                state=current_state
            )

        # 3. Retrieve Output
        masks = current_state.get("masks", None)
        
        if masks is not None:
            if masks.numel() == 0: return []
            
            masks_np = masks.cpu().numpy()
            print(f"[Executor Debug] Raw masks shape: {masks_np.shape}")
            
            # Reuse the robust extraction logic
            return self._process_mask_output(masks_np, self.img_h, self.img_w)
            
        return []

    def _process_mask_output(self, masks_np, target_h, target_w):
        extracted_masks = []
            
        # 4D Tensor Handling
        if masks_np.ndim == 4:
            B, D1, D2, D3 = masks_np.shape
            # If D3 is small (N masks) and others are large or 1
            if D3 <= 10: 
                # channels-last format (B, H, W, N)
                masks_np = masks_np.transpose(0, 3, 1, 2)
                
            # Flatten (B, N, H, W) -> List of (H, W)
            for b in range(masks_np.shape[0]):
                for c in range(masks_np.shape[1]):
                    extracted_masks.append(masks_np[b, c])
        
        # 3D Tensor Handling (N, H, W) or (H, W, N)
        elif masks_np.ndim == 3:
             # Is it (H, W, N)?
            if masks_np.shape[2] <= 10 and masks_np.shape[0] > 100:
                    masks_np = masks_np.transpose(2, 0, 1)

            for i in range(masks_np.shape[0]):
                extracted_masks.append(masks_np[i])
        else:
             # Fallback
             for i in range(masks_np.shape[0]):
                extracted_masks.append(masks_np[i])

        # Enforce Shape
        final_masks = []
        for m in extracted_masks:
            if m.shape != (target_h, target_w):
                # Tolerance for transposed
                if m.shape == (target_w, target_h):
                        m = m.T
                else:
                    m = cv2.resize(m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
            final_masks.append(m)
        
        return final_masks
