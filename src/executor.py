import torch
import numpy as np
import cv2
import requests
import base64
import io
from PIL import Image
from .utils import get_device
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class Executor:
    def __init__(self, model_path="facebook/sam3", device=None, remote_url=None):
        self.remote_url = remote_url
        self.device = device or get_device()
        
        if self.remote_url:
            self.model = None
            self.processor = None
        else:
            checkpoint_path = None if model_path == "facebook/sam3" else model_path
            self.model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=checkpoint_path
            )
            self.processor = Sam3Processor(self.model)

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
        return np.array(img).astype(bool)

    def execute(self, image_input, box, text_prompt):
        self.encode_image(image_input)
        return self.predict_masks(box, text_prompt)

    def encode_image(self, image_input):
        if self.remote_url:
            try:
                payload = {"image_base64": self._image_to_base64(image_input)}
                requests.post(f"{self.remote_url}/set_image", json=payload).raise_for_status()
                return True
            except Exception:
                return False
        
        self.active_state = None
        try:
            self.active_state = self.processor.set_image(image_input)
            if hasattr(image_input, 'shape'):
                self.img_h, self.img_w = image_input.shape[:2]
            else:
                self.img_w, self.img_h = image_input.size
        except Exception:
            self.active_state = None

    def predict_masks(self, box, text_prompt):
        if self.remote_url:
            try:
                payload = {
                    "box": list(map(float, box)) if box else None,
                    "text_prompt": text_prompt
                }
                response = requests.post(f"{self.remote_url}/predict", json=payload)
                response.raise_for_status()
                data = response.json()
                return [self._base64_to_mask(m) for m in data.get("masks_base64", [])]
            except Exception:
                return []

        if self.active_state is None:
            return []

        current_state = self.active_state.copy()
        
        if text_prompt:
            current_state = self.processor.set_text_prompt(
                prompt=text_prompt, 
                state=current_state
            )
            
        if box is not None:
            current_state = self.processor.add_geometric_prompt(
                box=box, 
                label=1, 
                state=current_state
            )

        masks = current_state.get("masks", None)
        if masks is None or masks.numel() == 0:
            return []
            
        masks_np = masks.cpu().numpy()
        return self._process_mask_output(masks_np, self.img_h, self.img_w)

    def _process_mask_output(self, masks_np, target_h, target_w):
        extracted_masks = []
        if masks_np.ndim == 4:
            B, D1, D2, D3 = masks_np.shape
            if D3 <= 10: 
                masks_np = masks_np.transpose(0, 3, 1, 2)
            for b in range(masks_np.shape[0]):
                for c in range(masks_np.shape[1]):
                    extracted_masks.append(masks_np[b, c])
        elif masks_np.ndim == 3:
            if masks_np.shape[2] <= 10 and masks_np.shape[0] > 100:
                    masks_np = masks_np.transpose(2, 0, 1)
            for i in range(masks_np.shape[0]):
                extracted_masks.append(masks_np[i])
        else:
             for i in range(masks_np.shape[0]):
                extracted_masks.append(masks_np[i])

        final_masks = []
        for m in extracted_masks:
            m = m > 0
            if m.shape != (target_h, target_w):
                if m.shape == (target_w, target_h):
                    m = m.T
                else:
                    m = cv2.resize(m.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
            final_masks.append(m)
        
        return final_masks
