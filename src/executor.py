import torch
import numpy as np
import cv2
import requests
from requests.adapters import HTTPAdapter
import base64
import io
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from PIL import Image
from .utils import get_device

import logging
logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self, 
        model_path: str = "facebook/sam3", 
        device: Optional[str] = None, 
        remote_url: Optional[str] = None,
        confidence_threshold: float = 0.5, 
        resolution: int = 1008,
        timeout: int = 600
    ):
        self.remote_url = remote_url
        self.device = device or get_device()
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.timeout = timeout
        
        self._cached_image: Optional[Image.Image] = None
        self._cached_image_size: Optional[Tuple[int, int]] = None
        self._session_active: bool = False
        self._active_state = None
        
        self.model = None
        self.processor = None

        if not self.remote_url:
             # Default to localhost if not provided, since we only support remote now
             self.remote_url = "http://localhost:8001"
             
        logger.info(f"Executor initialized in REMOTE mode. Target: {self.remote_url}")
        print(f"Executor initialized in REMOTE mode. Target: {self.remote_url}")
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.mount("http://localhost", adapter)
        self._verify_server_connection()

    def _verify_server_connection(self):
        try:
            response = self.session.get(f"{self.remote_url}/health", timeout=5)
            response.raise_for_status()
            health = response.json()
            logger.info(f"Server health: {health}")
            print(f"Server health: {health}")
            if health.get("status") != "healthy":
                logger.warning("Server status is not healthy")
        except requests.RequestException as e:
            logger.warning(f"Could not connect to SAM3 server at {self.remote_url}: {e}")
            print(f"Warning: Could not connect to SAM3 server at {self.remote_url}: {e}")



    def _image_to_base64(self, image_input: Union[np.ndarray, Image.Image]) -> str:
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype(np.uint8))
        else:
            image = image_input

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _base64_to_mask(self, b64_string: str) -> np.ndarray:
        img_data = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_data)).convert("L")
        return np.array(img).astype(bool)

    def _to_pil(self, image_input: Union[np.ndarray, Image.Image]) -> Image.Image:
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input.astype(np.uint8))
        return image_input

    def segment(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: Optional[str] = None,
        box: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        prompts_list: Optional[List[Dict]] = None
    ) -> List[np.ndarray]:
        return self._segment_remote(
            image, text_prompt, box, points, point_labels, prompts_list
        )

    def execute(
        self, 
        image_input: Union[np.ndarray, Image.Image], 
        box: Optional[List[float]], 
        text_prompt: Optional[str]
    ) -> List[np.ndarray]:
        return self.segment(image_input, text_prompt=text_prompt, box=box)

    def encode_image(self, image_input: Union[np.ndarray, Image.Image]) -> bool:
        pil_image = self._to_pil(image_input)
        self._cached_image = pil_image
        self._cached_image_size = pil_image.size

        try:
            payload = {"image_base64": self._image_to_base64(pil_image)}
            response = self.session.post(
                f"{self.remote_url}/set_image", 
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            self._session_active = True
            return True
        except Exception as e:
            logger.error(f"Remote image encoding failed: {e}")
            print(f"Remote image encoding failed: {e}")
            return False

    def predict_masks(
        self, 
        box: Optional[List[float]] = None, 
        text_prompt: Optional[str] = None
    ) -> List[np.ndarray]:
        if not self._session_active or self._cached_image is None:
            logger.error("No cached image. Call encode_image first.")
            print("Error: No cached image. Call encode_image first.")
            return []

        return self._predict_remote(box, text_prompt)

    def _segment_remote(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: Optional[str] = None,
        box: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        prompts_list: Optional[List[Dict]] = None
    ) -> List[np.ndarray]:
        pil_image = self._to_pil(image)
        orig_w, orig_h = pil_image.size

        prompts = []
        if prompts_list:
            # Batch mode: Use provided list
            for p in prompts_list:
                # Normalize boxes/points if needed
                if p["type"] == "box":
                    b = p["box"]
                    p_norm = p.copy()
                    # Only normalize if box values are > 1 (pixel coordinates)
                    # If already normalized (0-1 range), keep as is
                    if b and any(coord > 1 for coord in b):
                        p_norm["box"] = [
                            b[0] / orig_w, b[1] / orig_h,
                            b[2] / orig_w, b[3] / orig_h
                        ]
                    prompts.append(p_norm)
                else:
                    prompts.append(p)
        else:
            # Legacy single prompt mode
            if text_prompt:
                prompts.append({"type": "text", "text": text_prompt})
            if box:
                normalized_box = [
                    box[0] / orig_w,
                    box[1] / orig_h,
                    box[2] / orig_w,
                    box[3] / orig_h,
                ]
                prompts.append({"type": "box", "box": normalized_box, "label": True})
        if points and point_labels:
            normalized_points = [[p[0] / orig_w, p[1] / orig_h] for p in points]
            prompts.append({
                "type": "point", 
                "points": normalized_points, 
                "point_labels": point_labels
            })

        if not prompts:
            logger.warning("No prompts provided")
            return []

        try:
            payload = {
                "image": self._image_to_base64(pil_image),
                "prompts": prompts,
                "confidence_threshold": self.confidence_threshold,
            }
            response = self.session.post(
                f"{self.remote_url}/api/v1/image/segment",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            masks = [self._base64_to_mask(m) for m in data.get("masks", [])]
            logger.info(f"Remote segmentation: {len(masks)} masks, {data.get('inference_time_ms', 0):.1f}ms")
            return masks

        except Exception as e:
            logger.error(f"Remote segmentation failed: {e}")
            print(f"Remote segmentation failed: {e}")
            return []

    def _predict_remote(
        self, 
        box: Optional[List[float]] = None, 
        text_prompt: Optional[str] = None
    ) -> List[np.ndarray]:
        try:
            payload = {}
            if text_prompt:
                payload["text_prompt"] = text_prompt
            if box:
                payload["box"] = list(map(float, box))

            response = self.session.post(
                f"{self.remote_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            masks = [self._base64_to_mask(m) for m in data.get("masks_base64", [])]
            return masks

        except Exception as e:
            logger.error(f"Remote prediction failed: {e}")
            print(f"Remote prediction failed: {e}")
            return []



    def _process_mask_output(
        self, 
        masks: Any, 
        image_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        if masks is None:
            return []

        orig_w, orig_h = image_size

        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()

        extracted_masks = []

        if masks.ndim == 4:
            for b in range(masks.shape[0]):
                for n in range(masks.shape[1]):
                    extracted_masks.append(masks[b, n])
        elif masks.ndim == 3:
            for n in range(masks.shape[0]):
                extracted_masks.append(masks[n])
        elif masks.ndim == 2:
            extracted_masks.append(masks)

        final_masks = []
        for m in extracted_masks:
            if m.shape != (orig_h, orig_w):
                m = cv2.resize(
                    m.astype(np.float32), 
                    (orig_w, orig_h), 
                    interpolation=cv2.INTER_NEAREST
                )
            final_masks.append((m > 0).astype(bool))

        return final_masks

    def get_server_info(self) -> Optional[Dict]:
        if not self.remote_url:
            return {"mode": "local", "device": self.device}

        try:
            response = self.session.get(f"{self.remote_url}/models/info", timeout=5)
            response.raise_for_status()
            info = response.json()
            info["mode"] = "remote"
            info["url"] = self.remote_url
            return info
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")
            return None

    def is_healthy(self) -> bool:
        if self.remote_url:
            try:
                response = self.session.get(f"{self.remote_url}/health", timeout=5)
                return response.status_code == 200
            except:
                return False
        return False
