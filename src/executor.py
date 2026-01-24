import torch
import numpy as np
import cv2
import requests
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
        timeout: int = 120
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

        if self.remote_url:
            logger.info(f"Executor initialized in REMOTE mode. Target: {self.remote_url}")
            print(f"Executor initialized in REMOTE mode. Target: {self.remote_url}")
            self.session = requests.Session()
            self._verify_server_connection()
        else:
            logger.info(f"Executor initializing in LOCAL mode on {self.device}...")
            print(f"Executor initializing in LOCAL mode on {self.device}...")
            self.session = None
            self._load_local_model(model_path)

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

    def _load_local_model(self, model_path: str):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            logger.error(f"SAM3 not installed: {e}")
            raise ImportError("SAM3 library not installed. Install sam3 or use remote mode.")

        load_from_HF = False
        resolved_checkpoint_path = None

        if model_path == "facebook/sam3":
            load_from_HF = True
        elif Path(model_path).exists():
            resolved_checkpoint_path = model_path
        else:
            load_from_HF = True

        logger.info(f"Loading model: {resolved_checkpoint_path or model_path}")
        print(f"Loading model: {resolved_checkpoint_path or model_path}")

        try:
            self.model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=resolved_checkpoint_path,
                load_from_HF=load_from_HF,
                eval_mode=True
            )
        except Exception as e:
            logger.warning(f"First load attempt failed: {e}")
            try:
                self.model = build_sam3_image_model(
                    device=self.device,
                    ckpt_path=resolved_checkpoint_path
                )
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise

        self.processor = Sam3Processor(
            model=self.model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=self.confidence_threshold
        )
        logger.info("SAM3 model loaded successfully")
        print("SAM3 model loaded successfully")

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
    ) -> List[np.ndarray]:
        if self.remote_url:
            return self._segment_remote(image, text_prompt, box, points, point_labels)
        else:
            return self._segment_local(image, text_prompt, box, points, point_labels)

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

        if self.remote_url:
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
        else:
            try:
                self._active_state = self.processor.set_image(pil_image)
                self._session_active = True
                return True
            except Exception as e:
                logger.error(f"Local image encoding failed: {e}")
                print(f"Local image encoding failed: {e}")
                self._active_state = None
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

        if self.remote_url:
            return self._predict_remote(box, text_prompt)
        else:
            return self._predict_local(box, text_prompt)

    def _segment_remote(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: Optional[str] = None,
        box: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        pil_image = self._to_pil(image)
        orig_w, orig_h = pil_image.size

        prompts = []
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

    def _segment_local(
        self,
        image: Union[np.ndarray, Image.Image],
        text_prompt: Optional[str] = None,
        box: Optional[List[float]] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        pil_image = self._to_pil(image)
        orig_w, orig_h = pil_image.size

        try:
            state = self.processor.set_image(pil_image)

            if text_prompt:
                state = self.processor.set_text_prompt(prompt=text_prompt, state=state)

            if box:
                state = self.processor.add_geometric_prompt(
                    box=box,
                    label=1,
                    state=state
                )
                if "language_features" not in state.get("backbone_out", {}):
                    dummy_text = self.processor.model.backbone.forward_text(["visual"], device=self.device)
                    state["backbone_out"].update(dummy_text)
                if hasattr(self.processor, '_forward_grounding'):
                    state = self.processor._forward_grounding(state)

            if points and point_labels:
                pts_tensor = torch.tensor(points, device=self.device, dtype=torch.float32).view(-1, 1, 2)
                labels_tensor = torch.tensor(point_labels, device=self.device, dtype=torch.long).view(-1, 1)

                if "language_features" not in state.get("backbone_out", {}):
                    dummy_text = self.processor.model.backbone.forward_text(["visual"], device=self.device)
                    state["backbone_out"].update(dummy_text)

                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

                state["geometric_prompt"].append_points(points=pts_tensor, labels=labels_tensor)

                if hasattr(self.processor, '_forward_grounding'):
                    state = self.processor._forward_grounding(state)

            return self._process_mask_output(state.get("masks"), (orig_w, orig_h))

        except Exception as e:
            logger.error(f"Local segmentation failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _predict_local(
        self, 
        box: Optional[List[float]] = None, 
        text_prompt: Optional[str] = None
    ) -> List[np.ndarray]:
        if self._active_state is None:
            logger.error("No active state. Call encode_image first.")
            return []

        try:
            import copy
            state = copy.deepcopy(self._active_state)

            if text_prompt:
                state = self.processor.set_text_prompt(prompt=text_prompt, state=state)

            if box:
                state = self.processor.add_geometric_prompt(
                    box=box,
                    label=1,
                    state=state
                )

            if "language_features" not in state.get("backbone_out", {}):
                dummy_text = self.processor.model.backbone.forward_text(["visual"], device=self.device)
                state["backbone_out"].update(dummy_text)

            if "geometric_prompt" not in state:
                state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

            if hasattr(self.processor, '_forward_grounding'):
                state = self.processor._forward_grounding(state)

            return self._process_mask_output(state.get("masks"), self._cached_image_size)

        except Exception as e:
            logger.error(f"Local prediction failed: {e}")
            import traceback
            traceback.print_exc()
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
        else:
            return self.model is not None and self.processor is not None
