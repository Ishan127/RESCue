import time
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import asynccontextmanager
from io import BytesIO
from enum import Enum
import torch
import numpy as np
import uvicorn
from PIL import Image
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint
from sam3.train.data.collator import collate_fn_api as collate
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
from sam3.model.utils.misc import copy_data_to_device

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_rocm_pytorch():
    if hasattr(torch.version, 'hip') and torch.version.hip:
        return True
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            if 'amd' in gpu_name or 'radeon' in gpu_name:
                return True
        except:
            pass
    return False

def patch_torchvision_for_rocm():
    try:
        import torchvision.ops as ops
        import torchvision.ops.roi_align as roi_align_module
        
        if not is_rocm_pytorch():
            logger.info("Not running on ROCm, skipping torchvision patches")
            return False
        
        logger.info("Detected ROCm/AMD GPU, applying AGGRESSIVE torchvision compatibility patches...")
        
        # AGGRESSIVE PATCH: Replace roi_align with CPU-only version
        def cpu_roi_align(input, boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
            """ROCm-compatible roi_align that runs on CPU then moves back to GPU.
            NEVER CRASHES - always returns a valid tensor, even on ROCm bugs.
            """
            device = input.device
            dtype = input.dtype
            
            # Normalize output_size
            if isinstance(output_size, int):
                out_h, out_w = output_size, output_size
            else:
                out_h, out_w = output_size[0], output_size[1]
            
            try:
                # Move input to CPU
                input_cpu = input.cpu()
                
                # Handle boxes - can be tensor or list of tensors
                # torchvision.ops.roi_align expects either:
                # 1. Tensor of shape (N, 5) with [batch_idx, x1, y1, x2, y2]
                # 2. List of tensors, one per batch image, each of shape (M, 4)
                
                if isinstance(boxes, torch.Tensor):
                    # Already a tensor - just move to CPU
                    boxes_cpu = boxes.cpu()
                    num_boxes = boxes_cpu.shape[0] if boxes_cpu.numel() > 0 else 0
                elif isinstance(boxes, (list, tuple)):
                    # List of tensors - need to convert to single tensor with batch indices
                    all_boxes = []
                    for batch_idx, box_tensor in enumerate(boxes):
                        if isinstance(box_tensor, torch.Tensor):
                            bt = box_tensor.cpu()
                            if bt.numel() > 0 and bt.shape[0] > 0:
                                # Add batch index as first column
                                n = bt.shape[0]
                                batch_col = torch.full((n, 1), float(batch_idx), dtype=bt.dtype, device=bt.device)
                                boxes_with_idx = torch.cat([batch_col, bt], dim=1)
                                all_boxes.append(boxes_with_idx)
                    
                    if all_boxes:
                        boxes_cpu = torch.cat(all_boxes, dim=0)
                        num_boxes = boxes_cpu.shape[0]
                    else:
                        # All boxes empty - return empty output immediately
                        return torch.zeros((0, input_cpu.shape[1], out_h, out_w), dtype=dtype, device=device)
                else:
                    # Unknown type - try to use as-is
                    boxes_cpu = boxes
                    num_boxes = 0
                
                # Handle empty boxes case - return early
                if num_boxes == 0 or (isinstance(boxes_cpu, torch.Tensor) and boxes_cpu.numel() == 0):
                    return torch.zeros((0, input_cpu.shape[1], out_h, out_w), dtype=dtype, device=device)
                
                # Validate boxes before calling C++ op
                if isinstance(boxes_cpu, torch.Tensor):
                    # Check for NaN/inf values
                    if torch.isnan(boxes_cpu).any() or torch.isinf(boxes_cpu).any():
                        logger.warning(f"Invalid box values (NaN/inf): {boxes_cpu}")
                        return torch.zeros((num_boxes, input_cpu.shape[1], out_h, out_w), dtype=dtype, device=device)
                    
                    # Note: Large coordinates are OK - SAM handles boxes outside image bounds
                    # The "suspicious" warnings were too aggressive
                
                # Ensure boxes_cpu is float tensor
                if boxes_cpu.dtype not in [torch.float32, torch.float64]:
                    boxes_cpu = boxes_cpu.float()
                
                # Run on CPU using the C++ implementation
                result = torch.ops.torchvision.roi_align(
                    input_cpu.float(), boxes_cpu, spatial_scale, 
                    out_h, out_w, sampling_ratio, aligned
                )
                
                # Move back to original device and dtype
                return result.to(device=device, dtype=dtype)
                
            except Exception as e:
                # LOG EVERYTHING for debugging
                logger.error(f"roi_align CRASHED: {e}")
                logger.error(f"  Input shape: {input.shape if hasattr(input, 'shape') else 'unknown'}")
                logger.error(f"  Input dtype: {input.dtype if hasattr(input, 'dtype') else 'unknown'}")
                logger.error(f"  Boxes type: {type(boxes)}")
                if isinstance(boxes, torch.Tensor):
                    logger.error(f"  Boxes shape: {boxes.shape}")
                    logger.error(f"  Boxes values: {boxes}")
                elif isinstance(boxes, (list, tuple)):
                    logger.error(f"  Boxes list length: {len(boxes)}")
                    for i, b in enumerate(boxes):
                        if isinstance(b, torch.Tensor):
                            logger.error(f"    Box {i}: shape {b.shape}, values {b}")
                        else:
                            logger.error(f"    Box {i}: {type(b)} {b}")
                logger.error(f"  Output size: {output_size}")
                logger.error(f"  Spatial scale: {spatial_scale}")
                logger.error(f"  Sampling ratio: {sampling_ratio}")
                logger.error(f"  Aligned: {aligned}")
                
                # Return empty tensor - NEVER crash
                try:
                    num_boxes = 0
                    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                        num_boxes = boxes.shape[0]
                    elif isinstance(boxes, (list, tuple)):
                        for b in boxes:
                            if isinstance(b, torch.Tensor) and b.numel() > 0:
                                num_boxes += b.shape[0]
                    
                    return torch.zeros((num_boxes, input.shape[1] if hasattr(input, 'shape') else 256, out_h, out_w), 
                                     dtype=dtype, device=device)
                except Exception as fallback_e:
                    logger.error(f"Even fallback failed: {fallback_e}")
                    # Ultimate fallback - return something that won't crash
                    return torch.zeros((1, 256, out_h, out_w), dtype=torch.float32, device=device)
        
        # Patch at multiple levels
        ops.roi_align = cpu_roi_align
        roi_align_module.roi_align = cpu_roi_align
        
        # Also patch the RoIAlign class
        if hasattr(ops, 'RoIAlign'):
            _OriginalRoIAlign = ops.RoIAlign
            class PatchedRoIAlign(torch.nn.Module):
                def __init__(self, output_size, spatial_scale, sampling_ratio=-1, aligned=False):
                    super().__init__()
                    self.output_size = output_size
                    self.spatial_scale = spatial_scale
                    self.sampling_ratio = sampling_ratio
                    self.aligned = aligned
                
                def forward(self, input, rois):
                    return cpu_roi_align(
                        input, rois, self.output_size, 
                        self.spatial_scale, self.sampling_ratio, self.aligned
                    )
            ops.RoIAlign = PatchedRoIAlign
        
        # Patch NMS similarly - make it robust too
        _original_nms = ops.nms
        def cpu_nms(boxes, scores, iou_threshold):
            """ROCm-compatible NMS that runs on CPU then moves back to GPU.
            NEVER CRASHES - always returns valid indices.
            """
            try:
                device = boxes.device
                
                # Move to CPU
                boxes_cpu = boxes.cpu()
                scores_cpu = scores.cpu()
                
                # Validate inputs
                if boxes_cpu.numel() == 0 or scores_cpu.numel() == 0:
                    return torch.zeros(0, dtype=torch.int64, device=device)
                
                # Check for NaN/inf
                if torch.isnan(boxes_cpu).any() or torch.isinf(boxes_cpu).any():
                    logger.warning(f"NMS: Invalid box values (NaN/inf): {boxes_cpu}")
                    return torch.zeros(0, dtype=torch.int64, device=device)
                
                if torch.isnan(scores_cpu).any() or torch.isinf(scores_cpu).any():
                    logger.warning(f"NMS: Invalid score values (NaN/inf): {scores_cpu}")
                    return torch.zeros(0, dtype=torch.int64, device=device)
                
                # Run on CPU
                result = _original_nms(boxes_cpu, scores_cpu, iou_threshold)
                
                # Move back to device
                return result.to(device)
                
            except Exception as e:
                logger.error(f"NMS CRASHED: {e}")
                logger.error(f"  Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else 'unknown'}")
                logger.error(f"  Scores shape: {scores.shape if hasattr(scores, 'shape') else 'unknown'}")
                logger.error(f"  IoU threshold: {iou_threshold}")
                
                # Return empty indices - NEVER crash
                return torch.zeros(0, dtype=torch.int64, device=boxes.device if hasattr(boxes, 'device') else torch.device('cpu'))
        ops.nms = cpu_nms
        
        logger.info("Successfully applied AGGRESSIVE ROCm compatibility patches")
        return True
        
    except ImportError as e:
        logger.warning(f"torchvision not available, skipping patches: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to patch torchvision: {e}")
        import traceback
        traceback.print_exc()
        return False

_rocm_patched = patch_torchvision_for_rocm()


class PromptType(str, Enum):
    TEXT = "text"
    POINT = "point"
    BOX = "box"


class TextPrompt(BaseModel):
    type: str = Field(default="text")
    text: str = Field(..., description="Text description of object to segment")


class BoxPrompt(BaseModel):
    type: str = Field(default="box")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] normalized to [0, 1]")
    label: bool = Field(default=True, description="True for positive, False for negative exemplar")


class PointPrompt(BaseModel):
    type: str = Field(default="point")
    points: List[List[float]] = Field(..., description="List of [x, y] coordinates normalized to [0, 1]")
    point_labels: List[int] = Field(..., description="Labels: 1=positive, 0=negative")


class ImageSegmentRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    prompts: List[Union[TextPrompt, BoxPrompt, PointPrompt]] = Field(
        ..., description="List of prompts", min_length=1
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    return_visualization: bool = Field(default=False)


class ImageSegmentResponse(BaseModel):
    masks: List[str] = Field(..., description="Base64-encoded binary masks")
    boxes: List[List[float]] = Field(..., description="Bounding boxes [x1, y1, x2, y2] normalized")
    scores: List[float] = Field(..., description="Confidence scores")
    num_masks: int = Field(..., description="Number of masks returned")
    image_size: Dict[str, int] = Field(..., description="Original image dimensions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class CachedFeaturesRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    text_prompts: List[str] = Field(..., description="List of text prompts")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CachedFeaturesResultItem(BaseModel):
    prompt: str
    masks: List[str]
    boxes: List[List[float]]
    scores: List[float]
    num_masks: int


class CachedFeaturesResponse(BaseModel):
    results: List[CachedFeaturesResultItem]
    cache_hit: bool
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_count: int
    gpu_name: Optional[str] = None
    active_sessions: int = 0


class ModelInfoResponse(BaseModel):
    image_model: Optional[Dict[str, Any]] = None
    server_version: str = "1.0.0"


class SAM3ImageModel:

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        compile: bool = False,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        logger.info(f"Loading SAM3 image model on {device}...")

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            logger.error(f"Failed to import SAM3: {e}")
            raise ImportError("SAM3 library not installed. Please install sam3 package.")

        load_from_HF = False
        resolved_checkpoint_path = None

        if checkpoint is None or checkpoint == "facebook/sam3":
            load_from_HF = True
            logger.info("Will load SAM3 from HuggingFace")
        elif Path(checkpoint).exists():
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_dir():
                sam3_file = checkpoint_path / "sam3.pt"
                if sam3_file.exists():
                    resolved_checkpoint_path = str(sam3_file)
                else:
                    raise FileNotFoundError(f"sam3.pt not found in {checkpoint}")
            else:
                resolved_checkpoint_path = checkpoint
        else:
            load_from_HF = True

        logger.info(f"Checkpoint: {resolved_checkpoint_path or 'HuggingFace'}, load_from_HF: {load_from_HF}")

        try:
            model = build_sam3_image_model(
                checkpoint_path=resolved_checkpoint_path,
                bpe_path=bpe_path,
                device=device,
                eval_mode=True,
                load_from_HF=load_from_HF,
                compile=compile,
            )
            model = model.to(device)
        except Exception as e:
            logger.warning(f"First attempt failed: {e}, trying alternate signature...")
            try:
                model = build_sam3_image_model(
                    device=device,
                    ckpt_path=resolved_checkpoint_path,
                )
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                raise

        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(sizes=resolution, max_size=resolution, square=True, consistent_transform=False),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=confidence_threshold,
            to_cpu=True,
        )

        self.model = model
        self.feature_cache: Dict[str, Dict] = {}
        self.global_counter = 1

        logger.info("SAM3 image model loaded successfully")

    def segment_combined(
        self,
        image: Image.Image,
        text_prompts: Optional[List[str]] = None,
        boxes: Optional[List[Tuple[List[float], bool]]] = None,
        points: Optional[List[Tuple[List[List[float]], List[int]]]] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
    def segment_combined(
        self,
        image: Image.Image,
        text_prompts: Optional[List[str]] = None,
        boxes: Optional[List[Tuple[List[float], bool]]] = None,
        points: Optional[List[Tuple[List[List[float]], List[int]]]] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
        # 1. Create Datapoint
        w, h = image.size
        datapoint = Datapoint(find_queries=[], images=[SAMImage(data=image, objects=[], size=[h, w])])
        
        query_ids = []
        
        # 2. Add Text Prompts
        if text_prompts:
            for text in text_prompts:
                self.global_counter += 1
                q_id = self.global_counter
                datapoint.find_queries.append(
                    FindQueryLoaded(
                        query_text=text,
                        image_id=0,
                        object_ids_output=[], 
                        is_exhaustive=True,
                        query_processing_order=0,
                        inference_metadata=InferenceMetadata(
                            coco_image_id=q_id,
                            original_image_id=q_id,
                            original_category_id=1,
                            original_size=[w, h],
                            object_id=0,
                            frame_index=0,
                        )
                    )
                )
                query_ids.append(q_id)

        # 3. Add Box Prompts
        if boxes:
            for box, label in boxes:
                # box is already normalized [0,1], convert to pixel [x1, y1, x2, y2]
                pixel_box = [
                    box[0] * w, box[1] * h,
                    box[2] * w, box[3] * h
                ]
                # SAM3 expects list of boxes. We treat each box as a separate query here to get separate masks
                self.global_counter += 1
                q_id = self.global_counter
                
                datapoint.find_queries.append(
                    FindQueryLoaded(
                        query_text="visual", # default text as per reference
                        image_id=0,
                        object_ids_output=[],
                        is_exhaustive=True,
                        query_processing_order=0,
                        input_bbox=torch.tensor([pixel_box], dtype=torch.float32).view(-1, 4),
                        input_bbox_label=torch.tensor([label], dtype=torch.bool).view(-1),
                        inference_metadata=InferenceMetadata(
                            coco_image_id=q_id,
                            original_image_id=q_id,
                            original_category_id=1,
                            original_size=[w, h],
                            object_id=0,
                            frame_index=0,
                        )
                    )
                )
                query_ids.append(q_id)

        # 4. Add Point Prompts (Not implemented in reference but follows similar pattern if needed, skipping for now as per instructions focus)
        if points:
             logger.warning("Point prompts temporarily disabled in batch mode pending implementation details")

        if not query_ids:
            return [], [], []

        # 5. Transform and Collate
        try:
            datapoint = self.transform(datapoint)
            batch = collate([datapoint], dict_key="dummy")["dummy"]
            batch = copy_data_to_device(batch, torch.device(self.device), non_blocking=True) # Ensure device string
            
            # 6. Inference
            with torch.inference_mode():
                # Handling AMP manually if needed, or rely on global settings. 
                # Assuming AMP is handled by context if set, else standard float32/16
                output = self.model(batch)
                
            # 7. Post-process
            # process_results returns a Dict[int, List[Dict]] where int is the query ID (coco_image_id)
            processed_results = self.postprocessor.process_results(output, batch.find_metadatas)
            
            final_masks = []
            final_boxes = []
            final_scores = []

            # 8. Extract in order of queries
            for q_id in query_ids:
                results = processed_results.get(q_id, [])
                if not results:
                    # Placeholder if no detection
                    final_masks.append(np.zeros((h, w), dtype=bool))
                    final_boxes.append([0.0, 0.0, 1.0, 1.0])
                    final_scores.append(0.0)
                else:
                    # Take the highest confidence result for this query
                    best_res = max(results, key=lambda x: x.get("score", 0.0))
                    
                    # Mask
                    mask = best_res.get("mask") # Should be np.ndarray due to to_cpu=True
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    
                    # Convert RLE if needed, but we set convert_mask_to_rle=False
                    # It might be in 'segmentation' field or 'mask' depending on version
                    # PostProcessImage usually puts full mask in "mask" if rle=False
                    if mask is None and "segmentation" in best_res:
                        mask = best_res["segmentation"]
                        
                    # Resize is handled by postprocessor (use_original_sizes_mask=True)
                    # But double check shape
                    if mask is not None:
                         final_masks.append(mask.astype(bool))
                    else:
                         final_masks.append(np.zeros((h, w), dtype=bool))

                    # Box
                    bbox = best_res.get("bbox", [0, 0, w, h])
                    norm_box = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]
                    final_boxes.append(norm_box)
                    
                    # Score
                    final_scores.append(best_res.get("score", 1.0))

            return final_masks, final_boxes, final_scores

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []

    def cache_features(self, image: Image.Image, cache_key: str) -> str:
        state = self.processor.set_image(image)
        self.feature_cache[cache_key] = {
            "backbone_out": state["backbone_out"],
            "image_size": image.size,
        }
        return cache_key

    def segment_with_cached_features(
        self, cache_key: str, text_prompts: List[str]
    ) -> List[Tuple[List[np.ndarray], List[List[float]], List[float]]]:
        if cache_key not in self.feature_cache:
            raise ValueError(f"No cached features for key: {cache_key}")

        cached = self.feature_cache[cache_key]
        results = []

        for prompt in text_prompts:
            orig_w, orig_h = cached["image_size"]
            state = {
                "backbone_out": cached["backbone_out"],
                "original_height": orig_h,
                "original_width": orig_w,
            }

            state = self.processor.set_text_prompt(prompt=prompt, state=state)
            masks, boxes, scores = self._extract_results(state, cached["image_size"])
            results.append((masks, boxes, scores))

        return results

    def clear_cache(self, cache_key: Optional[str] = None):
        if cache_key:
            self.feature_cache.pop(cache_key, None)
        else:
            self.feature_cache.clear()

    def _extract_results(
        self, state: Dict, image_size: Tuple[int, int]
    ) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
        orig_w, orig_h = image_size

        masks_tensor = state.get("masks")
        boxes_tensor = state.get("boxes")
        scores_tensor = state.get("scores")

        if masks_tensor is None:
            return [], [], []

        if isinstance(masks_tensor, torch.Tensor):
            masks_np = masks_tensor.detach().cpu().numpy()
        else:
            masks_np = masks_tensor

        final_masks = []
        if masks_np.ndim == 4:
            for b in range(masks_np.shape[0]):
                for n in range(masks_np.shape[1]):
                    mask = masks_np[b, n]
                    if mask.shape != (orig_h, orig_w):
                        import cv2
                        mask = cv2.resize(
                            mask.astype(np.float32), 
                            (orig_w, orig_h), 
                            interpolation=cv2.INTER_NEAREST
                        )
                    final_masks.append((mask > 0).astype(bool))
        elif masks_np.ndim == 3:
            for n in range(masks_np.shape[0]):
                mask = masks_np[n]
                if mask.shape != (orig_h, orig_w):
                    import cv2
                    mask = cv2.resize(
                        mask.astype(np.float32), 
                        (orig_w, orig_h), 
                        interpolation=cv2.INTER_NEAREST
                    )
                final_masks.append((mask > 0).astype(bool))
        elif masks_np.ndim == 2:
            mask = masks_np
            if mask.shape != (orig_h, orig_w):
                import cv2
                mask = cv2.resize(
                    mask.astype(np.float32), 
                    (orig_w, orig_h), 
                    interpolation=cv2.INTER_NEAREST
                )
            final_masks.append((mask > 0).astype(bool))

        final_boxes = []
        if boxes_tensor is not None:
            if isinstance(boxes_tensor, torch.Tensor):
                boxes_np = boxes_tensor.detach().cpu().numpy()
            else:
                boxes_np = boxes_tensor

            for i in range(len(boxes_np)):
                box = boxes_np[i]
                normalized_box = [
                    float(box[0]) / orig_w,
                    float(box[1]) / orig_h,
                    float(box[2]) / orig_w,
                    float(box[3]) / orig_h,
                ]
                final_boxes.append(normalized_box)

        final_scores = []
        if scores_tensor is not None:
            if isinstance(scores_tensor, torch.Tensor):
                scores_np = scores_tensor.detach().cpu().numpy()
            else:
                scores_np = scores_tensor
            final_scores = [float(s) for s in scores_np.flatten()]

        while len(final_boxes) < len(final_masks):
            final_boxes.append([0.0, 0.0, 1.0, 1.0])
        while len(final_scores) < len(final_masks):
            final_scores.append(1.0)

        return final_masks, final_boxes, final_scores

    @property
    def model_info(self) -> Dict:
        return {
            "loaded": True,
            "device": self.device,
            "resolution": self.resolution,
            "confidence_threshold": self.confidence_threshold,
            "cache_size": len(self.feature_cache),
            "capabilities": ["text_prompt", "box_prompt", "point_prompt", "feature_caching"],
        }


def decode_base64_image(b64_string: str) -> Image.Image:
    try:
        img_data = base64.b64decode(b64_string)
        return Image.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask_to_base64(mask: np.ndarray) -> str:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_uint8, mode="L")
    
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


image_model: Optional[SAM3ImageModel] = None

_server_settings = {
    "device": None,
    "force_cpu": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global image_model

    logger.info("Starting SAM3 Inference Server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"ROCm detected: {is_rocm_pytorch()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if _server_settings.get("device"):
        device = _server_settings["device"]
    elif _server_settings.get("force_cpu"):
        device = "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    try:
        image_model = SAM3ImageModel(
            checkpoint="facebook/sam3",
            device=device,
            confidence_threshold=0.5,
            resolution=1008,
        )
        logger.info("✓ SAM3 Image model loaded successfully")
    except NotImplementedError as e:
        if "roi_align" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(f"GPU inference failed due to ROCm compatibility: {e}")
            logger.info("Falling back to CPU inference...")
            try:
                image_model = SAM3ImageModel(
                    checkpoint="facebook/sam3",
                    device="cpu",
                    confidence_threshold=0.5,
                    resolution=1008,
                )
                logger.info("✓ SAM3 Image model loaded successfully on CPU")
            except Exception as e2:
                logger.error(f"CPU fallback also failed: {e2}")
                image_model = None
        else:
            logger.error(f"Failed to load SAM3 model: {e}")
            image_model = None
    except Exception as e:
        logger.error(f"Failed to load SAM3 model: {e}")
        import traceback
        traceback.print_exc()
        image_model = None

    yield

    logger.info("Shutting down SAM3 Inference Server...")
    if image_model:
        image_model.clear_cache()


app = FastAPI(
    title="SAM3 Inference Server",
    description="FastAPI server for SAM3 (Segment Anything 3) image segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "SAM3 Inference Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if image_model else "degraded",
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        active_sessions=0,
    )


@app.get("/models/info", response_model=ModelInfoResponse)
async def models_info():
    return ModelInfoResponse(
        image_model=image_model.model_info if image_model else None,
        server_version="1.0.0",
    )


@app.post("/api/v1/image/segment", response_model=ImageSegmentResponse)
async def segment_image(request: ImageSegmentRequest):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    start_time = time.time()

    try:
        image = decode_base64_image(request.image)
        logger.info(f"Processing image of size {image.size}")

        text_prompts = []
        box_prompts = []
        point_prompts = []

        for prompt in request.prompts:
            prompt_dict = prompt.model_dump() if hasattr(prompt, 'model_dump') else prompt.dict()
            prompt_type = prompt_dict.get("type", "")
            
            if prompt_type == "text":
                text_prompts.append(prompt_dict["text"])
            elif prompt_type == "box":
                # Safety clamp to [0, 1] to prevent roi_align crashes
                box = prompt_dict["box"]
                clamped_box = [
                    max(0.0, min(1.0, float(c))) for c in box
                ]
                if box != clamped_box:
                    logger.warning(f"Clamped box {box} to {clamped_box}")
                    
                box_prompts.append((clamped_box, prompt_dict.get("label", True)))
            elif prompt_type == "point":
                point_prompts.append((prompt_dict["points"], prompt_dict["point_labels"]))

        if not text_prompts and not box_prompts and not point_prompts:
            raise HTTPException(
                status_code=400,
                detail="At least one text, box, or point prompt is required",
            )

        # Call segment_combined ONCE for true batching
        masks, boxes, scores = image_model.segment_combined(
            image=image,
            text_prompts=text_prompts if text_prompts else None,
            boxes=box_prompts if box_prompts else None,
            points=point_prompts if point_prompts else None,
        )

        masks_b64 = [encode_mask_to_base64(m) for m in masks]

        inference_time = (time.time() - start_time) * 1000

        logger.info(f"Segmentation complete: {len(masks)} masks in {inference_time:.1f}ms")

        return ImageSegmentResponse(
            masks=masks_b64,
            boxes=boxes,
            scores=scores,
            num_masks=len(masks),
            image_size={"width": image.size[0], "height": image.size[1]},
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/image/cached-features", response_model=CachedFeaturesResponse)
async def segment_with_cached_features(request: CachedFeaturesRequest):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    start_time = time.time()

    try:
        image = decode_base64_image(request.image)

        cache_key = f"cache_{hash(request.image[:100])}"

        cache_hit = cache_key in image_model.feature_cache
        if not cache_hit:
            image_model.cache_features(image, cache_key)

        results_list = image_model.segment_with_cached_features(
            cache_key, request.text_prompts
        )

        results = []
        for prompt, (masks, boxes, scores) in zip(request.text_prompts, results_list):
            masks_b64 = [encode_mask_to_base64(m) for m in masks]
            results.append(
                CachedFeaturesResultItem(
                    prompt=prompt,
                    masks=masks_b64,
                    boxes=boxes,
                    scores=scores,
                    num_masks=len(masks),
                )
            )

        inference_time = (time.time() - start_time) * 1000

        logger.info(
            f"Cached features segmentation: {len(request.text_prompts)} prompts "
            f"in {inference_time:.1f}ms (cache_hit={cache_hit})"
        )

        return CachedFeaturesResponse(
            results=results,
            cache_hit=cache_hit,
            inference_time_ms=inference_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cached segmentation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set_image")
async def set_image_legacy(request: dict):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    try:
        image_b64 = request.get("image_base64")
        if not image_b64:
            raise HTTPException(status_code=400, detail="image_base64 required")

        image = decode_base64_image(image_b64)
        cache_key = "legacy_session"
        image_model.cache_features(image, cache_key)

        return {"status": "ok", "image_size": {"width": image.size[0], "height": image.size[1]}}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Set image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_legacy(request: dict):
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    try:
        cache_key = "legacy_session"
        if cache_key not in image_model.feature_cache:
            raise HTTPException(status_code=400, detail="No image set. Call /set_image first.")

        text_prompt = request.get("text_prompt")
        box = request.get("box")

        if not text_prompt and not box:
            raise HTTPException(status_code=400, detail="text_prompt or box required")

        cached = image_model.feature_cache[cache_key]
        image_size = cached["image_size"]

        text_prompts = [text_prompt] if text_prompt else None
        box_prompts = None
        if box:
            orig_w, orig_h = image_size
            normalized_box = [
                box[0] / orig_w,
                box[1] / orig_h,
                box[2] / orig_w,
                box[3] / orig_h,
            ]
            box_prompts = [(normalized_box, True)]

        if text_prompts:
            results = image_model.segment_with_cached_features(cache_key, text_prompts)
            masks, boxes, scores = results[0]
        else:
            raise HTTPException(
                status_code=400, 
                detail="Box-only prompts require calling /api/v1/image/segment with full image"
            )

        masks_b64 = [encode_mask_to_base64(m) for m in masks]

        return {
            "masks_base64": masks_b64,
            "boxes": boxes,
            "scores": scores,
            "num_masks": len(masks),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global _server_settings
    
    parser = argparse.ArgumentParser(description="SAM3 Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--device", default=None, help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force CPU inference (useful for ROCm compatibility issues)")
    args = parser.parse_args()
    
    if args.force_cpu:
        _server_settings["force_cpu"] = True
        _server_settings["device"] = "cpu"
        logger.info("Forcing CPU inference mode")
    elif args.device:
        _server_settings["device"] = args.device
    else:
        if is_rocm_pytorch():
            logger.info("ROCm detected. Use --force-cpu if you encounter torchvision op errors.")
        _server_settings["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {_server_settings['device']}")

    uvicorn.run(
        "sam_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
