"""
SAM3 Inference Server - FastAPI Application
Based on rpol-recart/sam3_inference implementation pattern

Endpoints:
  - POST /api/v1/image/segment - Main segmentation endpoint
  - POST /health - Health check
  - GET /models/info - Model information
"""
import sys
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Schemas (based on rpol-recart/sam3_inference/api/schemas)
# ============================================================================

class PromptType(str, Enum):
    """Type of prompt for segmentation."""
    TEXT = "text"
    POINT = "point"
    BOX = "box"


class TextPrompt(BaseModel):
    """Text-based prompt."""
    type: str = Field(default="text")
    text: str = Field(..., description="Text description of object to segment")


class BoxPrompt(BaseModel):
    """Box-based prompt (normalized coordinates)."""
    type: str = Field(default="box")
    box: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2] normalized to [0, 1]")
    label: bool = Field(default=True, description="True for positive, False for negative exemplar")


class PointPrompt(BaseModel):
    """Point-based prompt."""
    type: str = Field(default="point")
    points: List[List[float]] = Field(..., description="List of [x, y] coordinates normalized to [0, 1]")
    point_labels: List[int] = Field(..., description="Labels: 1=positive, 0=negative")


class ImageSegmentRequest(BaseModel):
    """Request for image segmentation."""
    image: str = Field(..., description="Base64-encoded image")
    prompts: List[Union[TextPrompt, BoxPrompt, PointPrompt]] = Field(
        ..., description="List of prompts", min_length=1
    )
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    return_visualization: bool = Field(default=False)


class ImageSegmentResponse(BaseModel):
    """Response for image segmentation."""
    masks: List[str] = Field(..., description="Base64-encoded binary masks")
    boxes: List[List[float]] = Field(..., description="Bounding boxes [x1, y1, x2, y2] normalized")
    scores: List[float] = Field(..., description="Confidence scores")
    num_masks: int = Field(..., description="Number of masks returned")
    image_size: Dict[str, int] = Field(..., description="Original image dimensions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class CachedFeaturesRequest(BaseModel):
    """Request for cached features segmentation."""
    image: str = Field(..., description="Base64-encoded image")
    text_prompts: List[str] = Field(..., description="List of text prompts")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class CachedFeaturesResultItem(BaseModel):
    """Single result item for cached features."""
    prompt: str
    masks: List[str]
    boxes: List[List[float]]
    scores: List[float]
    num_masks: int


class CachedFeaturesResponse(BaseModel):
    """Response for cached features segmentation."""
    results: List[CachedFeaturesResultItem]
    cache_hit: bool
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    gpu_available: bool
    gpu_count: int
    gpu_name: Optional[str] = None
    active_sessions: int = 0


class ModelInfoResponse(BaseModel):
    """Model information response."""
    image_model: Optional[Dict[str, Any]] = None
    server_version: str = "1.0.0"


# ============================================================================
# SAM3 Image Model Wrapper (based on rpol-recart/sam3_inference/models/sam3_image.py)
# ============================================================================

class SAM3ImageModel:
    """Wrapper for SAM3 image inference."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        bpe_path: Optional[str] = None,
        device: str = "cuda:0",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        compile: bool = False,
    ):
        """Initialize SAM3 image model.

        Args:
            checkpoint: Model checkpoint path or HuggingFace ID
            bpe_path: Path to BPE tokenizer file
            device: Device to load model on
            confidence_threshold: Confidence threshold for filtering
            resolution: Input image resolution
            compile: Enable torch.compile optimization
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        logger.info(f"Loading SAM3 image model on {device}...")

        # Import SAM3 components
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            logger.error(f"Failed to import SAM3: {e}")
            raise ImportError("SAM3 library not installed. Please install sam3 package.")

        # Determine checkpoint loading strategy
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

        # Build model
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

        # Create processor
        self.processor = Sam3Processor(
            model=model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        # Feature cache for multiple prompts on same image
        self.feature_cache: Dict[str, Dict] = {}

        logger.info("SAM3 image model loaded successfully")

    def segment_combined(
        self,
        image: Image.Image,
        text_prompts: Optional[List[str]] = None,
        boxes: Optional[List[Tuple[List[float], bool]]] = None,
        points: Optional[List[Tuple[List[List[float]], List[int]]]] = None,
    ) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
        """Segment with combined prompts.

        Args:
            image: PIL Image
            text_prompts: List of text prompts
            boxes: List of (box, label) tuples, box is [x1, y1, x2, y2] normalized
            points: List of (points, labels) tuples

        Returns:
            Tuple of (masks as numpy arrays, boxes normalized, scores)
        """
        state = self.processor.set_image(image)
        orig_w, orig_h = image.size

        # Add text prompts
        if text_prompts:
            for text in text_prompts:
                state = self.processor.set_text_prompt(prompt=text, state=state)

        # Add box prompts (convert normalized to pixel coordinates)
        if boxes:
            for box, label in boxes:
                # box is [x1, y1, x2, y2] normalized
                box_pixels = [
                    box[0] * orig_w,
                    box[1] * orig_h,
                    box[2] * orig_w,
                    box[3] * orig_h,
                ]
                state = self.processor.add_geometric_prompt(
                    box=box_pixels, label=1 if label else 0, state=state
                )

        # Add point prompts
        if points:
            for point_list, point_labels in points:
                points_pixels = torch.tensor(
                    [[x * orig_w, y * orig_h] for x, y in point_list],
                    device=self.device, dtype=torch.float32
                ).view(-1, 1, 2)
                
                point_tensor_labels = torch.tensor(
                    point_labels, device=self.device, dtype=torch.long
                ).view(-1, 1)

                # Check if language features exist
                if "language_features" not in state.get("backbone_out", {}):
                    dummy_text_outputs = self.processor.model.backbone.forward_text(
                        ["visual"], device=self.device
                    )
                    state["backbone_out"].update(dummy_text_outputs)

                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = self.processor.model._get_dummy_prompt()

                state["geometric_prompt"].append_points(
                    points=points_pixels,
                    labels=point_tensor_labels
                )

                if hasattr(self.processor, '_forward_grounding'):
                    state = self.processor._forward_grounding(state)

        return self._extract_results(state, image.size)

    def cache_features(self, image: Image.Image, cache_key: str) -> str:
        """Cache image features for reuse with multiple prompts."""
        state = self.processor.set_image(image)
        self.feature_cache[cache_key] = {
            "backbone_out": state["backbone_out"],
            "image_size": image.size,
        }
        return cache_key

    def segment_with_cached_features(
        self, cache_key: str, text_prompts: List[str]
    ) -> List[Tuple[List[np.ndarray], List[List[float]], List[float]]]:
        """Segment using cached features with multiple text prompts."""
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
        """Clear feature cache."""
        if cache_key:
            self.feature_cache.pop(cache_key, None)
        else:
            self.feature_cache.clear()

    def _extract_results(
        self, state: Dict, image_size: Tuple[int, int]
    ) -> Tuple[List[np.ndarray], List[List[float]], List[float]]:
        """Extract and format results from inference state."""
        orig_w, orig_h = image_size

        # Get masks from state
        masks_tensor = state.get("masks")
        boxes_tensor = state.get("boxes")
        scores_tensor = state.get("scores")

        if masks_tensor is None:
            return [], [], []

        # Convert to numpy
        if isinstance(masks_tensor, torch.Tensor):
            masks_np = masks_tensor.detach().cpu().numpy()
        else:
            masks_np = masks_tensor

        # Process masks
        final_masks = []
        if masks_np.ndim == 4:
            # (B, N, H, W)
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
            # (N, H, W)
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

        # Process boxes (normalize to [0, 1])
        final_boxes = []
        if boxes_tensor is not None:
            if isinstance(boxes_tensor, torch.Tensor):
                boxes_np = boxes_tensor.detach().cpu().numpy()
            else:
                boxes_np = boxes_tensor

            for i in range(len(boxes_np)):
                box = boxes_np[i]
                # Normalize XYXY format
                normalized_box = [
                    float(box[0]) / orig_w,
                    float(box[1]) / orig_h,
                    float(box[2]) / orig_w,
                    float(box[3]) / orig_h,
                ]
                final_boxes.append(normalized_box)

        # Process scores
        final_scores = []
        if scores_tensor is not None:
            if isinstance(scores_tensor, torch.Tensor):
                scores_np = scores_tensor.detach().cpu().numpy()
            else:
                scores_np = scores_tensor
            final_scores = [float(s) for s in scores_np.flatten()]

        # Pad with default values if needed
        while len(final_boxes) < len(final_masks):
            final_boxes.append([0.0, 0.0, 1.0, 1.0])
        while len(final_scores) < len(final_masks):
            final_scores.append(1.0)

        return final_masks, final_boxes, final_scores

    @property
    def model_info(self) -> Dict:
        """Get model information."""
        return {
            "loaded": True,
            "device": self.device,
            "resolution": self.resolution,
            "confidence_threshold": self.confidence_threshold,
            "cache_size": len(self.feature_cache),
            "capabilities": ["text_prompt", "box_prompt", "point_prompt", "feature_caching"],
        }


# ============================================================================
# Utility Functions
# ============================================================================

def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        img_data = base64.b64decode(b64_string)
        return Image.open(BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode binary mask to base64 PNG string."""
    # Convert boolean mask to uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)
    mask_img = Image.fromarray(mask_uint8, mode="L")
    
    buffer = BytesIO()
    mask_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================================
# FastAPI Application
# ============================================================================

# Global model instance
image_model: Optional[SAM3ImageModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model loading/unloading."""
    global image_model

    logger.info("Starting SAM3 Inference Server...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load model from settings or defaults
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    try:
        image_model = SAM3ImageModel(
            checkpoint="facebook/sam3",
            device=device,
            confidence_threshold=0.5,
            resolution=1008,
        )
        logger.info("✓ SAM3 Image model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SAM3 model: {e}")
        # Don't raise - allow server to start for debugging
        image_model = None

    yield

    # Cleanup
    logger.info("Shutting down SAM3 Inference Server...")
    if image_model:
        image_model.clear_cache()


# Create FastAPI app
app = FastAPI(
    title="SAM3 Inference Server",
    description="FastAPI server for SAM3 (Segment Anything 3) image segmentation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAM3 Inference Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if image_model else "degraded",
        gpu_available=torch.cuda.is_available(),
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        active_sessions=0,
    )


@app.get("/models/info", response_model=ModelInfoResponse)
async def models_info():
    """Get loaded models information."""
    return ModelInfoResponse(
        image_model=image_model.model_info if image_model else None,
        server_version="1.0.0",
    )


@app.post("/api/v1/image/segment", response_model=ImageSegmentResponse)
async def segment_image(request: ImageSegmentRequest):
    """Segment image with prompts.

    Supports text prompts, box prompts, point prompts, and combinations.
    """
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image)
        logger.info(f"Processing image of size {image.size}")

        # Extract prompts by type
        text_prompts = []
        box_prompts = []
        point_prompts = []

        for prompt in request.prompts:
            prompt_dict = prompt.model_dump() if hasattr(prompt, 'model_dump') else prompt.dict()
            prompt_type = prompt_dict.get("type", "")
            
            if prompt_type == "text":
                text_prompts.append(prompt_dict["text"])
            elif prompt_type == "box":
                box_prompts.append((prompt_dict["box"], prompt_dict.get("label", True)))
            elif prompt_type == "point":
                point_prompts.append((prompt_dict["points"], prompt_dict["point_labels"]))

        # Ensure at least one prompt
        if not text_prompts and not box_prompts and not point_prompts:
            raise HTTPException(
                status_code=400,
                detail="At least one text, box, or point prompt is required",
            )

        # Segment with combined prompts
        masks, boxes, scores = image_model.segment_combined(
            image=image,
            text_prompts=text_prompts if text_prompts else None,
            boxes=box_prompts if box_prompts else None,
            points=point_prompts if point_prompts else None,
        )

        # Encode masks to base64
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
    """Segment with multiple text prompts using feature caching (faster)."""
    if image_model is None:
        raise HTTPException(status_code=503, detail="Image model not loaded")

    start_time = time.time()

    try:
        # Decode image
        image = decode_base64_image(request.image)

        # Generate cache key
        cache_key = f"cache_{hash(request.image[:100])}"

        # Check if already cached
        cache_hit = cache_key in image_model.feature_cache
        if not cache_hit:
            image_model.cache_features(image, cache_key)

        # Segment with cached features
        results_list = image_model.segment_with_cached_features(
            cache_key, request.text_prompts
        )

        # Format results
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


# ============================================================================
# Legacy endpoints for backward compatibility
# ============================================================================

@app.post("/set_image")
async def set_image_legacy(request: dict):
    """Legacy endpoint: Set image for subsequent predictions."""
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
    """Legacy endpoint: Predict masks using cached image."""
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

        # Get cached image info
        cached = image_model.feature_cache[cache_key]
        image_size = cached["image_size"]

        # Build prompts
        text_prompts = [text_prompt] if text_prompt else None
        box_prompts = None
        if box:
            # Legacy box format is [x1, y1, x2, y2] in pixels
            # Convert to normalized
            orig_w, orig_h = image_size
            normalized_box = [
                box[0] / orig_w,
                box[1] / orig_h,
                box[2] / orig_w,
                box[3] / orig_h,
            ]
            box_prompts = [(normalized_box, True)]

        # Re-decode image from cache backbone (we need full segment_combined)
        # For now, just use text prompts with cached features
        if text_prompts:
            results = image_model.segment_with_cached_features(cache_key, text_prompts)
            masks, boxes, scores = results[0]
        else:
            # For box-only, we need the original image - not ideal but works
            raise HTTPException(
                status_code=400, 
                detail="Box-only prompts require calling /api/v1/image/segment with full image"
            )

        # Encode masks
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


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run the server."""
    parser = argparse.ArgumentParser(description="SAM3 Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "sam_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
