import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
import numpy as np
from PIL import Image
import torch
import sys
import os
import argparse

# Add parent directory to path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.executor import Executor
from src.utils import get_device

app = FastAPI(title="SAM3 Server")

# Global executor instance
executor = None

class SegmentationRequest(BaseModel):
    image_base64: str
    box: Optional[List[float]] = None # [x1, y1, x2, y2]
    text_prompt: Optional[str] = None

class SegmentationResponse(BaseModel):
    masks_rle: List[dict] # simplified, or just return list of lists or base64 masks
    masks_base64: List[str]

def init_model():
    global executor
    device = get_device()
    executor = Executor(model_path="facebook/sam3", device=device)

@app.on_event("startup")
async def startup_event():
    init_model()

def mask_to_base64(mask_np):
    # mask_np is bool or uint8
    img = Image.fromarray((mask_np * 255).astype(np.uint8))
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def decode_base64_image(b64_string):
    img_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

import traceback

@app.post("/segment", response_model=SegmentationResponse)
async def segment(request: SegmentationRequest):
    if not executor:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        image = decode_base64_image(request.image_base64)
        image_np = np.array(image)
        
        # Executor expects: image_input, box, text_prompt
        print(f"Received request. Box: {request.box}, Prompt: {request.text_prompt}")
        masks = executor.execute(image_np, request.box, request.text_prompt)
        
        masks_b64 = []
        for mask in masks:
            masks_b64.append(mask_to_base64(mask))
            
        return SegmentationResponse(masks_rle=[], masks_base64=masks_b64)
    except Exception as e:
        print(f"Error during segmentation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
