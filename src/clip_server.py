
import os
import io
import base64
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
from transformers import AutoProcessor, AutoModel
import numpy as np

app = FastAPI(title="SigLIP Verifier Server")

# Model Global
MODEL = None
PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/siglip-so400m-patch14-384"

class VerifyRequest(BaseModel):
    crops: List[str]  # Base64 encoded images
    query: str

class VerifyResponse(BaseModel):
    scores: List[float]

@app.on_event("startup")
async def load_model():
    global MODEL, PROCESSOR
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    try:
        PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)
        MODEL = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        print("SigLIP Model Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Model load failed: {e}")

@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest):
    if not MODEL or not PROCESSOR:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.crops:
        return VerifyResponse(scores=[])
        
    try:
        # Decode images
        pil_images = []
        for b64 in request.crops:
            # Handle data URI prefix if present
            if "," in b64:
                b64 = b64.split(",")[1]
            img_data = base64.b64decode(b64)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            pil_images.append(image)
        
        # Inference
        texts = [request.query] 
        with torch.no_grad():
            inputs = PROCESSOR(text=texts, images=pil_images, padding="max_length", return_tensors="pt").to(DEVICE)
            outputs = MODEL(**inputs)
            
            # SigLIP: Logits are already scaled, just apply sigmoid
            logits_per_image = outputs.logits_per_image # [N_crops, 1_text]
            probs = torch.sigmoid(logits_per_image).cpu().numpy().flatten()
            
        scores = [float(p) * 100.0 for p in probs]
        return VerifyResponse(scores=scores)

    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)
