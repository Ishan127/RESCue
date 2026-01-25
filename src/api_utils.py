import base64
import os
import logging
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Default endpoints for dual-model setup
VERIFIER_API_BASE = os.environ.get("VERIFIER_API_BASE", "http://localhost:8000/v1")
PLANNER_API_BASE = os.environ.get("PLANNER_API_BASE", "http://localhost:8002/v1")

# Default model paths
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_client(base_url="http://localhost:8000/v1", api_key="EMPTY"):
    return OpenAI(base_url=base_url, api_key=api_key)

def get_planner_client():
    """Get client for the fast planning model (8B)."""
    return OpenAI(base_url=PLANNER_API_BASE, api_key="EMPTY")

def get_verifier_client():
    """Get client for the verification model (30B-A3B MoE)."""
    return OpenAI(base_url=VERIFIER_API_BASE, api_key="EMPTY")

def create_vision_message(text_prompt, image_path=None, base64_image=None):
    if base64_image is None:
        if image_path is None:
            raise ValueError("Either image_path or base64_image must be provided")
        base64_image = encode_image(image_path)
        # Detect image type from file extension
        ext = image_path.lower().split('.')[-1] if '.' in image_path else 'jpeg'
    else:
        ext = 'jpeg' # Default for in-memory
        
    mime_type = {
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'bmp': 'image/bmp',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
    }.get(ext, 'image/jpeg')
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    },
                },
            ],
        }
    ]
