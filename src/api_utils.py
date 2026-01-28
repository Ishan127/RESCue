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
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "Qwen/Qwen3-VL-30B-A3B-Thinking")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_client(base_url="http://localhost:8000/v1", api_key="EMPTY"):
    return OpenAI(base_url=base_url, api_key=api_key, timeout=600.0)

def get_planner_client():
    """Get client for the fast planning model (8B)."""
    return OpenAI(base_url=PLANNER_API_BASE, api_key="EMPTY", timeout=600.0)

def get_verifier_client():
    """Get client for the verification model (30B-A3B MoE)."""
    return OpenAI(base_url=VERIFIER_API_BASE, api_key="EMPTY", timeout=600.0)

def encode_pil_image(image):
    import io
    # OPTIMIZATION: Resize large images to reduce VLM token count (Speedup)
    max_dim = 1024
    w, h = image.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    buffer = io.BytesIO()
    # JPEG is much faster to encode/decode than alternatives and good enough for vision
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_vision_message(text_prompt, image_path=None, base64_image=None, image=None, image_first=True):
    if base64_image is None:
        if image is not None:
             base64_image = encode_pil_image(image)
             mime_type = 'image/jpeg'
        elif image_path is not None:
            base64_image = encode_image(image_path)
            # Detect image type from file extension
            ext = image_path.lower().split('.')[-1] if '.' in image_path else 'jpeg'
            mime_type = {
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp',
                'bmp': 'image/bmp',
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
            }.get(ext, 'image/jpeg')
        else:
            raise ValueError("Either image (PIL), image_path, or base64_image must be provided")
    else:
        mime_type = 'image/jpeg' # Default assumption
    
    text_content = {"type": "text", "text": text_prompt}
    image_content = {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{base64_image}"
        },
    }
    
    content = [image_content, text_content] if image_first else [text_content, image_content]

    return [
        {
            "role": "user",
            "content": content
        }
    ]
