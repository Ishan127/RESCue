import base64
import os
import logging
from openai import OpenAI

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_client(base_url="http://localhost:8000/v1", api_key="EMPTY"):
    return OpenAI(base_url=base_url, api_key=api_key)

def create_vision_message(text_prompt, image_path):
    base64_image = encode_image(image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ]
