import argparse
import os
from huggingface_hub import snapshot_download

def download_models():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set. Downloads may fail for gated models.")
    print("Downloading Qwen2.5-VL-72B-Instruct...")
    try:
        snapshot_download(repo_id="Qwen/Qwen2.5-VL-72B-Instruct", token=token)
        print("Qwen2.5-VL downloaded successfully.")
    except Exception as e:
        print(f"Error downloading Qwen: {e}")

    print("Downloading SAM 3...")
    try:
        snapshot_download(repo_id="facebook/sam3", token=token)
        print("SAM 3 downloaded successfully.")
    except Exception as e:
        print(f"Error downloading SAM 3: {e}")

if __name__ == "__main__":
    download_models()
