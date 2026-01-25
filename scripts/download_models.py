import argparse
import os
from huggingface_hub import snapshot_download

def download_models():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not set. Downloads may fail for gated models.")
    
    # Verifier model (32B-Thinking for accurate verification)
    print("Downloading Qwen3-VL-32B-Thinking (Verifier)...")
    try:
        snapshot_download(repo_id="Qwen/Qwen3-VL-32B-Thinking", token=token)
        print("Qwen3-VL-32B-Thinking downloaded successfully.")
    except Exception as e:
        print(f"Error downloading verifier model: {e}")

    # Planner model (8B for fast hypothesis generation)
    print("Downloading Qwen3-VL-8B-Instruct (Planner)...")
    try:
        snapshot_download(repo_id="Qwen/Qwen3-VL-8B-Instruct", token=token)
        print("Qwen3-VL-8B-Instruct downloaded successfully.")
    except Exception as e:
        print(f"Error downloading planner model: {e}")

    # SAM3 for segmentation
    print("Downloading SAM 3 (Executor)...")
    try:
        snapshot_download(repo_id="facebook/sam3", token=token)
        print("SAM 3 downloaded successfully.")
    except Exception as e:
        print(f"Error downloading SAM 3: {e}")

    # SigLIP for CLIP verification
    print("Downloading SigLIP2-Giant (CLIP Verifier)...")
    try:
        snapshot_download(repo_id="google/siglip2-giant-opt-patch16-384", token=token)
        print("SigLIP2-Giant downloaded successfully.")
    except Exception as e:
        print(f"Error downloading SigLIP: {e}")

if __name__ == "__main__":
    download_models()
