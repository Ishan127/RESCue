#!/bin/bash

echo "Setting up RESCue Environment..."

echo "Installing dependencies (requirements.txt)..."
pip install -r requirements.txt

# Force reinstall of torch with ROCm support AFTER requirements to prevent overwrite
echo "Uninstalling any wrong torch/vllm/amdsmi versions..."
pip uninstall -y torch torchvision torchaudio vllm amdsmi

echo "Installing PyTorch for ROCm (v2.4.0)..."
# Explicitly pin version to match vLLM requirements and prevent drift
pip install --no-cache-dir torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/rocm6.1

echo "Installing vLLM for ROCm..."
pip install --no-cache-dir vllm==0.6.3.post1 --extra-index-url https://wheels.vllm.ai/rocm61

echo "Setup Complete."
