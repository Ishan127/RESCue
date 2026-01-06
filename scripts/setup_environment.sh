#!/bin/bash

echo "Setting up RESCue Environment..."

echo "Installing dependencies (requirements.txt)..."
pip install -r requirements.txt

# Force reinstall of torch with ROCm support AFTER requirements to prevent overwrite
echo "Uninstalling any wrong torch/vllm versions..."
pip uninstall -y torch torchvision torchaudio vllm

echo "Installing PyTorch for ROCm..."
# Using ROCm 6.2 wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo "Installing vLLM for ROCm..."
pip install vllm==0.6.3.post1 --extra-index-url https://wheels.vllm.ai/rocm61

echo "Setup Complete."
