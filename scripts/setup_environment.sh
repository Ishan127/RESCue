#!/bin/bash

echo "Setting up RESCue Environment..."

# Force reinstall of torch with ROCm support
echo "Uninstalling existing torch/vllm to ensure clean slate..."
pip uninstall -y torch torchvision torchaudio vllm

echo "Installing PyTorch for ROCm..."
# Using ROCm 6.2 wheels which are generally compatible with 6.3 host driver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo "Installing vLLM for ROCm..."
pip install vllm --extra-index-url https://wheels.vllm.ai/rocm61

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup Complete."
