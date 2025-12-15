#!/bin/bash

echo "Cleaning up Python environment..."
echo "Uninstalling torch, torchvision, torchaudio, and vllm (NVIDIA versions)..."

pip uninstall -y torch torchvision torchaudio vllm
pip uninstall -y torch torchvision torchaudio vllm

echo "Cleanup complete."
echo "Please now verify your environment."
echo "If you have a pre-installed ROCm environment, the system packages should now be visible."
echo "If not, please install the ROCm-specific versions manually."
echo "Example (ROCm 6.1/6.2):"
echo "  pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.2"
