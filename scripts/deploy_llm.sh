#!/bin/bash
# Deploy Qwen3-VL-32B-Thinking using vLLM on 2 GPUs
# This model uses chain-of-thought reasoning for better verification
# Adjust --tensor-parallel-size if needed (e.g. 2 for 2 GPUs)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-32B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 40960 \
    --port 8000 \
    --host 0.0.0.0
