#!/bin/bash
# Deploy Qwen3-VL-30B-A3B-Instruct using vLLM on 2 GPUs
# Adjust --tensor-parallel-size if needed (e.g. 2 for 2 GPUs)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 8000 \
    --host 0.0.0.0
