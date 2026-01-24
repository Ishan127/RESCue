#!/bin/bash
# Deploy Qwen3-VL-32B-Thinking (VERIFIER) on GPUs 0,1
# Chain-of-thought reasoning for accurate mask verification
# Port: 8000

export CUDA_VISIBLE_DEVICES=0,1

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-32B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 40960 \
    --max-num-seqs 64 \
    --port 8000 \
    --host 0.0.0.0
