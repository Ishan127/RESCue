#!/bin/bash
# Deploy Qwen3-VL-32B-Thinking (VERIFIER) on GPUs 0,1,2,3
# Chain-of-thought reasoning for accurate mask verification
# Port: 8000

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 2048 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0
