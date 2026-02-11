#!/bin/bash
# Deploy Qwen3-VL-32B-Thinking (VERIFIER) on GPUs 0,1,2,3
# Chain-of-thought reasoning for accurate mask verification
# Port: 8000

# Instance 1: GPUs 0-3, Port 8000
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --structured-outputs-config '{"backend": "outlines"}' \
    --port 8000 \
    --host 0.0.0.0 &

# Instance 2: GPUs 4-7, Port 8004
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 8192 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --structured-outputs-config '{"backend": "outlines"}' \
    --port 8004 \
    --host 0.0.0.0 &

wait
