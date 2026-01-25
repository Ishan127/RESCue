#!/bin/bash
# Deploy Qwen3-VL-30B-A3B-Instruct (VERIFIER) on GPUs 0,1,2,3
# MoE architecture: 30B total, only 3B active - fast inference
# Port: 8000

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 1024 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0
