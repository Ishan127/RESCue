#!/bin/bash
# Deploy Qwen3-VL-8B (PLANNER) on GPU 2
# Fast hypothesis generation for box coordinates and query variations
# Port: 8002

export CUDA_VISIBLE_DEVICES=2

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --port 8002 \
    --host 0.0.0.0
