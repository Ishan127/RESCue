#!/bin/bash
# Deploy Qwen2.5-VL-7B (PLANNER) on GPU 2
# Fast hypothesis generation for box coordinates and query variations
# Port: 8002

export CUDA_VISIBLE_DEVICES=2

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --max-num-seqs 32 \
    --port 8002 \
    --host 0.0.0.0
