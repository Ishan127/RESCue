#!/bin/bash
# Deploy Qwen3-VL-8B-Instruct (PLANNER) on GPU 5
# Fast hypothesis generation for bounding box proposals
# Port: 8002

export CUDA_VISIBLE_DEVICES=5

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --port 8002 \
    --host 0.0.0.0
