#!/bin/bash
# ============================================================
# PHASE 1: PRE-COMPUTE PLANS
# Deploy Planner on ALL 8 GPUs, then generate 512 plans per sample
# ============================================================

set -e

echo "=============================================="
echo "PHASE 1: PRE-COMPUTE PLANS"
echo "=============================================="

# Step 1: Deploy Planner on 8 GPUs
echo "Step 1: Deploying Planner on all 8 GPUs..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 4096 \
    --enable-prefix-caching \
    --port 8002 \
    --host 0.0.0.0 &

PLANNER_PID=$!
echo "Planner PID: $PLANNER_PID"

# Wait for server to be ready
echo "Waiting for Planner server to be ready..."
sleep 60
for i in {1..30}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "Planner server ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 10
done

# Step 2: Run plan generation
echo ""
echo "Step 2: Generating plans..."
python scripts/precompute_all.py \
    --phase plans \
    --cache_dir cache \
    --max_n 512 \
    --workers 64 \
    --planner_url http://localhost:8002/v1

# Step 3: Shutdown planner
echo ""
echo "Step 3: Shutting down Planner..."
kill $PLANNER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 1 COMPLETE!"
echo "Plans saved to: cache/plans/plans.json"
echo "=============================================="
