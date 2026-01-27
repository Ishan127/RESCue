#!/bin/bash
# ============================================================
# PHASE 1: PRE-COMPUTE PLANS
# Deploy Planner in tmux session, then generate 512 plans per sample
# ============================================================

set -e

echo "=============================================="
echo "PHASE 1: PRE-COMPUTE PLANS"
echo "=============================================="

# Step 1: Start Planner in a tmux session
echo "Step 1: Deploying Planner on all 8 GPUs (in tmux session)..."

tmux kill-session -t planner 2>/dev/null || true
tmux new-session -d -s planner "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 4096 \
    --enable-prefix-caching \
    --port 8002 \
    --host 0.0.0.0
"

echo "Planner started in tmux session 'planner'"
echo "  View logs: tmux attach -t planner"

# Wait for server to be ready
echo ""
echo "Step 2: Waiting for Planner server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8002/health > /dev/null 2>&1; then
        echo "Planner server ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

# Check if ready
if ! curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "ERROR: Planner server failed to start. Check: tmux attach -t planner"
    exit 1
fi

# Step 3: Run plan generation
echo ""
echo "Step 3: Generating plans..."
python scripts/precompute_all.py \
    --phase plans \
    --cache_dir cache \
    --max_n 256 \
    --workers 4 \
    --planner_url http://localhost:8002/v1

# Step 4: Shutdown planner
echo ""
echo "Step 4: Shutting down Planner tmux session..."
tmux kill-session -t planner 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 1 COMPLETE!"
echo "Plans saved to: cache/plans/plans.json"
echo "=============================================="
