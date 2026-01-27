#!/bin/bash
# ============================================================
# PHASE 4: PRE-COMPUTE VLM SCORES
# Deploy Verifier in tmux session, then compute pointwise scores
# ============================================================

set -e

echo "=============================================="
echo "PHASE 4: PRE-COMPUTE VLM SCORES"
echo "=============================================="

# Step 1: Start Verifier in a tmux session
echo "Step 1: Deploying Verifier on all 8 GPUs (in tmux session)..."

tmux kill-session -t verifier 2>/dev/null || true
tmux new-session -d -s verifier "
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 4096 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0
"

echo "Verifier started in tmux session 'verifier'"
echo "  View logs: tmux attach -t verifier"

# Wait for server to be ready (30B takes longer)
echo ""
echo "Step 2: Waiting for Verifier server to be ready (may take 2-3 min)..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Verifier server ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

# Check if ready
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Verifier server failed to start. Check: tmux attach -t verifier"
    exit 1
fi

# Step 3: Run VLM scoring
echo ""
echo "Step 3: Computing VLM pointwise scores..."
python scripts/precompute_all.py \
    --phase vlm \
    --cache_dir cache \
    --workers 64 \
    --verifier_url http://localhost:8000/v1

# Step 4: Shutdown Verifier
echo ""
echo "Step 4: Shutting down Verifier tmux session..."
tmux kill-session -t verifier 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 4 COMPLETE!"
echo "VLM scores saved to: cache/masks/*/vlm_scores.json"
echo "=============================================="
