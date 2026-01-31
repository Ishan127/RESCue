#!/bin/bash
# ============================================================
# PHASE 4: PRE-COMPUTE VLM SCORES (2-GPU VERSION)
# Deploy Verifier in tmux session, then compute pointwise scores
# ============================================================

set -e

echo "=============================================="
echo "PHASE 4: PRE-COMPUTE VLM SCORES (2 GPU)"
echo "=============================================="

# Step 1: Start 1 Verifier with TP=2 on 2 GPUs
if curl -s http://localhost:8000/health >/dev/null; then
    echo "Step 1: Verifier already running. Skipping deployment."
else
    echo "Step 1: Deploying Verifier on 2 GPUs (TP=2)..."

    tmux kill-session -t verifier0 2>/dev/null || true

    echo "  Launching verifier0 (GPUs 0,1)..."
    tmux new-session -d -s verifier0 "
    export CUDA_VISIBLE_DEVICES=0,1
    export HIP_VISIBLE_DEVICES=0,1
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-VL-30B-A3B-Thinking \
        --trust-remote-code \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 8192 \
        --max-num-seqs 2048 \
        --dtype bfloat16 \
        --enable-prefix-caching \
        --port 8000 \
        --host 0.0.0.0
    "

    echo "Verifier deployed."

    # Wait for server to be ready
    echo ""
    echo "Step 2: Waiting for Verifier to be ready (may take 2-3 min)..."
    for i in {1..90}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Verifier ready! ($i attempts)"
            break
        fi
        echo "Waiting... ($i/90)"
        sleep 5
    done
fi

# Step 3: Run VLM scoring
echo ""
echo "Step 3: Computing VLM pointwise scores..."
ulimit -n 65536 2>/dev/null || true

python scripts/precompute_all.py \
    --phase vlm \
    --cache_dir cache \
    --workers 16 \
    --verifier_url "http://localhost:8000/v1"

# Step 4: Shutdown Verifiers - DISABLED for persistence
echo ""
echo "Step 4: Keeping Verifier alive for future runs..."

echo ""
echo "=============================================="
echo "PHASE 4 COMPLETE!"
echo "VLM scores saved to: cache/masks/*/vlm_scores.json"
echo "=============================================="
