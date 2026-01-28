#!/bin/bash
# ============================================================
# PHASE 4: PRE-COMPUTE VLM SCORES
# Deploy Verifier in tmux session, then compute pointwise scores
# ============================================================

set -e

echo "=============================================="
echo "PHASE 4: PRE-COMPUTE VLM SCORES"
echo "=============================================="

# Step 1: Start 2 Verifiers in tmux sessions (Split 4 GPUs each)
echo "Step 1: Deploying 2 Verifiers (verifier0, verifier1) on 4 GPUs each..."

tmux kill-session -t verifier0 2>/dev/null || true
tmux kill-session -t verifier1 2>/dev/null || true

# Verifier 0: GPUs 0-3, Port 8000
echo "  Launching verifier0 (GPUs 0-3)..."
tmux new-session -d -s verifier0 "
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HIP_VISIBLE_DEVICES=0,1,2,3
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 4096 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --port 8000 \
    --host 0.0.0.0
"

# Verifier 1: GPUs 4-7, Port 8001
echo "  Launching verifier1 (GPUs 4-7)..."
tmux new-session -d -s verifier1 "
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HIP_VISIBLE_DEVICES=4,5,6,7
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-30B-A3B-Thinking \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 4096 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --port 8001 \
    --host 0.0.0.0
"

echo "Verifiers deployed."
echo "  View logs: tmux attach -t verifier0 (or verifier1)"

# Wait for servers to be ready
echo ""
echo "Step 2: Waiting for Verifiers to be ready (may take 2-3 min)..."
for i in {1..90}; do
    count=0
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then count=$((count+1)); fi
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then count=$((count+1)); fi
    
    echo "Ready: $count/2 ($i/90)"
    if [ "$count" -eq 2 ]; then
        echo "All Verifiers ready!"
        break
    fi
    sleep 5
done

# Check if ready
if [ "$count" -lt 2 ]; then
    echo "ERROR: Not all Verifiers started. Check logs."
    exit 1
fi

# Step 3: Run VLM scoring
echo ""
echo "Step 3: Computing VLM pointwise scores (Split Load)..."
# Increase file limit for high concurrency
ulimit -n 65536 2>/dev/null || true

python scripts/precompute_all.py \
    --phase vlm \
    --cache_dir cache \
    --workers 256 \
    --verifier_url "http://localhost:8000/v1,http://localhost:8001/v1"

# Step 4: Shutdown Verifiers
echo ""
echo "Step 4: Shutting down Verifiers..."
tmux kill-session -t verifier0 2>/dev/null || true
tmux kill-session -t verifier1 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 4 COMPLETE!"
echo "VLM scores saved to: cache/masks/*/vlm_scores.json"
echo "=============================================="
