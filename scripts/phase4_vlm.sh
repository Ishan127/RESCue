#!/bin/bash
# ============================================================
# PHASE 4: PRE-COMPUTE VLM SCORES
# Deploy Verifier on ALL 8 GPUs, then compute pointwise scores
# ============================================================

set -e

echo "=============================================="
echo "PHASE 4: PRE-COMPUTE VLM SCORES"
echo "=============================================="

# Step 1: Deploy Verifier on 8 GPUs
echo "Step 1: Deploying Verifier on all 8 GPUs..."
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
    --host 0.0.0.0 &

VERIFIER_PID=$!
echo "Verifier PID: $VERIFIER_PID"

# Wait for server to be ready
echo "Waiting for Verifier server to be ready..."
sleep 120  # 30B model takes longer to load
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Verifier server ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 10
done

# Step 2: Run VLM scoring
echo ""
echo "Step 2: Computing VLM pointwise scores..."
python scripts/precompute_all.py \
    --phase vlm \
    --cache_dir cache \
    --workers 64 \
    --verifier_url http://localhost:8000/v1

# Step 3: Shutdown Verifier
echo ""
echo "Step 3: Shutting down Verifier..."
kill $VERIFIER_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 4 COMPLETE!"
echo "VLM scores saved to: cache/masks/*/vlm_scores.json"
echo "=============================================="
