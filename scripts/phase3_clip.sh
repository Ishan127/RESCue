#!/bin/bash
# ============================================================
# PHASE 3: PRE-COMPUTE CLIP SCORES
# Deploy CLIP server, then score all masks
# ============================================================

set -e

echo "=============================================="
echo "PHASE 3: PRE-COMPUTE CLIP SCORES"
echo "=============================================="

# Step 1: Deploy CLIP server
echo "Step 1: Deploying CLIP server..."
export CUDA_VISIBLE_DEVICES=7
export HIP_VISIBLE_DEVICES=7

python src/clip_server.py --host 0.0.0.0 --port 8003 &
CLIP_PID=$!
echo "CLIP PID: $CLIP_PID"

# Wait for server to be ready
echo "Waiting for CLIP server to be ready..."
sleep 30
for i in {1..20}; do
    if curl -s http://localhost:8003/health > /dev/null 2>&1; then
        echo "CLIP server ready!"
        break
    fi
    echo "Waiting... ($i/20)"
    sleep 5
done

# Step 2: Run CLIP scoring
echo ""
echo "Step 2: Computing CLIP scores..."
python scripts/precompute_all.py \
    --phase clip \
    --cache_dir cache \
    --workers 32 \
    --clip_url http://localhost:8003/verify

# Step 3: Shutdown CLIP
echo ""
echo "Step 3: Shutting down CLIP..."
kill $CLIP_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 3 COMPLETE!"
echo "CLIP scores saved to: cache/masks/*/clip_scores.json"
echo "=============================================="
