#!/bin/bash
# ============================================================
# PHASE 2: PRE-COMPUTE MASKS
# Deploy SAM on GPU 6 (or more), then generate masks for all hypotheses
# ============================================================

set -e

echo "=============================================="
echo "PHASE 2: PRE-COMPUTE MASKS"
echo "=============================================="

# Step 1: Deploy SAM server
echo "Step 1: Deploying SAM server..."
export CUDA_VISIBLE_DEVICES=6
export HIP_VISIBLE_DEVICES=6

python src/sam_server.py --host 0.0.0.0 --port 8001 &
SAM_PID=$!
echo "SAM PID: $SAM_PID"

# Wait for server to be ready
echo "Waiting for SAM server to be ready..."
sleep 30
for i in {1..20}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "SAM server ready!"
        break
    fi
    echo "Waiting... ($i/20)"
    sleep 5
done

# Step 2: Run mask generation
echo ""
echo "Step 2: Generating masks..."
python scripts/precompute_all.py \
    --phase masks \
    --cache_dir cache \
    --workers 16 \
    --sam_url http://localhost:8001

# Step 3: Shutdown SAM
echo ""
echo "Step 3: Shutting down SAM..."
kill $SAM_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 2 COMPLETE!"
echo "Masks saved to: cache/masks/"
echo "=============================================="
