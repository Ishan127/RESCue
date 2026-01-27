#!/bin/bash
# ============================================================
# PHASE 2: PRE-COMPUTE MASKS
# Deploy SAM in tmux session, then generate masks
# ============================================================

set -e

echo "=============================================="
echo "PHASE 2: PRE-COMPUTE MASKS"
echo "=============================================="

# Step 1: Start SAM in a tmux session
echo "Step 1: Deploying SAM server (in tmux session)..."

tmux kill-session -t sam 2>/dev/null || true
tmux new-session -d -s sam "
export CUDA_VISIBLE_DEVICES=6
export HIP_VISIBLE_DEVICES=6
python src/sam_server.py --host 0.0.0.0 --port 8001
"

echo "SAM started in tmux session 'sam'"
echo "  View logs: tmux attach -t sam"

# Wait for server to be ready
echo ""
echo "Step 2: Waiting for SAM server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "SAM server ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 5
done

# Check if ready
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    echo "ERROR: SAM server failed to start. Check: tmux attach -t sam"
    exit 1
fi

# Step 3: Run mask generation
echo ""
echo "Step 3: Generating masks..."
python scripts/precompute_all.py \
    --phase masks \
    --cache_dir cache \
    --workers 16 \
    --sam_url http://localhost:8001

# Step 4: Shutdown SAM
echo ""
echo "Step 4: Shutting down SAM tmux session..."
tmux kill-session -t sam 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 2 COMPLETE!"
echo "Masks saved to: cache/masks/"
echo "=============================================="
