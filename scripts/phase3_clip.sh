#!/bin/bash
# ============================================================
# PHASE 3: PRE-COMPUTE CLIP SCORES
# Deploy CLIP in tmux session, then score all masks
# ============================================================

set -e

echo "=============================================="
echo "PHASE 3: PRE-COMPUTE CLIP SCORES"
echo "=============================================="

# Step 1: Start CLIP in a tmux session
echo "Step 1: Deploying CLIP server (in tmux session)..."

tmux kill-session -t clip 2>/dev/null || true
tmux new-session -d -s clip "
export CUDA_VISIBLE_DEVICES=7
export HIP_VISIBLE_DEVICES=7
python src/clip_server.py --host 0.0.0.0 --port 8003
"

echo "CLIP started in tmux session 'clip'"
echo "  View logs: tmux attach -t clip"

# Wait for server to be ready
echo ""
echo "Step 2: Waiting for CLIP server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8003/health > /dev/null 2>&1; then
        echo "CLIP server ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 5
done

# Check if ready
if ! curl -s http://localhost:8003/health > /dev/null 2>&1; then
    echo "ERROR: CLIP server failed to start. Check: tmux attach -t clip"
    exit 1
fi

# Step 3: Run CLIP scoring
echo ""
echo "Step 3: Computing CLIP scores..."
python scripts/precompute_all.py \
    --phase clip \
    --cache_dir cache \
    --workers 32 \
    --clip_url http://localhost:8003/verify

# Step 4: Shutdown CLIP
echo ""
echo "Step 4: Shutting down CLIP tmux session..."
tmux kill-session -t clip 2>/dev/null || true

echo ""
echo "=============================================="
echo "PHASE 3 COMPLETE!"
echo "CLIP scores saved to: cache/masks/*/clip_scores.json"
echo "=============================================="
