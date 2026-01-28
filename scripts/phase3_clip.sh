#!/bin/bash
# ============================================================
# PHASE 3: PRE-COMPUTE CLIP SCORES
# Deploy CLIP in tmux session, then score all masks
# ============================================================

set -e

echo "=============================================="
echo "PHASE 3: PRE-COMPUTE CLIP SCORES"
echo "=============================================="

# Step 1: Start CLIP servers in tmux sessions
echo "Step 1: Deploying 8 CLIP servers (clip0..clip7)..."

# Cleanup old sessions
for i in {0..7}; do
    tmux kill-session -t clip$i 2>/dev/null || true
done

# Launch 8 servers
for i in {0..7}; do
    port=$((8011 + i))
    echo "  Launching clip$i on GPU $i, Port $port..."
    tmux new-session -d -s clip$i "
    export CUDA_VISIBLE_DEVICES=$i
    export HIP_VISIBLE_DEVICES=$i
    python src/clip_server.py --host 0.0.0.0 --port $port
    "
done

echo "CLIP servers deployed."
echo "  View logs: tmux attach -t clip0 (etc)"

# Wait for servers to be ready
echo ""
echo "Step 2: Waiting for CLIP servers to be ready..."
for i in {1..60}; do
    ready_count=0
    for node in {0..7}; do
        port=$((8011 + node))
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            ready_count=$((ready_count + 1))
        fi
    done
    
    echo "Servers ready: $ready_count/8 ($i/60)"
    if [ "$ready_count" -eq 8 ]; then
        echo "All CLIP servers ready!"
        break
    fi
    sleep 5
done

# Check if ready
if [ "$ready_count" -ne 8 ]; then
    echo "ERROR: Not all servers started successfully."
    exit 1
fi

# Step 3: Run CLIP scoring
echo ""
echo "Step 3: Computing CLIP scores (Distributed)..."
python scripts/precompute_all.py \
    --phase clip \
    --cache_dir cache \
    --workers 8 \
    --clip_url http://localhost:8011/verify

# Step 4: Shutdown CLIP
echo ""
echo "Step 4: Shutting down CLIP sessions..."
for i in {0..7}; do
    tmux kill-session -t clip$i 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "PHASE 3 COMPLETE!"
echo "CLIP scores saved to: cache/masks/*/clip_scores.json"
echo "=============================================="
