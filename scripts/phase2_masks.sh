#!/bin/bash
# ============================================================
# PHASE 2: PRE-COMPUTE MASKS
# Deploy SAM in tmux session, then generate masks
# ============================================================

set -e

echo "=============================================="
echo "PHASE 2: PRE-COMPUTE MASKS"
echo "=============================================="

# Step 1: Start SAM servers in tmux sessions
echo "Step 1: Deploying 8 SAM servers (sam0..sam7)..."

# Cleanup old sessions
for i in {0..7}; do
    tmux kill-session -t sam$i 2>/dev/null || true
done

# Launch 8 servers
for i in {0..7}; do
    port=$((8001 + i))
    echo "  Launching sam$i on GPU $i, Port $port..."
    tmux new-session -d -s sam$i "
    export CUDA_VISIBLE_DEVICES=$i
    export HIP_VISIBLE_DEVICES=$i
    python src/sam_server.py --host 0.0.0.0 --port $port
    "
done

echo "SAM servers deployed."
echo "  View logs: tmux attach -t sam0 (etc)"

# Wait for servers to be ready
echo ""
echo "Step 2: Waiting for SAM servers to be ready..."
for i in {1..60}; do
    ready_count=0
    for node in {0..7}; do
        port=$((8001 + node))
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            ready_count=$((ready_count + 1))
        fi
    done
    
    echo "Servers ready: $ready_count/8 ($i/60)"
    if [ "$ready_count" -eq 8 ]; then
        echo "All SAM servers ready!"
        break
    fi
    sleep 5
done

# Check if ready
if [ "$ready_count" -ne 8 ]; then
    echo "ERROR: Not all servers started successfully."
    exit 1
fi

# Step 3: Run mask generation
echo ""
echo "Step 3: Generating masks (Batch Mode: 10 calls/image, Distributed)..."
py scripts/precompute_all.py \
    --phase masks \
    --cache_dir cache \
    --workers 8 \
    --sam_url http://localhost:8001 \
    --max_n 256

# Step 4: Shutdown SAM
echo ""
echo "Step 4: Shutting down SAM sessions..."
for i in {0..7}; do
    tmux kill-session -t sam$i 2>/dev/null || true
done

echo ""
echo "=============================================="
echo "PHASE 2 COMPLETE!"
echo "Masks saved to: cache/masks/"
echo "=============================================="
