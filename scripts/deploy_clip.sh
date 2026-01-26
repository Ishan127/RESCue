#!/bin/bash
# Deploy SigLIP Verifier Server
# Port: 8003
# GPU Assignment: Same as SAM (Device 4) or 5 if separate
# User requested running on same GPU as SAM

export CUDA_VISIBLE_DEVICES=7
export HIP_VISIBLE_DEVICES=7

PORT=8003
HOST=0.0.0.0

echo "Starting SigLIP Verifier Server on $HOST:$PORT..."
echo "GPU: Device 7"

# Install deps if needed (usually assumed present)
# pip install uvicorn fastapi

# Kill existing
pkill -f "src/clip_server.py" 2>/dev/null || true
sleep 1

# Start Server
python src/clip_server.py --host $HOST --port $PORT
