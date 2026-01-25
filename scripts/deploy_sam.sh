#!/bin/bash
# Deploy Single SAM3 Server
# GPU Assignment
export CUDA_VISIBLE_DEVICES=4
export HIP_VISIBLE_DEVICES=4

PORT=${SAM_PORT:-8001}
HOST=${SAM_HOST:-0.0.0.0}

echo "Starting SAM3 Single Node on $HOST:$PORT..."
echo "GPU: Device 3"

# Kill any existing process on this port or logic
pkill -f "sam_server.py" 2>/dev/null || true
pkill -f "sam_load_balancer.py" 2>/dev/null || true
sleep 1

# Start Server
python src/sam_server.py --host $HOST --port $PORT
