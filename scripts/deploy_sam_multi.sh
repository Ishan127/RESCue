#!/bin/bash
# Deploy multiple SAM3 instances for parallel processing
# Uses GPU 3 with multiple server instances on different ports
# Ports: 8001, 8003, 8004, 8005

set -e

# GPU Assignment - all SAM instances share GPU 3
export CUDA_VISIBLE_DEVICES=3

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Number of SAM instances
NUM_INSTANCES=${NUM_SAM_INSTANCES:-4}
BASE_PORT=8001
HOST=${SAM_HOST:-0.0.0.0}

echo -e "${GREEN}Starting $NUM_INSTANCES SAM3 Server Instances...${NC}"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check SAM3 installation
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 library: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed. Please install sam3 package.${NC}"
    exit 1
}

# Start multiple instances
PIDS=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    if [ $i -eq 0 ]; then
        PORT=$BASE_PORT
    else
        PORT=$((BASE_PORT + i + 1))  # 8001, 8003, 8004, 8005...
    fi
    
    echo -e "${YELLOW}Starting SAM instance $i on port $PORT...${NC}"
    
    python -m src.sam_server \
        --host $HOST \
        --port $PORT &
    
    PIDS+=($!)
    
    # Small delay between starts
    sleep 2
done

echo -e "${GREEN}Started ${#PIDS[@]} SAM instances on ports: 8001, 8003, 8004, 8005${NC}"
echo -e "${YELLOW}PIDs: ${PIDS[@]}${NC}"

# Wait for all
wait
