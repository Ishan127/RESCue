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

# Number of SAM instances - with 256GB GPU and ~10GB per instance, can run many!
NUM_INSTANCES=${NUM_SAM_INSTANCES:-8}
BASE_PORT=8001
HOST=${SAM_HOST:-0.0.0.0}

echo -e "${GREEN}Starting $NUM_INSTANCES SAM3 Server Instances on GPU 3 (256GB)...${NC}"
echo -e "${YELLOW}Each instance uses ~10GB, total: ~$((NUM_INSTANCES * 10))GB${NC}"

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check SAM3 installation
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 library: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed. Please install sam3 package.${NC}"
    exit 1
}

# Start multiple instances
PIDS=()
PORTS=()

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))  # 8001, 8002, 8003, ...
    PORTS+=($PORT)
    
    echo -e "${YELLOW}Starting SAM instance $i on port $PORT...${NC}"
    
    python src/sam_server.py \
        --host $HOST \
        --port $PORT &
    
    PIDS+=($!)
    
    # Shorter delay since we have plenty of GPU memory
    sleep 3
done

echo ""
echo -e "${GREEN}Started ${#PIDS[@]} SAM instances${NC}"
echo -e "${GREEN}Ports: ${PORTS[*]}${NC}"
echo ""
echo -e "${YELLOW}For evaluation, use:${NC}"
URLS=$(printf "http://localhost:%s," "${PORTS[@]}")
URLS=${URLS%,}  # Remove trailing comma
echo -e "${GREEN}--executor_urls \"$URLS\"${NC}"

# Wait for all
wait
