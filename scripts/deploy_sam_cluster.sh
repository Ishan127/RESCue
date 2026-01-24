#!/bin/bash
# Deploy 16 SAM3 instances with a single load balancer on port 8001
# Backend instances run on ports 8010-8025
# Load balancer exposes single endpoint on port 8001

set -e

# GPU Assignment - all SAM instances share GPU 3
export CUDA_VISIBLE_DEVICES=3

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
NUM_INSTANCES=${NUM_SAM_INSTANCES:-16}
BACKEND_BASE_PORT=8010
LB_PORT=8001
HOST=${SAM_HOST:-0.0.0.0}

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}SAM3 Cluster Deployment${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}Instances: $NUM_INSTANCES${NC}"
echo -e "${YELLOW}Backend ports: $BACKEND_BASE_PORT - $((BACKEND_BASE_PORT + NUM_INSTANCES - 1))${NC}"
echo -e "${YELLOW}Load balancer port: $LB_PORT${NC}"
echo -e "${YELLOW}GPU Memory: ~$((NUM_INSTANCES * 10))GB${NC}"
echo ""

# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check SAM3 installation
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 library: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed. Please install sam3 package.${NC}"
    exit 1
}

# Start backend instances
PIDS=()
BACKEND_URLS=""

echo -e "${YELLOW}Starting $NUM_INSTANCES SAM backend instances...${NC}"

for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BACKEND_BASE_PORT + i))
    
    echo -e "  Starting backend $i on port $PORT..."
    
    python src/sam_server.py \
        --host $HOST \
        --port $PORT \
        > /tmp/sam_backend_$PORT.log 2>&1 &
    
    PIDS+=($!)
    
    if [ -z "$BACKEND_URLS" ]; then
        BACKEND_URLS="http://localhost:$PORT"
    else
        BACKEND_URLS="$BACKEND_URLS,http://localhost:$PORT"
    fi
    
    # Short delay to stagger startup
    sleep 2
done

echo ""
echo -e "${YELLOW}Waiting for backends to initialize (30s)...${NC}"
sleep 30

# Check how many backends are healthy
HEALTHY=0
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BACKEND_BASE_PORT + i))
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        ((HEALTHY++))
    fi
done
echo -e "${GREEN}$HEALTHY/$NUM_INSTANCES backends healthy${NC}"

# Start load balancer
echo ""
echo -e "${YELLOW}Starting load balancer on port $LB_PORT...${NC}"

python src/sam_load_balancer.py \
    --port $LB_PORT \
    --host $HOST \
    --backends "$BACKEND_URLS" &

LB_PID=$!

sleep 3

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}SAM3 Cluster Ready!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${GREEN}Single endpoint: http://localhost:$LB_PORT${NC}"
echo -e "${YELLOW}Health check: curl http://localhost:$LB_PORT/health${NC}"
echo -e "${YELLOW}Stats: curl http://localhost:$LB_PORT/stats${NC}"
echo ""
echo -e "${GREEN}For evaluation, use:${NC}"
echo -e "${GREEN}  --executor_urls \"http://localhost:$LB_PORT\"${NC}"
echo ""
echo -e "${YELLOW}Backend PIDs: ${PIDS[@]}${NC}"
echo -e "${YELLOW}Load Balancer PID: $LB_PID${NC}"
echo ""
echo -e "${YELLOW}To stop all: pkill -f sam_server; pkill -f sam_load_balancer${NC}"

# Wait for load balancer (keeps script running)
wait $LB_PID
