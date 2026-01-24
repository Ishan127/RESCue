#!/bin/bash
# Deploy SAM3 cluster with load balancer
# Works both inside and outside tmux - uses background processes

# GPU Assignment - all SAM instances share GPU 3
export CUDA_VISIBLE_DEVICES=3
export HIP_VISIBLE_DEVICES=3

# Configuration
NUM_INSTANCES=${NUM_SAM_INSTANCES:-24}
BACKEND_BASE_PORT=8010
LB_PORT=8001
HOST=${SAM_HOST:-0.0.0.0}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}SAM3 Cluster Deployment${NC}"
echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}Instances: $NUM_INSTANCES${NC}"
echo -e "${YELLOW}Backend ports: $BACKEND_BASE_PORT - $((BACKEND_BASE_PORT + NUM_INSTANCES - 1))${NC}"
echo -e "${YELLOW}Load balancer port: $LB_PORT${NC}"
echo ""

# Kill any existing SAM processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "sam_server.py" 2>/dev/null || true
pkill -f "sam_load_balancer.py" 2>/dev/null || true
sleep 2

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed!${NC}"
    exit 1
}

# Create log directory
mkdir -p /tmp/sam_logs

# Build backend URLs
BACKEND_URLS=""
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BACKEND_BASE_PORT + i))
    if [ -z "$BACKEND_URLS" ]; then
        BACKEND_URLS="http://localhost:$PORT"
    else
        BACKEND_URLS="$BACKEND_URLS,http://localhost:$PORT"
    fi
done

# Start backends in batches using nohup
echo -e "${YELLOW}Starting $NUM_INSTANCES SAM backends...${NC}"
BATCH_SIZE=4
PIDS=()

for batch_start in $(seq 0 $BATCH_SIZE $((NUM_INSTANCES - 1))); do
    batch_end=$((batch_start + BATCH_SIZE - 1))
    if [ $batch_end -ge $NUM_INSTANCES ]; then
        batch_end=$((NUM_INSTANCES - 1))
    fi
    
    echo -e "  Starting backends $batch_start-$batch_end..."
    
    for i in $(seq $batch_start $batch_end); do
        PORT=$((BACKEND_BASE_PORT + i))
        
        # Start backend with nohup
        nohup python src/sam_server.py --host $HOST --port $PORT \
            > /tmp/sam_logs/sam_$PORT.log 2>&1 &
        PIDS+=($!)
    done
    
    # Wait a bit between batches for GPU memory allocation
    sleep 5
done

echo -e "${YELLOW}Backend PIDs: ${PIDS[*]}${NC}"

# Wait for backends to initialize (Active Polling)
echo -e "${YELLOW}Waiting for backends to initialize (Max 600s)...${NC}"
START_TIME=$(date +%s)
MAX_WAIT=600

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ $ELAPSED -gt $MAX_WAIT ]; then
        echo -e "\n${RED}Timeout waiting for backends!${NC}"
        break
    fi

    READY_COUNT=0
    # Check health of all backends
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BACKEND_BASE_PORT + i))
        # Short timeout info check
        if curl -s -m 0.5 "http://localhost:$PORT/health" > /dev/null; then
            ((READY_COUNT++))
        fi
    done

    echo -ne "\r${YELLOW}Status: $READY_COUNT/$NUM_INSTANCES backends ready [Elapsed: ${ELAPSED}s]   ${NC}"
    
    if [ $READY_COUNT -eq $NUM_INSTANCES ]; then
        echo -e "\n${GREEN}All backends successfully initialized!${NC}"
        break
    fi
    
    sleep 5
done
echo ""

if [ $READY_COUNT -lt $NUM_INSTANCES ]; then
    echo -e "${RED}Deployment failed: Only $READY_COUNT/$NUM_INSTANCES backends ready.${NC}"
    # Continue to let the detailed check show WHICH ones failed, or exit?
    # Let's let it continue so the user sees which ports failed in the next block.
fi

# Check backend health
echo -e "${YELLOW}Checking backend health...${NC}"
HEALTHY=0
HEALTHY_URLS=""
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BACKEND_BASE_PORT + i))
    if curl -s --connect-timeout 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
        ((HEALTHY++))
        echo -e "  Port $PORT: ${GREEN}OK${NC}"
        if [ -z "$HEALTHY_URLS" ]; then
            HEALTHY_URLS="http://localhost:$PORT"
        else
            HEALTHY_URLS="$HEALTHY_URLS,http://localhost:$PORT"
        fi
    else
        echo -e "  Port $PORT: ${RED}FAILED${NC}"
    fi
done

echo ""
echo -e "${GREEN}$HEALTHY/$NUM_INSTANCES backends healthy${NC}"

if [ $HEALTHY -eq 0 ]; then
    echo -e "${RED}No backends started successfully!${NC}"
    echo ""
    echo "Last 50 lines from first backend log:"
    tail -50 /tmp/sam_logs/sam_8010.log 2>/dev/null || echo "No log file found"
    exit 1
fi

# Start load balancer with healthy backends only
echo ""
echo -e "${YELLOW}Starting load balancer on port $LB_PORT...${NC}"
nohup python src/sam_load_balancer.py \
    --port $LB_PORT \
    --host $HOST \
    --backends "$HEALTHY_URLS" \
    > /tmp/sam_logs/sam_lb.log 2>&1 &
LB_PID=$!

echo -e "${YELLOW}Load balancer PID: $LB_PID${NC}"
sleep 5

# Verify load balancer
if curl -s --connect-timeout 5 "http://localhost:$LB_PORT/health" > /dev/null 2>&1; then
    echo -e "${GREEN}Load balancer started successfully!${NC}"
else
    echo -e "${RED}Load balancer failed to start!${NC}"
    echo "Last 30 lines from load balancer log:"
    tail -30 /tmp/sam_logs/sam_lb.log 2>/dev/null || echo "No log file"
    exit 1
fi

# Final status
echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}SAM3 Cluster Ready!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo -e "${GREEN}Endpoint: http://localhost:$LB_PORT${NC}"
echo -e "${YELLOW}Health:   curl http://localhost:$LB_PORT/health${NC}"
echo -e "${YELLOW}Stats:    curl http://localhost:$LB_PORT/stats${NC}"
echo ""
echo -e "${YELLOW}Logs: /tmp/sam_logs/${NC}"
echo -e "${YELLOW}  Backend logs: /tmp/sam_logs/sam_80XX.log${NC}"
echo -e "${YELLOW}  LB log: /tmp/sam_logs/sam_lb.log${NC}"
echo ""
echo -e "${GREEN}To stop all: pkill -f sam_server; pkill -f sam_load_balancer${NC}"
echo ""

# Show health
echo "Cluster health:"
curl -s "http://localhost:$LB_PORT/health" | python -m json.tool 2>/dev/null || true
