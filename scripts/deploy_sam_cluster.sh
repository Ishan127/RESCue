#!/bin/bash
# Deploy SAM3 cluster with load balancer
# Uses tmux sessions to keep processes running

set -e

# GPU Assignment - all SAM instances share GPU 3
export CUDA_VISIBLE_DEVICES=3
export HIP_VISIBLE_DEVICES=3

# Configuration
NUM_INSTANCES=${NUM_SAM_INSTANCES:-16}
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
tmux kill-session -t sam_cluster 2>/dev/null || true
sleep 2

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed!${NC}"
    exit 1
}

# Create tmux session
echo -e "${YELLOW}Creating tmux session 'sam_cluster'...${NC}"
tmux new-session -d -s sam_cluster -n "control"

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

# Start backends in batches (4 at a time to avoid overwhelming GPU)
echo -e "${YELLOW}Starting $NUM_INSTANCES SAM backends...${NC}"
BATCH_SIZE=4

for batch_start in $(seq 0 $BATCH_SIZE $((NUM_INSTANCES - 1))); do
    batch_end=$((batch_start + BATCH_SIZE - 1))
    if [ $batch_end -ge $NUM_INSTANCES ]; then
        batch_end=$((NUM_INSTANCES - 1))
    fi
    
    echo -e "  Starting backends $batch_start-$batch_end..."
    
    for i in $(seq $batch_start $batch_end); do
        PORT=$((BACKEND_BASE_PORT + i))
        
        # Create a new tmux window for each backend
        tmux new-window -t sam_cluster -n "sam_$PORT" \
            "CUDA_VISIBLE_DEVICES=3 HIP_VISIBLE_DEVICES=3 python src/sam_server.py --host $HOST --port $PORT 2>&1 | tee /tmp/sam_$PORT.log; echo 'Process exited, press enter to close'; read"
    done
    
    # Wait a bit between batches
    sleep 5
done

# Wait for backends to initialize
echo -e "${YELLOW}Waiting for backends to load models (60s)...${NC}"
for i in $(seq 1 12); do
    echo -n "."
    sleep 5
done
echo ""

# Check backend health
echo -e "${YELLOW}Checking backend health...${NC}"
HEALTHY=0
FAILED_PORTS=""
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BACKEND_BASE_PORT + i))
    if curl -s --connect-timeout 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
        ((HEALTHY++))
        echo -e "  Port $PORT: ${GREEN}OK${NC}"
    else
        echo -e "  Port $PORT: ${RED}FAILED${NC}"
        FAILED_PORTS="$FAILED_PORTS $PORT"
    fi
done

echo ""
echo -e "${GREEN}$HEALTHY/$NUM_INSTANCES backends healthy${NC}"

if [ $HEALTHY -eq 0 ]; then
    echo -e "${RED}No backends started successfully!${NC}"
    echo -e "${RED}Check logs: cat /tmp/sam_8010.log${NC}"
    echo ""
    echo "Last 50 lines from first backend:"
    tail -50 /tmp/sam_8010.log 2>/dev/null || echo "No log file found"
    exit 1
fi

# Update BACKEND_URLS to only include healthy backends
if [ $HEALTHY -lt $NUM_INSTANCES ]; then
    echo -e "${YELLOW}Rebuilding backend list with healthy instances only...${NC}"
    BACKEND_URLS=""
    for i in $(seq 0 $((NUM_INSTANCES - 1))); do
        PORT=$((BACKEND_BASE_PORT + i))
        if curl -s --connect-timeout 2 "http://localhost:$PORT/health" > /dev/null 2>&1; then
            if [ -z "$BACKEND_URLS" ]; then
                BACKEND_URLS="http://localhost:$PORT"
            else
                BACKEND_URLS="$BACKEND_URLS,http://localhost:$PORT"
            fi
        fi
    done
fi

# Start load balancer
echo ""
echo -e "${YELLOW}Starting load balancer on port $LB_PORT...${NC}"
tmux new-window -t sam_cluster -n "loadbalancer" \
    "python src/sam_load_balancer.py --port $LB_PORT --host $HOST --backends '$BACKEND_URLS' 2>&1 | tee /tmp/sam_lb.log; echo 'LB exited, press enter'; read"

sleep 5

# Verify load balancer
if curl -s --connect-timeout 5 "http://localhost:$LB_PORT/health" > /dev/null 2>&1; then
    echo -e "${GREEN}Load balancer started successfully!${NC}"
else
    echo -e "${RED}Load balancer failed to start!${NC}"
    echo "Last 30 lines from load balancer log:"
    tail -30 /tmp/sam_lb.log 2>/dev/null || echo "No log file"
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
echo -e "${GREEN}Tmux session: sam_cluster${NC}"
echo -e "${YELLOW}  Attach: tmux attach -t sam_cluster${NC}"
echo -e "${YELLOW}  List windows: tmux list-windows -t sam_cluster${NC}"
echo ""
echo -e "${GREEN}To stop: tmux kill-session -t sam_cluster${NC}"
echo ""

# Show health
curl -s "http://localhost:$LB_PORT/health" | python -m json.tool 2>/dev/null || true
