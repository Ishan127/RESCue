#!/bin/bash
# Deploy SAM3 (EXECUTOR) on GPU 3
# Segmentation model for mask generation
# Port: 8001

set -e

# GPU Assignment
export CUDA_VISIBLE_DEVICES=3

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PORT=${SAM_PORT:-8001}
HOST=${SAM_HOST:-0.0.0.0}
FORCE_CPU=${SAM_FORCE_CPU:-false}

echo -e "${GREEN}Starting SAM3 Inference Server...${NC}"

# Check CUDA availability
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"

# Check for ROCm/AMD GPU
echo -e "${YELLOW}Checking for ROCm/AMD GPU...${NC}"
IS_ROCM=$(python -c "import torch; print('yes' if (hasattr(torch.version, 'hip') and torch.version.hip) or (torch.cuda.is_available() and 'amd' in torch.cuda.get_device_name(0).lower()) else 'no')" 2>/dev/null || echo "no")
if [ "$IS_ROCM" = "yes" ]; then
    echo -e "${YELLOW}ROCm/AMD GPU detected! Some torchvision ops may not work.${NC}"
    echo -e "${YELLOW}Use --force-cpu flag or set SAM_FORCE_CPU=true if you see errors.${NC}"
fi

# Check if SAM3 is installed
echo -e "${YELLOW}Checking SAM3 installation...${NC}"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 library: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed. Please install sam3 package.${NC}"
    exit 1
}

# Build command args
EXTRA_ARGS=""
if [ "$FORCE_CPU" = "true" ]; then
    EXTRA_ARGS="--force-cpu"
    echo -e "${YELLOW}Forcing CPU inference mode${NC}"
fi

# Start server
echo -e "${GREEN}Launching SAM3 server on ${HOST}:${PORT}...${NC}"
echo -e "${YELLOW}API docs: http://${HOST}:${PORT}/docs${NC}"
echo -e "${YELLOW}Health check: http://${HOST}:${PORT}/health${NC}"

python src/sam_server.py --host ${HOST} --port ${PORT} ${EXTRA_ARGS} "$@"
