#!/bin/bash
# Deploy SAM3 Inference Server
# Based on rpol-recart/sam3_inference architecture

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
PORT=${SAM_PORT:-8001}
HOST=${SAM_HOST:-0.0.0.0}

echo -e "${GREEN}Starting SAM3 Inference Server...${NC}"

# Check CUDA availability
echo -e "${YELLOW}Checking CUDA availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'No CUDA')"

# Check if SAM3 is installed
echo -e "${YELLOW}Checking SAM3 installation...${NC}"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 library: OK')" 2>/dev/null || {
    echo -e "${RED}SAM3 not installed. Please install sam3 package.${NC}"
    exit 1
}

# Start server
echo -e "${GREEN}Launching SAM3 server on ${HOST}:${PORT}...${NC}"
echo -e "${YELLOW}API docs: http://${HOST}:${PORT}/docs${NC}"
echo -e "${YELLOW}Health check: http://${HOST}:${PORT}/health${NC}"

python src/sam_server.py --host ${HOST} --port ${PORT} "$@"
