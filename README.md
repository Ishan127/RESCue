# RESCue: Reasoning Segmentation with Cut-the-chase usage

## Prerequisites
- **Hardware**: 4x MI325X GPUs (or equivalent high-VRAM setup).
- **Software**: ROCm 7.1.1, vLLM 0.14.0, Python 3.10+.

## Architecture

RESCue uses a **dual-model** setup for optimal speed vs quality:

| Service | Model | GPU(s) | Port |
|---------|-------|--------|------|
| **Verifier** | Qwen3-VL-32B-Thinking | 0, 1 | 8000 |
| **Planner** | Qwen3-VL-8B-Instruct | 2 | 8002 |
| **Executor** | SAM3 | 3 | 8001 |

## 1. Environment Setup

The system is split into **Client** (Verification/Orchestration) and **Server** (Model Serving).

### Client Setup (Local Machine / Front-End)
Runs the logic, interacts with endpoints. **No Torch/vLLM required.**
```bash
pip install -r requirements.txt
# Ensure requirements.txt contains: numpy, pillow, opencv-python, requests, datasets, tqdm, fastapi, uvicorn
```

### Server Setup (GPU Nodes)

**VLM Nodes (LLM)**:
```bash
pip install vllm==0.14.0
```

**Executor Node (SAM)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install "git+https://github.com/facebookresearch/sam3.git"
pip install fastapi uvicorn pillow numpy requests
```

## 2. Download Models

Pre-download the model weights to avoid timeouts during deployment:

```bash
python scripts/download_models.py
```

## 3. Deploy Model Endpoints

Run the following services in **separate terminals** (e.g., using `tmux`).

### Terminal 1: Verifier (32B-Thinking) Service

Deploys Qwen3-VL-32B-Thinking on vLLM (chain-of-thought model for verification).

```bash
./scripts/deploy_llm.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8000`*

### Terminal 2: Planner (7B) Service  

Deploys Qwen3-VL-8B-Instruct for fast hypothesis generation.

```bash
./scripts/deploy_planner_llm.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8002`*

### Terminal 3: Executor (SAM3) Service

Deploys the SAM3 Segmentation Server.

```bash
./scripts/deploy_sam.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8001`*

## 4. Run Benchmark

Once all services are up, run the main orchestration script:

```bash
python main_benchmark.py
```

### Multi-N Evaluation (Recommended)

Generate once, evaluate for multiple N values:

```bash
python scripts/evaluate_multi_n.py --fraction 0.1 --max_n 64
```

## Environment Variables

Override model endpoints if needed:
```bash
export PLANNER_API_BASE=http://localhost:8002/v1
export VERIFIER_API_BASE=http://localhost:8000/v1
```

## Custom Usage

To run inference on a single image:

```bash
python scripts/run_inference.py \
  --image example.jpg \
  --query "the red car in the background" \
  --planner_url http://localhost:8002/v1 \
  --verifier_url http://localhost:8000/v1 \
  --executor_url http://localhost:8001
```
