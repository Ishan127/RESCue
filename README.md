# RESCue: Reasoning Segmentation with Cut-the-chase usage

RESCue is a high-performance referring image segmentation system that uses a planner-verifier-executor architecture to achieve state-of-the-art accuracy while maintaining efficiency.

## Prerequisites
- **Hardware**: 4x MI325X GPUs (256GB each) or equivalent high-VRAM setup
- **Software**: ROCm 7.1.1, vLLM 0.14.0, Python 3.10+
- **Memory**: ~160GB for LLM models

## Architecture

RESCue uses a **dual-model + single SAM** setup for optimal speed vs quality:

| Service | Model | GPU(s) | Port | Purpose |
|---------|-------|--------|------|---------|
| **Verifier** | Qwen3-VL-32B-Thinking | 0, 1 | 8000 | Tournament-based mask ranking |
| **Planner** | Qwen3-VL-8B-Instruct | 2 | 8002 | Fast hypothesis generation |
| **SAM Server** | SAM3 | 3 | 8001 | High-quality segmentation |

### Key Features
- **Pipeline Parallelism**: Process multiple samples concurrently across all GPUs
- **Single SAM Node**: Efficient single-node execution with internal batching
- **ROCm Compatibility**: Aggressive patches for AMD GPU torchvision operations
- **Tournament Verification**: Efficient mask ranking without pairwise comparisons

## 1. Environment Setup

### Client Setup (Local Machine / Front-End)
Runs the logic, interacts with endpoints. **No Torch/vLLM required.**
```bash
pip install -r requirements.txt
```

### Server Setup (GPU Nodes)

**VLM Nodes (LLM)**:
```bash
pip install vllm==0.14.0
```

**SAM Node (GPU 3)**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install "git+https://github.com/facebookresearch/sam3.git"
pip install fastapi uvicorn pillow numpy requests aiohttp
```

## 2. Download Models

Pre-download the model weights to avoid timeouts during deployment:

```bash
python scripts/download_models.py
```

## 3. Deploy Model Endpoints

Run the following services in **separate terminals** (e.g., using `tmux`).

### Terminal 1: Verifier (32B-Thinking) Service

Deploys Qwen3-VL-32B-Thinking on vLLM (tensor-parallel=2 for chain-of-thought verification).

```bash
bash scripts/deploy_llm.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8000`*

### Terminal 2: Planner (8B) Service

Deploys Qwen3-VL-8B-Instruct for fast hypothesis generation.

```bash
bash scripts/deploy_planner_llm.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8002`*

### Terminal 3: SAM Server

Deploys a single SAM3 instance (supports internal concurrency).

```bash
bash scripts/deploy_sam.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8001`*

## 4. Run Evaluation

### Pipeline-Parallel Evaluation (Recommended)

Processes multiple samples concurrently using all 4 GPUs:

```bash
python scripts/evaluate_pipeline.py \
  --fraction 0.1 \
  --parallel_requests 4 \
  --mode comparative
```

**Parameters:**
- `--fraction`: Fraction of dataset to evaluate (0.1 = 10%)
- `--parallel_requests`: Concurrent SAM requests (4 recommended for single node)
- `--mode`: `comparative` (tournament) or `heuristic` (fast scoring)
- `--max_n`: Maximum hypotheses to generate per sample (optimized for 64-128, default: 64)

## 5. Results

Results are saved as JSON files with IoU scores for different N values:

```
results_pipeline_comparative_10pct.json
```

Contains mean IoU, oracle performance, and timing data.

## Environment Variables

Override endpoints if needed:
```bash
export PLANNER_API_BASE=http://localhost:8002/v1
export VERIFIER_API_BASE=http://localhost:8000/v1
export EXECUTOR_URL=http://localhost:8001
```

## Troubleshooting

### Common Issues

**ROCm Errors**: The SAM server includes automatic patches for `roi_align` and `nms` operations.

**Timeout Errors**: Ensure all services are running and accessible:
```bash
curl http://localhost:8000/health  # Verifier
curl http://localhost:8002/health  # Planner
curl http://localhost:8001/health  # SAM Server
```

## Architecture Details

- **Planner**: Generates diverse bounding box hypotheses for referring expressions
- **Executor**: Segments images using SAM3
- **Verifier**: Ranks masks using tournament elimination (O(N) instead of O(N²))
- **Pipeline**: Concurrent processing of planning, execution, and verification stages
