# RESCue: Reasoning Segmentation with Cut-the-chase usage

## Prerequisites
- **Hardware**: 2x MI325X GPUs (or equivalent high-VRAM setup).
- **Software**: ROCm 7.1.1, vLLM 0.14.0, Python 3.10+.

## 1. Environment Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
pip install vllm==0.14.0
pip install "git+https://github.com/facebookresearch/segment-anything-2.git"
```

*(Note: Adjust pytorch/rocm versions as needed for your specific hardware).*

## 2. Download Models

Pre-download the model weights to avoid timeouts during deployment:

```bash
python scripts/download_models.py
```

## 3. Deploy Model Endpoints

Run the following services in **separate terminals** (e.g., using `tmux`).

### Terminal 1: Planner (LLM) Service

Deploys Qwen3-VL-30B-A3B-Instruct on vLLM.

```bash
./scripts/deploy_llm.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8000`*

### Terminal 2: Executor (SAM3) Service

Deploys the SAM3 Segmentation Server.

```bash
./scripts/deploy_sam.sh
```
*Wait for: `Uvicorn running on http://0.0.0.0:8001`*

## 4. Run Benchmark

Once both services are up, run the main orchestration script. This handles data downloading, health checks, and evaluation.

```bash
python main_benchmark.py
```

## Custom Usage

To run inference on a single image:

```bash
python scripts/run_inference.py \
  --image example.jpg \
  --query "the red car in the background" \
  --planner_url http://localhost:8000/v1 \
  --executor_url http://localhost:8001
```
