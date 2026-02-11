# RESCue: Reasoning Segmentation with Cut-the-chase usage

RESCue is a high-performance **referring image segmentation system** that leverages a **Planner-Verifier-Executor** architecture to achieve state-of-the-art accuracy with optimized efficiency. It combines the reasoning capabilities of Large Language Models (LLMs) with the precision of the Segment Anything Model (SAM) and the semantic understanding of CLIP.

## 🚀 Key Features

*   **Planner-Verifier-Executor Architecture**:
    *   **Planner**: Generates diverse bounding box hypotheses based on the referring expression using a VLM.
    *   **Executor**: Segments the image using SAM3 based on the generated boxes.
    *   **Verifier**: ranks the resulting masks using a hybrid scoring system (VLM reasoning + CLIP score + Consistency).
*   **Pipeline Parallelism**: Concurrent processing of planning, execution, and verification stages to maximize GPU utilization.
*   **Tournament Refinement**: Efficient pyramid tournament strategy to rank masks with minimal VLM calls.
*   **Hardware Optimized**:
    *   **ROCm Support**: Includes automatic patches for `roi_align` and `nms` on AMD GPUs to prevent crashes.
    *   **CUDA Support**: Compatible with NVIDIA GPUs via standard PyTorch.

---

## 🛠️ System Requirements

### Hardware
*   **Primary Tested Setup**: 8x AMD MI325X GPUs (256GB VRAM each).
*   **Alternative Setup**: NVIDIA GPUs (A100/H100 recommended) with at least 80GB VRAM per node for the larger models (30B+).
*   **Memory**: ~160GB+ System RAM recommended for loading large LLMs.

### Software
*   **OS**: Linux (Ubuntu 20.04+ recommended)
*   **Python**: 3.10+
*   **Drivers**: ROCm 6.0+ (for AMD) or CUDA 12.0+ (for NVIDIA)

---

## 📦 Installation

### 1. Client & Environment Setup
Install the core dependencies for the logic and orchestration code.

```bash
# Clone the repository
git clone https://github.com/your-repo/RESCue.git
cd RESCue

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Model Server Dependencies
The system relies on several model servers. You may need to install specific packages on the nodes running these servers.

*   **VLM Nodes (Planner & Verifier)**:
    ```bash
    pip install vllm==0.4.0  # Or latest compatible version
    ```

*   **SAM & CLIP Nodes**:
    ```bash
    pip install fastapi uvicorn pillow numpy requests aiohttp transformers
    # Install SAM3 from source
    pip install "git+https://github.com/facebookresearch/sam3.git"
    ```

### 3. Download Models
Pre-download the necessary model weights to avoid timeouts during runtime.

```bash
python scripts/download_models.py
```

---

## 🏗️ Architecture & Deployment

RESCue uses a **distributed service architecture**. For the full performance setup (8 GPUs), deploy the services as follows:

| Service | Model | Default GPU(s) | Port | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Verifier** | `Qwen/Qwen3-VL-30B-A3B-Thinking` | 0, 1, 2, 3 | **8000** | Performs detailed chain-of-thought verification of segments. |
| **Planner** | `Qwen/Qwen3-VL-8B-Instruct` | 4, 5 | **8002** | Rapidly generates initial bounding box hypotheses. |
| **SAM Server** | `facebook/sam3` | 6 | **8001** | High-fidelity image segmentation server. |
| **CLIP Server** | `google/siglip2-giant` | 7 | **8003** | Visual-semantic scoring for valid candidates. |

### deploying Services

Run each command in a **separate terminal** (or use `tmux`/`screen`).

#### 1. Verifier Service (Port 8000)
```bash
# Deploys the 30B reasoning model (Tensor Parallel = 4)
bash scripts/deploy_llm.sh
```

#### 2. Planner Service (Port 8002)
```bash
# Deploys the 8B instruction model
bash scripts/deploy_planner_llm.sh
```

#### 3. SAM Server (Port 8001)
```bash
# Deploys the Segment Anything Model
bash scripts/deploy_sam.sh
```

#### 4. CLIP Server (Port 8003)
```bash
# Deploys the SigLIP verification server
bash scripts/deploy_clip.sh
```

---

## ⚙️ Configuration

You can override default behaviors using environment variables.

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `VERIFIER_API_BASE` | `http://localhost:8000/v1` | URL for the Verifier VLM service. |
| `PLANNER_API_BASE` | `http://localhost:8002/v1` | URL for the Planner VLM service. |
| `EXECUTOR_URL` | `http://localhost:8001` | URL for the SAM3 server. |
| `CLIP_SERVER_URL` | `http://localhost:8003/verify` | URL for the CLIP scoring endpoint. |
| `VERIFIER_MODEL` | `Qwen/Qwen3-VL-30B-A3B-Thinking` | HuggingFace model ID for the verifier. |
| `PLANNER_MODEL` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model ID for the planner. |

---

## 📊 Evaluation

### Running the Pipeline
To evaluate the system on your dataset (or a fraction of it), use the `evaluate_pipeline.py` script. This script orchestrates the parallel execution of all components.

```bash
python scripts/evaluate_pipeline.py \
  --fraction 0.1 \
  --parallel_requests 4 \
  --mode comparative
```

**Arguments:**
*   `--fraction`: Portion of the dataset to evaluate (e.g., `0.1` for 10%).
*   `--parallel_requests`: Number of concurrent requests to the SAM server.
*   `--mode`:
    *   `comparative`: Detailed tournament-based ranking (Slower, Higher Accuracy).
    *   `heuristic`: Fast scoring based on geometric properties and simple CLIP scores (Faster).

### Results
Results are saved to `results_pipeline_comparative_{fraction}.json`, containing:
*   Mean IoU (Intersection over Union)
*   Oracle Performance (Best possible IoU from hypotheses)
*   Latency/Timing metrics

---

## ❓ Troubleshooting

### ROCm / AMD GPU Issues
If you encounter errors related to `roi_align` or `nms` on AMD GPUs:
*   **Solution**: The `sam_server.py` includes a specialized patch (`patch_torchvision_for_rocm`) that automatically reroutes these specific operations to the CPU to avoid ROCm kernel crashes. **Do not disable this patch** unless you are sure your ROCm version (6.1+) supports these operations natively without fault.

### Connection Refused (Timeouts)
*   Ensure all 4 services are running and listening on their respective ports (`8000`, `8001`, `8002`, `8003`).
*   Check that you have pre-downloaded models using `scripts/download_models.py` to prevent startup timeouts.
*   Verify firewall settings if running on a remote cluster.
