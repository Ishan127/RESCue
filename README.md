# RESCue: Search-Based Segmentation

This repository implements **Search-Based Segmentation (RESCue)**, a framework that establishes Inference-Time Scaling Laws for reasoning segmentation. It decouples the process into a **Planner** (Qwen2.5-VL), an **Executor** (SAM 3), and a **Verifier** (VLM-as-a-Judge).

## Abstract
The paradigm of Artificial Intelligence has recently bifurcated. While models continue to scale via massive pre-training (System 1), a new frontier of models has emerged, trading test-time compute for enhanced reasoning capabilities (System 2). RESCue applies this to segmentation by generating diverse visual hypotheses and verifying them to select the best mask.

## Environment Setup (AMD MI325X)

This codebase is optimized for **AMD MI325X** GPUs running **ROCm 6.3.0**.

### Prerequisites
- **OS**: Linux (tested on user's AMD environment) / Windows (for development)
- **GPU**: AMD MI325X (or compatible ROCm GPU)
- **ROCm**: 6.3.0
- **PyTorch**: 2.6.0 (ROCm compatible)
- **vLLM**: 0.6.4 (ROCm compatible)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd RESCue
    ```

2.  **Set up the Python Environment**:
    It is recommended to use the pre-configured environment `AMD AAC VLLM_0_6_4_Rocm_6_3_0_Pytorch_2_6_0` if available, or create a new one:

    ```bash
    # If starting from scratch (ensure python 3.10+ is used)
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    > [!IMPORTANT]
    > **Do NOT reinstall torch/vllm from PyPI.** The `requirements.txt` has `torch` and `vllm` commented out to preserve your pre-installed ROCm versions. If you need to install them manually, use the ROCm-specific index url (e.g., `--index-url https://download.pytorch.org/whl/rocm6.2`).
    
    > [!WARNING]
    > **If you encounter "Found no NVIDIA driver" errors:**
    > This means you have installed the wrong (Nvidia) PyTorch/vLLM. Run the fix script:
    > ```bash
    > bash scripts/fix_env.sh
    > ```

    *Note: `sam3` is installed directly from the official GitHub repository.*

## Usage

### 1. Download Models
Download the required model weights (Qwen2.5-VL and SAM 3 checkpoints):
```bash
python scripts/download_models.py
```

### 2. Download Data
Download the **ReasonSeg** validation dataset for evaluation:
```bash
python scripts/download_data.py
```

### 3. Run Inference (Single Image)
Run the RESCue pipeline on an image with a text query:
```bash
python scripts/run_inference.py --image "access/demo.jpg" --query "the object that would break if the boy jumps" --N 4
```
- `--N`: Number of reasoning paths to sample (Inference-time scaling parameter).

### 4. Run Evaluation (Dataset)
Evaluate the model on a fraction of the ReasonSeg dataset:
```bash
python scripts/evaluate.py --fraction 0.1 --N 4
```
- `--fraction`: Proportion of the dataset to evaluate (e.g., `0.1` for 10%).

## Docker Support

You can run the entire workflow in a Docker container, pre-configured with ROCm support.

### 1. Build the Image
```bash
docker build -t rescue-app .
```

### 2. Run with ROCm GPU Support
To ensure the container can access the AMD MI325X GPU, you must pass the DRI and KFD devices and the video group:

```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HF_TOKEN=$HF_TOKEN \
    rescue-app
```
*Note: Pass your Hugging Face token as an environment variable.*

## Architecture

- **Planner**: `Qwen2-VL-72B-Instruct` generates bounding box hypotheses via Visual Chain-of-Thought (vCoT).
- **Executor**: `SAM 3` grounds the boxes + text concepts into pixel masks.
- **Verifier**: `Qwen2-VL-72B-Instruct` (VLM-as-a-Judge) scores the masks based on visual alignment with the query.

## Directory Structure
```
RESCue/
├── src/
│   ├── rescue_pipeline.py  # Main System 2 Loop
│   ├── planner.py          # Qwen2.5-VL Interface
│   ├── executor.py         # SAM 3 Interface
│   ├── verifier.py         # VLM-as-a-Judge
│   └── utils.py            # Visualization & Helpers
├── scripts/
│   ├── setup_environment.sh
│   ├── download_models.py
│   ├── download_data.py
│   ├── evaluate.py
│   └── run_inference.py
├── tests/
└── requirements.txt
```
