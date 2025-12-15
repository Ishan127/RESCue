---
description: Setup Docker on Windows with WSL2 for ROCm usage
---

# Docker Setup Guide (Windows + WSL2)

This workflow guides you through setting up Docker Desktop on Windows with WSL2 support, which is required to run the Linux-based RESCue container.

> [!WARNING]
> **MI325X Compatibility**: Running AMD Instinct MI300/MI325X GPUs inside standard Docker containers on Windows via WSL2 is **experimental** and inherently complex. The preferred deployment for this hardware is **Native Linux**. Proceed with the understanding that GPU passthrough (binding `/dev/kfd`) may not function correctly in WSL2 depending on your specific driver version and kernel.

## 1. Enable WSL2

Open PowerShell as **Administrator** and run:

```powershell
wsl --install
```

*Restart your computer if prompted.*

After restart, ensure you have a default Linux distribution (usually Ubuntu) installed.
Verify WSL version:
```powershell
wsl --status
```
Ensure "Default Version" is 2.

## 2. Install Docker Desktop

1.  Download **Docker Desktop for Windows** from [docker.com](https://www.docker.com/products/docker-desktop/).
2.  Run the installer.
3.  Ensure **"Use WSL 2 instead of Hyper-V"** is checked during installation.
4.  Restart Windows.

## 3. Configure Docker Desktop

1.  Open Docker Desktop.
2.  Go to **Settings (Gear Icon)** -> **Resources** -> **WSL Integration**.
3.  Ensure "Enable integration with my default WSL distro" is **checked**.
4.  (Optional) If you installed a specific Ubuntu distro, toggle the switch for it.
5.  Click **Apply & Restart**.

## 4. Verify GPU Access (The Critical Step)

1.  Open your Ubuntu terminal (WSL).
2.  Check for AMD kernel devices:
    ```bash
    ls -l /dev/kfd /dev/dri
    ```
    *   **Success**: You see `/dev/kfd` and `/dev/dri/renderD128`. This means the kernel sees the GPU.
    *   **Failure**: If these are missing, Docker will NOT be able to use the GPU. You likely need to install the **AMD Software: Adrenalin Edition** or **PRO Edition** drivers on Windows side, which map the GPU to proper WSL components.

## 5. Run the Container

Use a localized version of the run command (in PowerShell or WSL):

```bash
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HF_TOKEN=$HF_TOKEN \
    rescue-app
```

> [!TIP]
> If direct Docker execution fails, consider running the Python scripts **directly on your host Windows** machine if you have the AMD HIP SDK installed. The provided Python scripts (`src/*.py`) use generic `torch` calls that should work on Windows native if you have the correct `torch-directml` or ROCm-Windows build.
