---
description: Setup Docker on Ubuntu for AMD ROCm usage
---

# Docker Setup Guide (Ubuntu Linux)

This workflow guides you through setting up Docker on an Ubuntu machine with AMD ROCm support. This is the **native and recommended** way to run RESCue.

## 1. Install Docker Engine

Remove old versions (if any):
```bash
sudo apt-get remove docker docker-engine docker.io containerd runc
```

Set up the repository:
```bash
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=\"$(dpkg --print-architecture)\" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Install Docker:
```bash
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 2. Post-installation Steps (Manage Docker as Non-Root)

To run docker without `sudo`:

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

## 3. Verify ROCm GPU Access

Ensure your Ubuntu host has the kernel drivers installed. You should see the KFD device:

```bash
ls -l /dev/kfd
```

## 4. Run the Container

Build and run the RESCue container. We pass `/dev/kfd` and `/dev/dri` to allow the container to access the MI325X GPU.

```bash
# Build
docker build -t rescue-app .

# Run
docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add video \
    -e HF_TOKEN=$HF_TOKEN \
    rescue-app
```
