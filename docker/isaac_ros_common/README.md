# Isaac ROS Common — Setup & Deployment Guide

> Complete guide for setting up and deploying the [javilinos/isaac_ros_common](https://github.com/javilinos/isaac_ros_common) development environment, from host prerequisites through container launch.

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Step 1 — Install NVIDIA GPU Driver](#step-1--install-nvidia-gpu-driver)
4. [Step 2 — Install Docker Engine](#step-2--install-docker-engine)
5. [Step 3 — Install NVIDIA Container Toolkit](#step-3--install-nvidia-container-toolkit)
6. [Step 4 — Configure Docker for NVIDIA Runtime](#step-4--configure-docker-for-nvidia-runtime)
7. [Step 5 — Clone this Repository](#step-6--clone-this-repository)
8. [Step 6 — Launch the Dev Container](#step-7--launch-the-dev-container)
9. [Step 7 — Launch via Docker Compose](#step-8--launch-via-docker-compose)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Overview

This repository is a fork of [NVIDIA-ISAAC-ROS/isaac_ros_common](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common) and provides Dockerfiles, scripts, and utilities for developing with the Isaac ROS suite. The Docker images ship pre-compiled ROS 2 Humble binaries and all the NVIDIA frameworks needed to work with Isaac ROS packages.

There are **two ways** to launch the development container:

| Method | Path | Description |
|--------|------|-------------|
| `run_dev.sh` | `scripts/run_dev.sh` | Interactive single-container dev environment |
| `compose.sh` | `compose/compose.sh` | Docker Compose-based deployment |

Both methods require the same host-level prerequisites described below.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Ubuntu 22.04 (Jammy) or Ubuntu 24.04 (Noble) |
| **GPU** | NVIDIA GPU — Ampere architecture or newer, 8 GB+ VRAM |
| **Driver** | NVIDIA Driver 535+ (check with `nvidia-smi`) |
| **CUDA** | CUDA 12.0+ (bundled with the driver) |
| **Disk** | 32 GB+ free (Docker images are large) |
| **Platforms** | x86_64 with discrete NVIDIA GPU **or** Jetson (JetPack 5.1.2+) |

---

## Step 1 — Install NVIDIA GPU Driver

Install the latest NVIDIA GPU driver from Ubuntu's official repositories:

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
```

Reboot and verify:

```bash
sudo reboot
# After reboot:
nvidia-smi
```

Confirm that `nvidia-smi` displays your GPU, driver version, and CUDA version.

---

## Step 2 — Install Docker Engine

Install Docker CE from Docker's official APT repository.

### 2.1 — Add Docker's GPG key and repository

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# Add Docker's official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 2.2 — Install Docker packages

```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 2.3 — Enable Docker for your user (no sudo)

```bash
sudo usermod -aG docker $USER
newgrp docker
```

> **Note:** You may need to log out and back in for group changes to take effect.

### 2.4 — Verify Docker

```bash
docker run --rm hello-world
```

---

## Step 3 — Install NVIDIA Container Toolkit

The NVIDIA Container Toolkit enables Docker to access the host GPU from inside containers.

### 3.1 — Add the NVIDIA Container Toolkit repository

```bash
# Install prerequisites
sudo apt-get update && sudo apt-get install -y ca-certificates curl gnupg2

# Add GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 3.2 — Install the toolkit

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

---

## Step 4 — Configure Docker for NVIDIA Runtime

Tell Docker to use the NVIDIA runtime, then restart the daemon:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl daemon-reload && sudo systemctl restart docker
```

### Verify GPU access from Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

You should see the same `nvidia-smi` output as on the host. If this fails, revisit Steps 1–3.


## Step 5 — Clone this Repository

```bash
cd ~
git clone https://github.com/javilinos/isaac_ros_common.git
```

---

## Step 6 — Launch the Dev Container (`run_dev.sh`)

The `run_dev.sh` script builds (or pulls) the Isaac ROS Docker image and drops you into an interactive shell inside the container.

```bash
cd ~/isaac_ros_common/scripts
./run_dev.sh
```

---

## Step 7 — Launch via Docker Compose (`compose.sh`)

For a Docker Compose-based workflow, use the `compose/` directory. This method will deploy aerostack and the aerogenia project.

```bash
cd ~/isaac_ros_common/compose
./compose.sh
```

To stop and remove the containers:

```bash
docker compose down
```

---

## Troubleshooting

### Docker permission denied

If you see `permission denied` errors when running Docker commands:

```bash
sudo usermod -aG docker $USER
# Then log out and log back in, or run:
newgrp docker
```

### nvidia-smi not found inside the container

Make sure the NVIDIA Container Toolkit is properly installed and configured (Steps 3–4). Verify with:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### run_dev.sh does nothing or fails immediately

Ensure you are **not** running the script with `sudo`. The script checks that your user is in the `docker` group. If you recently added yourself, log out and back in.

### Image build fails with network errors

Large Docker images may time out. Retry the build or use the `-b` flag to skip building if a pre-built image is available.

### Container cannot access devices (cameras, sensors)

The container runs with `--privileged` and `--network host` by default. If you still have issues, check that your device is visible on the host (`ls /dev/video*`) and that udev rules are configured.

---

## References

- [NVIDIA Isaac ROS Getting Started](https://nvidia-isaac-ros.github.io/getting_started/index.html) — Official setup documentation
- [NVIDIA Isaac ROS Common Docs](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html) — Repository-specific documentation
- [Docker Engine Install (Ubuntu)](https://docs.docker.com/engine/install/ubuntu/) — Official Docker installation guide
- [NVIDIA Container Toolkit Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — Official NVIDIA Container Toolkit docs
- [Isaac ROS Troubleshooting](https://nvidia-isaac-ros.github.io/troubleshooting/index.html) — Known issues and fixes
