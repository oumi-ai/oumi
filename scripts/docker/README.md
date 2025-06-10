# Oumi Docker

This directory contains Docker-related scripts and documentation for building and running Oumi in containers.

## Quick Start

### Using Pre-built Images

Pre-built Docker images are available from GitHub Container Registry:

```bash
# Pull the latest image
docker pull ghcr.io/oumi-ai/oumi:latest

# Pull a specific version
docker pull ghcr.io/oumi-ai/oumi:0.1.0

# Run the container with GPU support
docker run -it --gpus all ghcr.io/oumi-ai/oumi:latest
```

### Building Locally

Use the `build_docker.sh` script to build Docker images locally:

```bash
# Build with latest oumi version
./scripts/docker/build_docker.sh

# Build with custom tag
./scripts/docker/build_docker.sh my-tag

# Build with custom tag and specific oumi version
./scripts/docker/build_docker.sh my-tag 0.1.0

# Show help
./scripts/docker/build_docker.sh --help
```

The script accepts two optional positional arguments:
1. `TAG`: Docker image tag (default: `latest`)
2. `OUMI_VERSION`: Specific oumi version to install (default: latest)

## Docker Image Details

### Base Image
- PyTorch 2.5.1 with CUDA 12.4 and cuDNN 9 runtime
- Linux x86_64 (amd64) architecture
- GPU-enabled with NVIDIA driver support

### Installed Components
- Oumi package with GPU extras (`oumi[gpu]`)
- System utilities: git, vim, htop, tree, screen, curl
- Python package manager: uv

### User Configuration
- Non-root user: `oumi`
- Working directory: `/oumi_workdir`

## Running Oumi in Docker

### Basic Usage

```bash
# Interactive shell
docker run -it --gpus all ghcr.io/oumi-ai/oumi:latest bash

# Run oumi command directly
docker run --rm --gpus all ghcr.io/oumi-ai/oumi:latest oumi --version

# Mount local directory
docker run -it --gpus all -v $(pwd):/oumi_workdir ghcr.io/oumi-ai/oumi:latest
```

### Training Example

```bash
# Mount configs and run training
docker run --rm --gpus all \
  -v $(pwd)/configs:/oumi_workdir/configs \
  -v $(pwd)/output:/oumi_workdir/output \
  ghcr.io/oumi-ai/oumi:latest \
  oumi train -c configs/recipes/gpt2/pretraining/train.yaml
```

### Inference Example

```bash
# Run inference with mounted model
docker run -it --gpus all \
  -v $(pwd)/models:/oumi_workdir/models \
  ghcr.io/oumi-ai/oumi:latest \
  oumi infer -c configs/recipes/llama3_1/inference/8b_infer.yaml --interactive
```

## Authentication

To push images to GitHub Container Registry, you need to authenticate:

```bash
# Login with GitHub Personal Access Token (PAT)
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Or use GitHub CLI
gh auth token | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

Your PAT needs the `write:packages` scope to push container images.

## CI/CD Integration

The repository includes a GitHub Actions workflow (`.github/workflows/release_docker.yaml`) that automatically builds and publishes Docker images when:
- A new release is published (tagged with the release version)
- Manually triggered via workflow dispatch

## Troubleshooting

### GPU Not Available
Ensure you have:
- NVIDIA Docker runtime installed (`nvidia-docker2` package)
- NVIDIA drivers installed on the host
- Used the `--gpus all` flag when running the container

### Permission Denied
The container runs as the `oumi` user by default. If you need root access:
```bash
docker run -it --user root --gpus all ghcr.io/oumi-ai/oumi:latest bash
```

### Out of Memory
Adjust shared memory size for multi-GPU training:
```bash
docker run --rm --gpus all --shm-size=16gb ghcr.io/oumi-ai/oumi:latest
```
