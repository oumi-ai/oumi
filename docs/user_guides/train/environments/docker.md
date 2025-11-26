# Using Docker

Oumi provides pre-built Docker images that include all necessary dependencies for training, evaluation, and inference. Using Docker eliminates installation complexity and ensures consistent environments across different systems.

## Available Images

Oumi Docker images are published to the GitHub Container Registry at: [ghcr.io/oumi-ai/oumi](https://ghcr.io/oumi-ai/oumi)

### Image Tags

| Tag | Description | Use Case |
|-----|-------------|----------|
| `latest` | Latest stable release | Production use |
| `0.x.x`, `0.x`, `x` | Specific version | Version pinning |
| `pr-XXXX` | Pull request builds | Testing specific changes |

### Platform Support

Oumi Docker images support multiple architectures:

| Platform | Architecture | GPU Support | PyTorch Version | Image Size |
|----------|--------------|-------------|-----------------|------------|
| **AMD64** | x86_64/Intel/AMD | CUDA 12.8 | PyTorch 2.8.0 with CUDA | ~14GB |
<!-- | **ARM64** | Apple Silicon, ARM servers | CPU only | PyTorch 2.8.0 CPU | ~3GB | -->

**Supported Operating Systems**:

- ✅ **Linux** (AMD64/ARM64) - Native containers
- ✅ **macOS** (Intel/Apple Silicon) - via Docker Desktop
- ✅ **Windows** (Intel/AMD) - via Docker Desktop + WSL2

````{note}
**Mac Silicon Users**: If you need GPU-compatible images for development or testing, use the `--platform linux/amd64` flag when pulling or running images:

```bash
docker pull --platform linux/amd64 ghcr.io/oumi-ai/oumi:latest
docker run --platform linux/amd64 -it --rm ghcr.io/oumi-ai/oumi:latest bash
```
````

## Quick Start

### Pull the Image

```bash
docker pull ghcr.io/oumi-ai/oumi:latest
```

### Verify Installation

```bash
docker run --rm ghcr.io/oumi-ai/oumi:latest oumi --help
```

### Interactive Shell

Launch an interactive container to explore Oumi:

```bash
docker run -it --rm ghcr.io/oumi-ai/oumi:latest bash
```

Once inside, you can run any Oumi command:

```bash
oumi env  # Check environment info
oumi --help  # View available commands
```

### Using NVIDIA GPUs

```bash
docker run --gpus all -it --rm ghcr.io/oumi-ai/oumi:latest bash
```

The `--gpus all` flag makes all available GPUs accessible to the container.

### Verify GPU Access

Inside the container, verify GPU access:

```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Working with Data and Models

To persist data, models, and outputs, mount volumes from your host machine into the container.

```bash
docker run -it --rm \
  -v $(pwd)/data:/oumi_workdir/data \
  -v $(pwd)/outputs:/oumi_workdir/outputs \
  -v ~/.cache/huggingface:/home/oumi/.cache/huggingface \
  ghcr.io/oumi-ai/oumi:latest bash
```

This command:

- Mounts `./data` from your host to `/oumi_workdir/data` in the container
- Mounts `./outputs` from your host to `/oumi_workdir/outputs` in the container
- The HuggingFace cache mount (`~/.cache/huggingface`) prevents re-downloading models each time you run a container.
- Any changes to these directories persist after the container exits

### Building Custom Images

If you need custom dependencies or configurations, you can build your own image:

1. Clone the Oumi repository:

   ```bash
   git clone https://github.com/oumi-ai/oumi.git
   cd oumi
   ```

2. Modify the `Dockerfile` as needed

3. Build the image:

   ```bash
   docker build -t my-oumi:latest .
   ```

4. Use your custom image:

   ```bash
   docker run -it --rm --gpus all my-oumi:latest bash
   ```

## Environment Variables

You can pass environment variables to configure Oumi behavior:

```bash
docker run -it --rm \
  --gpus all \
  -e WANDB_API_KEY=your_wandb_key \
  -e HF_TOKEN=your_hf_token \
  -e OUMI_LOG_LEVEL=DEBUG \
  ghcr.io/oumi-ai/oumi:latest bash
```

## Next Steps

- See the {doc}`quickstart guide </get_started/quickstart>` for training examples
- Learn about {doc}`training configuration </user_guides/train/configuration>`
- Explore {doc}`evaluation </user_guides/evaluate/evaluate>` and {doc}`inference </user_guides/infer/infer>` guides
- Check out {doc}`remote training </user_guides/launch/launch>` for cloud deployments

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Oumi GitHub Container Registry](https://github.com/oumi-ai/oumi/pkgs/container/oumi)
