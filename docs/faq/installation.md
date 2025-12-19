# Installation FAQ

Common questions and solutions for installing Oumi.

## Python Version Issues

### Which Python versions are supported?

Oumi requires Python 3.9 or later, but **not** Python 3.13+. We recommend Python 3.10, 3.11, or 3.12 for best compatibility.

```bash
# Check your Python version
python --version
```

### I'm getting "No module named 'resource'" on Windows

This error occurs when trying to run Oumi natively on Windows:

```text
ModuleNotFoundError: No module named 'resource'
```

**Solution**: Use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) instead of running natively on Windows.

```bash
# Install WSL
wsl --install

# Then install Oumi inside WSL
pip install oumi
```

### I'm getting version compatibility errors on Intel Mac

Oumi only supports Apple Silicon Macs (M1/M2/M3), not Intel Macs. This is because PyTorch dropped support for Intel Macs.

```text
× No solution found when resolving dependencies:
  ... no wheels with a matching platform tag (e.g., `macosx_10_16_x86_64`)
```

**Solution**: Use [Google Colab](https://colab.research.google.com/) or a cloud-based environment instead.

## Dependency Issues

### How do I install GPU support?

For NVIDIA GPUs, install with the GPU extras:

```bash
pip install "oumi[gpu]"
```

This will install vLLM and other GPU-accelerated dependencies.

### How do I install for a specific cloud provider?

Install the appropriate cloud extras:

```bash
pip install "oumi[aws]"     # For Amazon Web Services
pip install "oumi[azure]"   # For Microsoft Azure
pip install "oumi[gcp]"     # For Google Cloud Platform
pip install "oumi[lambda]"  # For Lambda Cloud
pip install "oumi[runpod]"  # For RunPod

# Multiple providers at once
pip install "oumi[aws,gcp,azure]"
```

### pip install fails with dependency conflicts

Try installing in a fresh virtual environment:

```bash
# Create a new virtual environment
python -m venv oumi-env

# Activate it
source oumi-env/bin/activate  # Linux/Mac
# or
oumi-env\Scripts\activate     # Windows

# Install Oumi
pip install oumi
```

### I'm getting "Could not find a version that satisfies the requirement"

This usually means you're on an unsupported platform or Python version.

**Check**:

1. Python version is 3.9-3.12
2. You're using a supported OS (Linux, macOS with Apple Silicon, or Windows with WSL)
3. You have a fresh virtual environment

## Virtual Environment Issues

### Should I use venv, conda, or uv?

All three work well with Oumi:

::::{tab-set}
:::{tab-item} venv

```bash
python -m venv .env
source .env/bin/activate
pip install oumi
```

:::

:::{tab-item} conda

```bash
conda create -n oumi python=3.11
conda activate oumi
pip install oumi
```

:::

:::{tab-item} uv

```bash
uv venv
source .venv/bin/activate
uv pip install oumi
```

:::
::::

### Conda environment activation doesn't persist

If your conda environment doesn't persist between terminal sessions, add this to your shell profile:

```bash
# Add to ~/.bashrc or ~/.zshrc
conda activate oumi
```

## Docker Issues

### How do I run Oumi in Docker?

Pull and run the official Docker image:

```bash
# Pull the latest image
docker pull ghcr.io/oumi-ai/oumi:latest

# Run with GPU support
docker run --gpus all -it --rm ghcr.io/oumi-ai/oumi:latest oumi --help
```

For more details, see the {doc}`Docker guide </user_guides/train/environments/docker>`.

### Docker GPU support not working

Ensure you have the NVIDIA Container Toolkit installed:

```bash
# Install NVIDIA Container Toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Verification

### How do I verify Oumi is installed correctly?

Run the help command:

```bash
oumi --help
```

You should see the available commands listed.

### How do I check which optional dependencies are installed?

Check your installed packages:

```bash
pip list | grep -E "oumi|vllm|torch|transformers"
```

## See Also

- {doc}`/get_started/installation` - Full installation guide
- {doc}`troubleshooting` - General troubleshooting
- {doc}`/development/dev_setup` - Developer environment setup
