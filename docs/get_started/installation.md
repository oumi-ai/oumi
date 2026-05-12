# Installation

This guide will help you install Oumi OSS and its dependencies.

## Requirements

❗NOTE: Since PyTorch dropped support for Intel Macs, you cannot install Oumi OSS on those machines. Consider running Oumi OSS on free Colab GPU instances, using our {doc}`notebook tutorials </get_started/tutorials>`!

Before installing Oumi, ensure you have the following:

- Python 3.9 or later
- pip (Python package installer)
- Git (if cloning the repository; required for steps 2 and 3)
- 10-20GB of disk space

We recommend using a virtual environment to install Oumi OSS. You can find instructions for setting up a Conda environment in the {doc}`/development/dev_setup` guide.

## Installation Methods

You can install Oumi OSS using one of the following methods:

### 1. Install from PyPI (Recommended)

To prevent dependency conflicts, let's start by creating a virtual environment. We'll use `venv` below, but you are welcome to use the environment manager of your choice ([conda](/development/dev_setup), [uvx](https://docs.astral.sh/uv/concepts/tools/), etc):

::::{tab-set}
:::{tab-item} Linux / MacOS

```{code-block} shell
python -m venv .env
source .env/bin/activate
```

:::

:::{tab-item} Windows

```{code-block} shell
python -m venv .env
.env/Scripts/activate
```

:::
::::

Once that's done, you're ready to install Oumi OSS!

To install the latest stable version of Oumi OSS, run:

```bash
uv pip install oumi
```

Don't have uv? [Install it](https://docs.astral.sh/uv/getting-started/installation/) or use `pip` instead of `uv pip` in the commands below.

### 2. Install from Source

For the latest development version, you can install Oumi OSS directly from the GitHub repository:

::::{tab-set}
:::{tab-item} SSH

```{code-block} shell
uv pip install git+ssh://git@github.com/oumi-ai/oumi.git
```

:::

:::{tab-item} HTTPS

```{code-block} shell
uv pip install git+https://github.com/oumi-ai/oumi.git
```

:::
::::

### 3. Clone and Install

If you want to contribute to Oumi OSS or need the full source code, you can clone the repository and install it:

::::{tab-set}
:::{tab-item} SSH

```{code-block} shell
git clone git@github.com:oumi-ai/oumi.git
cd oumi
uv pip install -e ".[dev]"
```

:::

:::{tab-item} HTTPS

```{code-block} shell
git clone https://github.com/oumi-ai/oumi.git
cd oumi
uv pip install -e ".[dev]"
```

:::
::::

For more information on setting up your dev environment for contributing to Oumi OSS , please
see our [dev setup guide](../development/dev_setup.md).

The `-e` flag installs the project in "editable" mode. This means that changes made to the source code will be immediately reflected in the installed package without needing to reinstall it. This is particularly helpful when you're actively developing features and want to test your changes quickly. It creates a link to the project's source code instead of copying the files, allowing you to modify the code and see the effects immediately in your Python environment.

### 4. Quick Install Script (Experimental)

Try Oumi OSS without setting up a Python environment. This installs Oumi OSS in an isolated environment:

```bash
curl -LsSf https://oumi.ai/install.sh | bash
```

## Recommended: pick a tested stack

Oumi's base install uses wide version bands so `pip install oumi` works in many environments, but the resolver can pick incompatible combinations from those bands silently. Compose a `cudaNNN` extra (which routes torch to the matching PyTorch wheel index) with `torch29,tf4` for the GPU stack, or use `tf5` alone for CPU/dev:

| Stack | Install command (uv) |
|-------|----------------------|
| GPU, CUDA 12.6 | `uv pip install "oumi[torch29,tf4,cuda126]"` |
| GPU, CUDA 12.8 (Docker default) | `uv pip install "oumi[torch29,tf4,cuda128]"` |
| GPU, CUDA 12.9 | `uv pip install "oumi[torch29,tf4,cuda129]"` |
| CPU / dev | `uv pip install "oumi[tf5]"` |

Compose with cloud and dev extras as usual: `uv pip install "oumi[torch29,tf4,cuda128,dev,gcp]"`.

```{warning}
**uv only.** The `cudaNNN` extras are no-ops under plain `pip` because pip ignores `[tool.uv.sources]` in `pyproject.toml`. A pip install of `oumi[cuda128]` will silently pull the CPU torch wheel. For pip, pass an explicit index:

​```bash
pip install "oumi[torch29,tf4]" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple
​```

If you suspect this happened, `oumi env` will warn you when CUDA hardware is detected but the installed torch is CPU-only.
```

**CUDA 12.2 / 12.3 / 12.4 / 12.5 drivers**: use the `cuda126` extra. PyTorch dropped cu124 wheels after torch 2.6, so cu126 is the lowest available index for the `torch29` stack; cu126 wheels are forward-compatible via CUDA enhanced compatibility.

These extras cover the two regimes the codebase actively supports (see `src/oumi/utils/packaging.py` for the runtime version branches). When vLLM ships a release that supports transformers v5, a combined `torch2X + tf5` stack will be added.

## Optional Dependencies

Oumi OSS has several optional features that require additional dependencies:

- For GPU support without pinning a specific torch minor (resolver picks the
  newest compatible versions in the wide base bands — convenient for one-off
  installs, but use a `torch{NN}` extra above if you want reproducibility):

  ```bash
  uv pip install "oumi[gpu]"  # Only if you have an Nvidia or AMD GPU
  ```

- For development and testing:

  ```bash
  uv pip install "oumi[dev]"
  ```

- For specific cloud providers:

  ```bash
  uv pip install "oumi[aws]"     # For Amazon Web Services
  uv pip install "oumi[azure]"   # For Microsoft Azure
  uv pip install "oumi[gcp]"     # For Google Cloud Platform
  uv pip install "oumi[lambda]"  # For Lambda Cloud
  uv pip install "oumi[runpod]"  # For RunPod
  ```

  You can install multiple cloud dependencies by combining them, e.g.:

  ```bash
  uv pip install "oumi[aws,azure,gcp]"
  ```

## Verifying the Installation

After installation, you can verify that Oumi OSS is installed correctly by running:

```bash
oumi --help
```

This should print the help message for Oumi OSS.

### Using Docker

You can also use the oumi docker images:

```bash
# Pull the latest image
docker pull ghcr.io/oumi-ai/oumi:latest

# Run oumi
docker run --gpus all -it --rm ghcr.io/oumi-ai/oumi:latest oumi --help
```

For comprehensive Docker usage, including volume mounting, training, evaluation, and inference examples, see the {doc}`Docker guide </user_guides/train/environments/docker>`.

## Troubleshooting

If you encounter any issues during installation, please check the [troubleshooting guide](/faq/troubleshooting.md).

If you're still having problems, please [open an issue](https://github.com/oumi-ai/oumi/issues) on our GitHub repository, or send us a message on [Discord](https://discord.gg/oumi).

## Next Steps

Now that you have Oumi OSS installed, you can proceed to the [Quickstart Guide](quickstart.md) to begin using the library.
