# Installation

This guide will help you install Oumi and its dependencies.

## Requirements

❗NOTE: Since PyTorch dropped support for Intel Macs, you cannot install Oumi on those machines. Consider running Oumi on free Colab GPU instances, using our {doc}`notebook tutorials </get_started/tutorials>`!

Before installing Oumi, ensure you have the following:

- Python 3.9 or later
- pip (Python package installer)
- Git (if cloning the repository; required for steps 1 and 2)
- Adequate disk space (see [Storage Requirements](#storage-requirements) below)

We recommend using a virtual environment to install Oumi. You can find instructions for setting up a Conda environment in the {doc}`/development/dev_setup` guide.

### Storage Requirements

Oumi requires sufficient disk space for models, datasets, checkpoints, and cache directories. The amount of storage needed depends on your specific use case, but here are general guidelines:

#### Minimum Storage Requirements

For basic usage (small model inference and evaluation):

- **~50 GB minimum** - Enough for:
  - Oumi installation and dependencies (~5-10 GB)
  - Small models (1B-3B parameters) (~2-8 GB per model)
  - Small datasets (~1-5 GB)
  - Model cache and temporary files (~10-20 GB)

#### Recommended Storage for Training

For training and fine-tuning workflows:

- **200-500 GB recommended** - Sufficient for:
  - Multiple model checkpoints during training
  - Medium-sized models (7B-13B parameters)
  - Moderate-sized datasets
  - Training artifacts and logs

#### Large-Scale Usage

For production deployments or large-scale training:

- **1-5 TB recommended** - Required for:
  - Large models (70B+ parameters)
  - Multiple large model variants
  - Large pre-training datasets (100GB+)
  - Multiple training checkpoints
  - Extensive experiment logs and artifacts

#### Storage Breakdown by Component

Here's what contributes to storage usage:

**Model Weights** (cached at `~/.cache/huggingface/hub` by default):

| Model Size | BF16/FP16 | FP32 | Q4 Quantized |
|-----------|-----------|------|-------------|
| 1B        | ~2 GB     | ~4 GB | ~0.5 GB |
| 3B        | ~6 GB     | ~12 GB | ~1.5 GB |
| 7B        | ~14 GB    | ~28 GB | ~3.5 GB |
| 13B       | ~26 GB    | ~52 GB | ~6.5 GB |
| 33B       | ~66 GB    | ~132 GB | ~16.5 GB |
| 70B       | ~140 GB   | ~280 GB | ~35 GB |

**Training Checkpoints**: Each checkpoint is approximately the same size as the model weights. By default, Oumi may save multiple checkpoints during training. Configure `training.save_total_limit` to control the number of checkpoints retained.

**Datasets**: Varies widely based on use case:
- Small fine-tuning datasets: 1-100 MB
- Medium datasets: 100 MB - 10 GB  
- Large pre-training datasets: 10 GB - 1 TB+

**Cache and Temporary Files**: 10-50 GB for intermediate processing, compilation caches, and temporary artifacts.

#### Managing Storage

To manage storage effectively:

1. **Configure cache directories**: Set `HF_HOME` or `TRANSFORMERS_CACHE` environment variables to control where models are cached:

   ```bash
   export HF_HOME=/path/to/large/disk/.cache/huggingface
   ```

2. **Clean up cache**: Use Oumi's cache management tools to remove unused models:

   ```bash
   oumi cache ls  # List cached models
   oumi cache rm <model-name>  # Remove specific cached model
   ```

3. **Limit checkpoints**: Configure checkpoint retention in your training config:

   ```yaml
   training:
     save_total_limit: 2  # Keep only the 2 most recent checkpoints
   ```

4. **Use cloud storage**: For remote training, mount cloud storage (GCS, S3, Azure Blob) to avoid filling local disks. See {doc}`/user_guides/launch/launch` for details.

5. **Use quantization**: Employ quantized models (Q4, Q8) to reduce storage requirements by 2-4x with minimal quality impact.

## Installation Methods

You can install Oumi using one of the following methods:

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

Once that's done, you're ready to install Oumi!

To install the latest stable version of Oumi, run:

```bash
pip install oumi
```

### 2. Install from Source

For the latest development version, you can install Oumi directly from the GitHub repository:

::::{tab-set}
:::{tab-item} SSH

```{code-block} shell
pip install git+ssh://git@github.com/oumi-ai/oumi.git
```

:::

:::{tab-item} HTTPS

```{code-block} shell
pip install git+https://github.com/oumi-ai/oumi.git
```

:::
::::

### 3. Clone and Install

If you want to contribute to Oumi or need the full source code, you can clone the repository and install it:

::::{tab-set}
:::{tab-item} SSH

```{code-block} shell
git clone git@github.com:oumi-ai/oumi.git
cd oumi
pip install -e ".[dev]"
```

:::

:::{tab-item} HTTPS

```{code-block} shell
git clone https://github.com/oumi-ai/oumi.git
cd oumi
pip install -e ".[dev]"
```

:::
::::

For more information on setting up your dev environment for contributing to Oumi, please
see our [dev setup guide](../development/dev_setup.md).

The `-e` flag installs the project in "editable" mode. This means that changes made to the source code will be immediately reflected in the installed package without needing to reinstall it. This is particularly helpful when you're actively developing features and want to test your changes quickly. It creates a link to the project's source code instead of copying the files, allowing you to modify the code and see the effects immediately in your Python environment.

## Optional Dependencies

Oumi has several optional features that require additional dependencies:

- For GPU support:

  ```bash
  pip install "oumi[gpu]"  # Only if you have an Nvidia or AMD GPU
  ```

- For development and testing:

  ```bash
  pip install "oumi[dev]"
  ```

- For specific cloud providers:

  ```bash
  pip install "oumi[aws]"     # For Amazon Web Services
  pip install "oumi[azure]"   # For Microsoft Azure
  pip install "oumi[gcp]"     # For Google Cloud Platform
  pip install "oumi[lambda]"  # For Lambda Cloud
  pip install "oumi[runpod]"  # For RunPod
  ```

  You can install multiple cloud dependencies by combining them, e.g.:

  ```bash
  pip install "oumi[aws,azure,gcp]"
  ```

## Verifying the Installation

After installation, you can verify that Oumi is installed correctly by running:

```bash
oumi --help
```

This should print the help message for Oumi.

## Troubleshooting

If you encounter any issues during installation, please check the [troubleshooting guide](/faq/troubleshooting.md).

If you're still having problems, please [open an issue](https://github.com/oumi-ai/oumi/issues) on our GitHub repository, or send us a message on [Discord](https://discord.gg/oumi).

## Next Steps

Now that you have Oumi installed, you can proceed to the [Quickstart Guide](quickstart.md) to begin using the library.
