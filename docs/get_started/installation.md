# Installation

This guide will help you install Oumi and its dependencies.

## Requirements

❗NOTE: Since PyTorch dropped support for Intel Macs, you cannot install Oumi there. Consider running Oumi on free Colab GPU instances, using our {doc}`notebook tutorials </get_started/tutorials>`!

Before installing Oumi, ensure you have the following:

- Python 3.9 or later
- pip (Python package installer)
- Git (if cloning the repository; required for steps 1 and 2)

We recommend using a virtual environment to install Oumi. You can find instructions for setting up a Conda environment in the {doc}`/development/dev_setup` guide.

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
