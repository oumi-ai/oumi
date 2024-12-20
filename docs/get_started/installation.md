# Installation

This guide will help you install Oumi and its dependencies.

## Requirements

Before installing Oumi, ensure you have the following:

- Python 3.9 or later
- pip (Python package installer)
- Git (for cloning the repository)

We recommend using a virtual environment to install Oumi. You can find instructions for setting up a conda environment in the {doc}`development/dev_setup` guide.

## Installation Methods

You can install Oumi using one of the following methods:

### 1. Install from Source (Recommended)

For the latest development version, you can install Oumi directly from the GitHub repository:

::::{tab-set-code}
:::{code-block} SSH
pip install git+ssh://git@github.com/oumi-ai/oumi.git
:::

:::{code-block} HTTP
pip install git+<https://github.com/oumi-ai/oumi.git>
:::
::::

### 2. Clone and Install

If you want to contribute to Oumi or need the full source code, you can clone the repository and install it:

```bash
git clone https://github.com/oumi-ai/oumi.git
cd oumi
pip install -e ".[dev]"
```

The `-e` flag installs the project in "editable" mode. This means that changes made to the source code will be immediately reflected in the installed package without needing to reinstall it. This is particularly helpful when you're actively developing features and want to test your changes quickly. It creates a link to the project's source code instead of copying the files, allowing you to modify the code and see the effects immediately in your Python environment.

### 3. Install from PyPI

To install the latest stable version of Oumi, run:

```bash
pip install oumi
```

## Optional Dependencies

Oumi has several optional features that require additional dependencies:

- For GPU support:

  ```bash
  pip install ".[gpu]"  # Only if you have an Nvidia or AMD GPU
  ```

- For development and testing:

  ```bash
  pip install ".[dev]"
  ```

- For specific cloud providers:

  ```bash
  pip install ".[aws]"     # For Amazon Web Services
  pip install ".[azure]"   # For Microsoft Azure
  pip install ".[gcp]"     # For Google Cloud Platform
  pip install ".[lambda]"  # For Lambda Cloud
  pip install ".[runpod]"  # For RunPod
  ```

  You can install multiple cloud dependencies by combining them, e.g.:

  ```bash
  pip install oumi[aws,azure,gcp]
  ```

## Verifying the Installation

After installation, you can verify that Oumi is installed correctly by running:

```bash
oumi --help
```

This should print the help message for Oumi.

## Troubleshooting

If you encounter any issues during installation, please check the [troubleshooting guide](../faq/troubleshooting.md).

If you're still having problems, please [open an issue](https://github.com/oumi-ai/oumi/issues) on our GitHub repository, or send us a message on [Discord](https://discord.gg/S74NxTDh7v).

## Next Steps

Now that you have Oumi installed, you can proceed to the [Quickstart Guide](quickstart.md) to begin using the library.
