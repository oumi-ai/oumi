# Notebook Integration

This guide covers how to use Oumi in `Jupyter` notebooks, `VSCode`, and `Google Colab` for interactive model training and experimentation.

## Jupyter Setup

### 1. Install Requirements

You can install `oumi` with `Jupyter` in two ways:

**Option 1:** Install everything at once with dev dependencies:

```bash
# Install Oumi with development dependencies (includes Jupyter)
pip install oumi[dev]
```

**Option 2:** Install Jupyter and Oumi separately:

```bash
# Install Jupyter
pip install jupyterlab ipykernel

# Install Oumi
pip install oumi
```

If you're using `conda`, you also need to register the Jupyter kernel. This allows you to easily use the kernel in your Jupyter notebooks:

```bash
# Note: replace "oumi" with the name of your environment
python -m ipykernel install --user --name oumi
```

### 2. Launch Jupyter

Start Jupyter Lab (recommended) or Notebook:

```bash
# Jupyter Lab (recommended)
jupyter lab

# Classic Notebook
jupyter notebook
```

When creating a new notebook, select the "oumi" kernel from the kernel selector.

## VSCode Setup

[Notebooks in the Oumi repository](https://github.com/oumi-ai/oumi/tree/main/notebooks) can be run directly in VSCode on your local machine. Make sure to select the `oumi` Conda environment as the kernel when first running the notebook.

It's also possible to use VSCode to run notebooks backed by a cloud node, if you need more powerful GPUs for your workload. For example, to create and connect to a GCP node with 4 A100s, run:

```shell
make gcpcode ARGS="--resources.accelerators A100:4"
```

This command is defined in our [Makefile](https://github.com/oumi-ai/oumi/blob/main/Makefile), and uses the {doc}`Oumi launcher </user_guides/launch/launch>` to create the remote node. Edit the `ARGS` to adjust the accelerators and remote cloud to your needs; see the {py:class}`~oumi.core.configs.JobConfig` class for an overview of configurable parameters.

After the new VSCode window backed by the remote node is open, install the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) on the remote VSCode instance.  Then, select "Python Environments..." after trying to run your notebook in order to select the correct kernel.

```{tip}
To automatically install the Jupyter extension every time you open a remote VSCode instance, add the following line to your [VSCode user settings JSON file](https://code.visualstudio.com/docs/getstarted/settings#_settings-json-file): `"remote.SSH.defaultExtensions": ["ms-toolsai.jupyter"],`. You could also add other useful extensions like `"charliermarsh.ruff"` and `"ms-python.python"`.
```

## Training Workflow

### 1. Initialize Configuration

Start by importing necessary modules and loading your configuration:

```python
from oumi.core.configs import TrainingConfig
from oumi.builders import build_trainer

# Load configuration
config = TrainingConfig.from_yaml("path/to/config.yaml")
```

For configuration options, refer to the {doc}`/user_guides/train/configuration` guide.

You can find multiple examples of configurations in the {doc}`/resources/recipes` section.

### 2. Data Exploration

Notebooks are very useful for exploring and analyzing your datasets:

```python
from oumi.builders import build_tokenizer
from oumi.core.configs import ModelParams
from oumi.datasets import AlpacaDataset

# Initialize tokenizer and dataset
tokenizer = build_tokenizer(ModelParams(model_name="Qwen/Qwen2-1.5B-Instruct"))
dataset = AlpacaDataset(tokenizer=tokenizer)

# Print a few examples
for i in range(3):
    conversation = dataset.conversation(i)
    print(f"Example {i + 1}:")
    for message in conversation.messages:
        print(f"{message.role}: {message.content[:100]}...")  # Truncate for brevity
    print("\n")
```

For more details on datasets, see the {doc}`/resources/datasets/datasets` section.

### 3. Training

Start your training process:

```python
# Start training
trainer = build_trainer(config)
trainer.train()
```

For training best practices, see the {doc}`/user_guides/train/train` guide.

## Debugging Tips

### Managing GPU Memory

Managing resources is crucial in notebooks. Here's how to clean up.

Note that once a model is loaded in memory in a cell, it will stay there unless you explicitly clear the memory or restart the kernel to fully free up resources:

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Delete unused objects and collect garbage
del trainer
import gc
gc.collect()
```

### Using Magic Commands

Jupyter notebooks provide helpful magic commands for debugging and profiling:

**Memory & Variable Management:**

- `%who`, `%who_ls`, `%whos` - List variables in current namespace with varying detail levels

**Performance Profiling:**

- `%%time` - Time execution of an entire cell, `%time` - Time execution of a single line
- `%%memit` - Measure memory usage of an entire cell, `%memit` - Measure memory usage of a single line

**Documentation & Source Code:**

- `?object` or `object?` - Show object's docstring and basic info
- `??object` or `object??` - Show object's source code if available

**Debugging:**

- `%debug` - Enter debug mode after an exception
- `%pdb` - Enable automatic post-mortem debugging on exceptions

```{tip}
Running `%lsmagic` will list all available magic commands, and `%magic` will show detailed documentation.
```

## Next Steps

- Set up {doc}`monitoring tools </user_guides/train/monitoring>` for tracking progress
- Check out {doc}`configuration options </user_guides/train/configuration>` for detailed settings
- Explore {doc}`VSCode integration </user_guides/train/environments/vscode>` for a full IDE experience
