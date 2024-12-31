# Notebook Integration

This guide covers how to use Oumi in `Jupyter` notebooks or `Google Colab` for interactive model training and experimentation.

## Setup

### 1. Install Requirements

You can install `oumi` with `Jupyter` in two ways:

Option 1: Install everything at once with dev dependencies:
```bash
# Install Oumi with development dependencies (includes Jupyter)
pip install oumi[dev]
```

Option 2: Install Jupyter and Oumi separately:
```bash
# Install Jupyter
pip install jupyterlab ipykernel

# Install Oumi
pip install oumi
```

If you're using `conda`, then register the Jupyter kernel:
```bash
# Note: If you're using conda, 'oumi' should be the name of your conda environment
# If you're not using conda, you can still use this name or choose a different one
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

- Learn about {doc}`common workflows </user_guides/train/common_workflows>` for better performance
- Set up {doc}`monitoring tools </user_guides/train/monitoring>` for tracking progress
- Check out {doc}`configuration options </user_guides/train/configuration>` for detailed settings
- Explore {doc}`VSCode integration </user_guides/train/environments/vscode>` for a full IDE experience
