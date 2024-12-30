# Training

```{toctree}
:maxdepth: 2
:caption: Training
:hidden:

training_methods
environments/environments
configuration
common_workflows
monitoring
```

## Overview

Oumi provides a comprehensive training framework that supports:

- **Multiple Training Methods**: {ref}`Supervised Fine-Tuning (SFT) <supervised-fine-tuning-sft>` to adapt models to your specific tasks, {ref}`Vision-Language SFT <vision-language-sft>` for multimodal models, {ref}`Pretraining <pretraining>` for training from scratch, and {ref}`Direct Preference Optimization (DPO) <direct-preference-optimization-dpo>` for preference-based fine-tuning
- **Flexible Environments**: Train on {doc}`local machines <environments/local>`, with {doc}`VSCode integration <environments/vscode>`, or in {doc}`Jupyter notebooks <environments/notebooks>`
- **Scalable Training**: From single GPU to distributed training with DDP and FSDP support
- **Production Features**: {doc}`Configuration management <configuration>`, {doc}`common workflows <common_workflows>`, and {doc}`monitoring & debugging tools <monitoring>`

## Quick Start

The fastest way to start is using a pre-configured recipe:

::::{tab-set-code}
:::{code-block} bash
# Train a small model (SmolLM-135M)
oumi train -c configs/recipes/smollm/sft/135m/train_quickstart.yaml
:::

:::{code-block} python
from oumi.core.train import train
from oumi.core.configs import TrainingConfig

# Load config from file
config = TrainingConfig.from_yaml("configs/recipes/smollm/sft/135m/train_quickstart.yaml")

# Start training
train(config)
:::
::::

This will:

1. Download a small pre-trained model: `SmolLM-135M`
2. Load a sample dataset: `tatsu-lab/alpaca`
3. Run supervised fine-tuning using the `TRL_SFT` trainer
4. Save the trained model to `config.output_dir`

## Configuration Guide

Oumi uses YAML files for configuration, Here's a basic example with key parameters explained:

```yaml
model:
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"  # Base model to fine-tune
  trust_remote_code: true  # Required for some model architectures
  dtype: "bfloat16"  # Training precision (float32, float16, or bfloat16)

data:
  train:  # Training dataset mixture
    datasets:
      - dataset_name: "tatsu-lab/alpaca"  # Training dataset
        split: "train"  # Dataset split to use

training:
  output_dir: "output/my_training_run" # Where to save outputs
  num_train_epochs: 3 # Number of training epochs
  learning_rate: 5e-5 # Learning rate
  save_steps: 100  # Checkpoint frequency
```

You can override any config values via CLI or Python API:

::::{tab-set-code}
:::{code-block} bash
oumi train -c config.yaml \
  --training.learning_rate 1e-4 \
  --training.num_train_epochs 5
:::

:::{code-block} python
from oumi.core.configs import TrainingConfig
from oumitrain import train

# Load base config
config = TrainingConfig.from_yaml("config.yaml")

# Override specific values
config.training.learning_rate = 1e-4
config.training.num_train_epochs = 5

# Start training
train(config)
:::
::::

## Common Workflows

### Fine-tuning a Pre-trained Model

```yaml
model:
  model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  trust_remote_code: true
  dtype: "bfloat16"

data:
  train:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train"

training:
  output_dir: "output/llama-finetuned"
  optimizer: "adamw_torch_fused"
  learning_rate: 2e-5
  max_steps: 10
```

### Multi-GPU Training

```bash
# Using DDP (DistributedDataParallel)
oumi distributed torchrun -m oumi train \
  -c configs/recipes/llama3_2/sft/3b_full/train.yaml

# Using FSDP (Fully Sharded Data Parallel)
oumi distributed torchrun -m oumi train \
  -c configs/recipes/llama3_2/sft/3b_full/train.yaml \
  --fsdp.enable_fsdp true \
  --fsdp.sharding_strategy FULL_SHARD
```

### Launch Remote Training

```bash
oumi launch up -c configs/recipes/llama3_2/sft/3b_full/gcp_job.yaml --cluster llama3b-sft
```

### Using Local Datasets

```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft_jsonl"
        dataset_path: "/path/to/dataset.jsonl"
```

## Training Output

During training, Oumi will:

1. Create an output directory with:
   - Model checkpoints
   - Training logs
   - TensorBoard events
   - Config backup
2. Display progress in terminal
3. Log metrics (if configured)

## Next Steps

- Learn about different {doc}`training methods <training_methods>`
- Explore {doc}`configuration options <configuration>`
- Set up {doc}`distributed training <common_workflows>`
- Try different {doc}`training environments <environments/environments>`
