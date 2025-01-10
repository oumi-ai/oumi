# Training

```{toctree}
:maxdepth: 2
:caption: Training
:hidden:

training_methods
environments/environments
configuration
monitoring
```

## Overview

Oumi provides an end-to-end training framework designed to handle everything from small fine-tuning experiments to large-scale pre-training runs.

Our goal is to allow you to start small, in a notebook or local machine, and easily scale up as your needs grow while maintaining a consistent interface across different training scenarios and environments.

Key features include:

- **Multiple Training Methods**: {ref}`Supervised Fine-Tuning (SFT) <supervised-fine-tuning-sft>` to adapt models to your specific tasks, {ref}`Vision-Language SFT <vision-language-sft>` for multimodal models, {ref}`Pretraining <pretraining>` for training from scratch, and {ref}`Direct Preference Optimization (DPO) <direct-preference-optimization-dpo>` for preference-based fine-tuning
- **Parameter-Efficient Fine-Tuning (PEFT) & Full Fine-Tuning (FFT)**: Support for multiple PEFT methods including LoRA for efficient adapter training, QLoRA for quantized fine-tuning with 4-bit precision, and full fine-tuning for maximum performance
- **Flexible Environments**: Train on {doc}`local machines <environments/local>`, with {doc}`VSCode integration <environments/vscode>`, in {doc}`Jupyter notebooks <environments/notebooks>`, or in a {doc}`cloud environment <environments/cloud>`
- **Production-Ready**: Ensure reproducibility through {doc}`YAML-based configurations <configuration>` and gain insights with comprehensive {doc}`monitoring & debugging tools <monitoring>`
- **Scalable Training**: Scale from single-GPU training to multi-node distributed training using DDP or FSDP

## Quick Start

The fastest way to get started with training is using one of our pre-configured recipes.

For example, to train a small model (`SmolLM-135M`) on a sample dataset (`tatsu-lab/alpaca`), you can use the following command:

::::{tab-set-code}
:::{code-block} bash
# Train a small model (SmolLM-135M)
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
:::

:::{code-block} python
from oumi.core.train import train
from oumi.core.configs import TrainingConfig

# Load config from file
config = TrainingConfig.from_yaml("configs/recipes/smollm/sft/135m/quickstart_train.yaml")

# Start training
train(config)
:::
::::

Running this config will:

1. Download a small pre-trained model: `SmolLM-135M`
2. Load a sample dataset: `tatsu-lab/alpaca`
3. Run supervised fine-tuning using the `TRL_SFT` trainer
4. Save the trained model to `config.output_dir`

## Configuration Guide

At the heart of Oumi's training system is a YAML-based configuration framework. This allows you to define all aspects of your training run in a single, version-controlled file.

Here's a basic example with key parameters explained:

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

You can override any value either through the CLI or programmatically:

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

In the following sections, we'll cover some common workflows for training.

### Fine-tuning a Pre-trained Model

The simplest workflow is to fine-tune a pre-trained model on a dataset. This will perform fully finetune the model using SFT (supervised fine-tuning).

```yaml
model:
  model_name: "meta-llama/Meta-Llama-3.2-3B-Instruct"  # Replace with your model
  trust_remote_code: true
  dtype: "bfloat16"

data:
  train:  # Training dataset mixture, can be a single dataset or a list of datasets
    datasets:
      - dataset_name: "yahma/alpaca-cleaned" # Replace with your dataset, or add more datasets
        split: "train"

training:
  output_dir: "output/llama-finetuned"  # Where to save outputs
  optimizer: "adamw_torch_fused"
  learning_rate: 2e-5
  max_steps: 10  # Number of training steps
```

### Multi-GPU Training

To train with multiple GPUs, we can extend that same configuration to use distributed training, using either DDP or FSDP:

```bash
# Using DDP (DistributedDataParallel)
oumi distributed torchrun -m \
  oumi train \
  -c configs/recipes/llama3_2/sft/3b_full/train.yaml

# Using FSDP (Fully Sharded Data Parallel)
oumi distributed torchrun -m \
  oumi train \
  -c configs/recipes/llama3_2/sft/3b_full/train.yaml \
  --fsdp.enable_fsdp true \
  --fsdp.sharding_strategy FULL_SHARD
```

### Launch Remote Training

To kick off a training run on a cloud environment, you can use the launcher system.

This will create a GCP job with the specified configuration and start training:

```bash
oumi launch up -c configs/recipes/llama3_2/sft/3b_full/gcp_job.yaml --cluster llama3b-sft
```

Thanks to the integration with [Skypilot](https://skypilot.readthedocs.io/en/latest/), most cloud providers are supported -- make sure to check out {doc}`/user_guides/launch/launch` for more details.

### Using Custom Datasets

To use your own datasets, you can specify the path to the dataset in the configuration.

```yaml
data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "/path/to/dataset.jsonl"
```

In this case, the dataset is expected to be in the `conversation` format. See {doc}`/resources/datasets/data_formats` for all the supported formats, and {doc}`/resources/datasets/custom_datasets` for customizing the preprocessing at runtime if needed.

## Training Output

Throughout the training process, we generate logs and artifacts to help you track progress and debug issues in the `config.output_dir` directory.

This includes model checkpoints for resuming training, detailed training logs, TensorBoard events for visualization, and a backup of the training configuration.

## Next Steps

Now that we covered the basics, as a next step you can:

- Learn about different {doc}`training methods <training_methods>`
- Set up your {doc}`training environment <environments/environments>` and get started training
- Explore {doc}`configuration options <configuration>`
