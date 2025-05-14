# Local Training

This guide covers how to train models on your local machine or server using Oumi's command-line interface. Whether you're working on a laptop or a multi-GPU server, this guide will help you get started with local training.

For cloud-based training options, see {doc}`/user_guides/launch/launch`.

## Prerequisites

Before starting local training, ensure you have:

1. **Hardware Requirements**
   - CUDA-capable GPU(s) recommended
   - Sufficient RAM (16GB minimum)
   - Adequate disk space for storing your models and datasets

2. **Software Setup**
   - Python environment configured & `oumi` installed

For detailed installation instructions, refer to our {doc}`/get_started/installation` guide.

## Basic Usage

### Command Line Interface

The main command for training is `oumi train`. The CLI provides a flexible way to configure your training runs through both YAML configs and command-line parameter overrides.

```bash
# Basic usage
oumi train -c path/to/config.yaml

# With parameter overrides
oumi train -c path/to/config.yaml \
  --training.learning_rate 1e-4 \
  --training.num_train_epochs 5
```

For a complete reference of configuration options, see {doc}`/user_guides/train/configuration`.

## Training with GPUs

Oumi supports both single and multi-GPU training setups.

### Single GPU Training

For training on a specific GPU:

```bash
# Using CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 oumi train -c config.yaml

# Using device parameter
oumi train -c config.yaml --model.device_map cuda:0
```

### Multi-GPU Training

For distributed training across multiple GPUs:

```bash
# Using DDP
torchrun --standalone --nproc-per-node=<NUM_GPUS> oumi train -c config.yaml

# Using FSDP
oumi distributed torchrun -m oumi train -c config.yaml --fsdp.enable_fsdp true
```

For more details on distributed training options, see {doc}`/user_guides/train/train`.

## Monitoring

Effective monitoring is crucial for understanding your model's training progress. You have multiple options to monitor your training progress:

### Terminal Output

Monitor training progress directly in the terminal:

```bash
# Configure logging
oumi train -c config.yaml --training.logging_steps 10
```

### TensorBoard

Monitor metrics with TensorBoard for rich visualizations:

First add the following to your `train.yaml` config file:

```yaml
training:
  enable_tensorboard: true
  output_dir: oumi_output_dir
  logging_steps: 10
```

Then run the following command to start TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir oumi_output_dir
```

### Weights & Biases

You can also track experiments with W&B for collaborative projects. Make sure to [set up](https://oumi.ai/docs/latest/development/dev_setup.html#optional-set-up-weights-and-biases)
W&B on your local machine first.

```yaml
training:
  enable_wandb: true
  run_name: "experiment-1"
  logging_steps: 10
```

For more monitoring options and best practices, see {doc}`/user_guides/train/monitoring`.

## Next Steps

- Set up {doc}`monitoring tools </user_guides/train/monitoring>` for tracking progress
- Check out {doc}`configuration options </user_guides/train/configuration>` for detailed settings
- Seamlessly scale up your job to run on {doc}`cloud clusters </user_guides/launch/launch>`
