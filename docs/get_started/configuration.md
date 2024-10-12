# Configuration

```{contents}
:local:
:depth: 2
```

## Introduction to Oumi Configuration

Oumi uses YAML configuration files to specify various aspects of model training, evaluation, and inference. These configuration files allow you to customize your experiments without modifying the code.

## Configuration File Structure

A typical Oumi configuration file is structured into several main sections:

- `model`: Specifies the model architecture and parameters
- `data`: Defines the datasets for training and evaluation
- `training`: Sets training hyperparameters and options
- `evaluation`: Configures evaluation settings
- `inference`: Specifies inference parameters

## Common Configuration Options

### Model Configuration

```yaml
model:
  model_name: "gpt2"  # Name of the pre-trained model to use
  model_revision: "main"  # Specific model revision (if applicable)
  trust_remote_code: false  # Whether to trust remote code when loading the model
```

### Data Configuration

```yaml
data:
  train:
    datasets:
      - dataset_name: "tatsu-lab/alpaca"
        split: "train"
  validation:
    datasets:
      - dataset_name: "tatsu-lab/alpaca"
        split: "validation"
```

### Training Configuration

```yaml
training:
  output_dir: "output"
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  num_train_epochs: 3
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 500
  logging_steps: 100
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 1000
```

### Evaluation Configuration

```yaml
evaluation:
  metric_for_best_model: "accuracy"
  greater_is_better: true
  load_best_model_at_end: true
```

### Inference Configuration

```yaml
inference:
  max_new_tokens: 100
  do_sample: true
  temperature: 0.7
  top_p: 0.9
```

## Advanced Configuration

### Custom Datasets

To use a custom dataset, you need to register it and then reference it in your configuration:

```yaml
data:
  train:
    datasets:
      - dataset_name: "my_custom_dataset"
        split: "train"
```

### Distributed Training

For multi-GPU or multi-node training, you can specify additional parameters:

```yaml
training:
  deepspeed: "ds_config.json"  # Path to DeepSpeed config file
  fp16: true  # Enable mixed precision training
```

## Best Practices

1. Version control your configuration files along with your code.
2. Use environment variables for sensitive information (e.g., API keys).
3. Create separate configuration files for different experiments or model versions.
4. Comment your configuration files to explain non-obvious choices.

## Troubleshooting

Common configuration issues include:

- Typos in parameter names
- Incorrect indentation in YAML files
- Mismatched data types (e.g., using a string instead of an integer)

If you encounter issues, double-check your configuration file for these common mistakes.

## Next Steps

- Explore the [Configuration Examples](https://github.com/oumi-ai/oumi/tree/main/configs/oumi) in the Oumi repository.
- Learn about [Custom Datasets](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Datasets%20Tutorial.ipynb) to extend Oumi's capabilities.
- Check out the [API Reference](../api/index.md) for detailed information on configuration options.
