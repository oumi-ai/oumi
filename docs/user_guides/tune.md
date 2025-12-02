# Hyperparameter Tuning

(hyperparameter-tuning)=

## Introduction

Finding the right hyperparameters can make the difference between a mediocre model and state-of-the-art performance. Oumi provides `oumi tune`, a built-in hyperparameter optimization module powered by [Optuna](https://optuna.org/) that makes systematic hyperparameter search effortless.

With `oumi tune`, you can:

- 🔍 **Automatic Search**: Systematically search through hyperparameter spaces using advanced algorithms (TPE, random sampling)
- 🎯 **Multi-Objective Optimization**: Optimize for multiple metrics simultaneously (e.g., minimize loss while maximizing accuracy)
- 📊 **Smart Sampling**: Use log-uniform sampling for learning rates, categorical choices for optimizers, and more
- 💾 **Full Tracking**: Automatically save results, best models, and detailed trial logs
- 🚀 **Easy Integration**: Works seamlessly with all Oumi training workflows

## Installation

To use the hyperparameter tuning feature, install Oumi with the `tune` extra:

```bash
pip install oumi[tune]
```

This installs Optuna and related dependencies needed for hyperparameter tuning.

## Quick Start

### Basic Usage

Create a tuning configuration file (`tune.yaml`):

```yaml
model:
  model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
  model_max_length: 2048
  torch_dtype_str: "bfloat16"

data:
  train:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train[90%:]"
  validation:
    datasets:
      - dataset_name: "yahma/alpaca-cleaned"
        split: "train[:10%]"

tuning:
  n_trials: 10

  # Define hyperparameters to search
  tunable_training_params:
    learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-2

  # Fixed parameters (not tuned)
  fixed_training_params:
    trainer_type: TRL_SFT
    per_device_train_batch_size: 1
    max_steps: 1000

  # Metrics to optimize
  evaluation_metrics: ["eval_loss"]
  evaluation_direction: ["minimize"]

  tuner_type: OPTUNA
  tuner_sampler: "TPESampler"
```

Run tuning with a single command:

```bash
oumi tune -c tune.yaml
```

## Configuration

### Parameter Types

Oumi supports several parameter types for defining search spaces:

#### Categorical Parameters

Choose from a discrete set of options:

```yaml
tunable_training_params:
  optimizer:
    type: categorical
    choices: ["adamw_torch", "sgd", "adafactor"]
```

#### Integer Parameters

Sample integers within a range:

```yaml
tunable_training_params:
  gradient_accumulation_steps:
    type: int
    low: 1
    high: 8
```

#### Float Parameters (Uniform)

Sample floats uniformly within a range:

```yaml
tunable_training_params:
  warmup_ratio:
    type: uniform
    low: 0.0
    high: 0.3
```

#### Float Parameters (Log-Uniform)

Sample floats on a logarithmic scale (ideal for learning rates):

```yaml
tunable_training_params:
  learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-2
```

### Training Parameters

You can tune any training parameter by adding it to `tunable_training_params`:

```yaml
tuning:
  tunable_training_params:
    learning_rate:
      type: loguniform
      low: 1e-5
      high: 1e-2

    per_device_train_batch_size:
      type: categorical
      choices: [2, 4, 8]

    num_train_epochs:
      type: int
      low: 1
      high: 5

    weight_decay:
      type: uniform
      low: 0.0
      high: 0.1
```

### PEFT Parameters

For efficient fine-tuning with LoRA or QLoRA, tune PEFT parameters:

```yaml
tuning:
  tunable_peft_params:
    lora_r:
      type: categorical
      choices: [4, 8, 16, 32]

    lora_alpha:
      type: categorical
      choices: [8, 16, 32, 64]

    lora_dropout:
      type: uniform
      low: 0.0
      high: 0.1

  fixed_peft_params:
    q_lora: false
    lora_target_modules: ["q_proj", "v_proj"]
```

### Multi-Objective Optimization

Optimize for multiple metrics simultaneously:

```yaml
tuning:
  # Multiple evaluation metrics
  evaluation_metrics: ["eval_loss", "eval_mean_token_accuracy"]
  evaluation_direction: ["minimize", "maximize"]

  # Optuna will find the Pareto frontier of trials
```

When using multi-objective optimization, use `get_best_trials()` (plural) instead of `get_best_trial()` to retrieve the Pareto-optimal trials.

## Tuner Configuration

### Tuner Type

Currently, Oumi supports the Optuna tuner:

```yaml
tuning:
  tuner_type: OPTUNA
```

### Samplers

Choose from different sampling strategies:

#### TPE Sampler (Recommended)

Tree-structured Parzen Estimator - efficient Bayesian optimization:

```yaml
tuning:
  tuner_sampler: "TPESampler"
```

#### Random Sampler

Simple random sampling (good baseline):

```yaml
tuning:
  tuner_sampler: "RandomSampler"
```

## Advanced Usage

### Custom Evaluation Metrics

You can define custom evaluation metrics to optimize:

```yaml
tuning:
  evaluation_metrics: ["eval_loss", "eval_accuracy", "custom_metric"]
  evaluation_direction: ["minimize", "maximize", "maximize"]

  custom_eval_metrics:
    - name: "custom_metric"
      function: "my_module.compute_custom_metric"
```

## Output and Results

### Output Structure

Tuning results are saved in the output directory:

```
tuning_output/
├── trials_results.csv          # Summary of all trials
├── trial_0/                    # First trial
│   ├── checkpoint-100/
│   └── logs/
├── trial_1/                    # Second trial
│   ├── checkpoint-100/
│   └── logs/
└── ...
```

### Results CSV

The `trials_results.csv` file contains:

- Trial number
- Hyperparameter values for each trial
- Evaluation metrics for each trial
- Trial status (completed, failed, etc.)

### Best Model

The best model checkpoint is saved in the trial directory with the best evaluation metric(s).

## See Also

- {doc}`/user_guides/train/train` - Training guide
- {doc}`/resources/recipes` - Pre-configured recipes
- [Optuna Documentation](https://optuna.readthedocs.io/) - Optuna's official documentation
- {doc}`/api/oumi.core.tuners` - API reference for tuners
