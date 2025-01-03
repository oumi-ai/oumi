# Evaluation

```{toctree}
:maxdepth: 2
:caption: Evaluation
:hidden:

mcqa
generative
custom_lm_harness
```

## Overview

Oumi provides comprehensive evaluation capabilities through multiple benchmark types and frameworks, allowing you to assess language models across various dimensions and tasks. The framework is designed for reproducibility and extensibility, featuring batch processing optimization, flexible configuration system, and comprehensive experiment tracking through Weights & Biases integration.

All evaluations are automatically logged and versioned, capturing model configurations, evaluation parameters, and environmental details to ensure reproducible results. The framework supports both local execution and distributed evaluation for larger experiments.

### Benchmark Types

| Type | Description | When to Use | Get Started |
|------|-------------|-------------|-------------|
| **Generative Benchmarks** | Evaluate model's ability to generate contextual responses | Best for assessing instruction-following capabilities, response quality, and multi-turn dialogue performance | See {doc}`Generative Evaluation page </user_guides/evaluate/generative>` |
| **Multiple Choice (LM-Eval)** | Assess knowledge and reasoning through structured questions | Ideal for measuring factual knowledge, reasoning capabilities, and performance on established benchmarks | See [Basic Configuration](#basic-configuration) and [Supported Tasks](#supported-tasks) |
| **LLM as Judge** | Qualitative assessment using LLMs | Perfect for subjective evaluation of response quality, safety, and alignment with custom criteria | See {doc}`Judge documentation </user_guides/judge/judge>` |

## Quick Start

### Using the CLI

The simplest way to evaluate a model is through the Oumi CLI:

```bash
oumi evaluate -c configs/oumi/phi3.eval.lm_harness.yaml
```

To run evaluation with multiple GPUs:
```bash
oumi distributed torchrun -m oumi evaluate -c configs/oumi/phi3.eval.lm_harness.yaml
```

### Using the Python API

For more programmatic control, you can use the Python API:

```python
from oumi import evaluate
from oumi.core.configs import EvaluationConfig

# Load configuration from YAML
config = EvaluationConfig.from_yaml("configs/oumi/phi3.eval.lm_harness.yaml")

# Run evaluation
evaluate(config)
```

## Configuration

### Basic Configuration

A minimal evaluation configuration file looks like this:

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True

tasks:
  - evaluation_platform: lm_harness
    task_name: huggingface_leaderboard_v1
    num_fewshot: 0
    num_samples: 100

generation:
  batch_size: "auto"  # Let LM Harness optimize batch size
  max_new_tokens: 512
  temperature: 0.0

enable_wandb: true
output_dir: "evaluation_results"
run_name: "phi3-evaluation"
```

### Advanced Configuration

For more complex evaluations, you can specify multiple tasks:

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True
  adapter_model: "path/to/adapter"  # Optional: For adapter-based models

tasks:
  # LM Harness Tasks
  - evaluation_platform: lm_harness
    task_name: mmlu
    num_fewshot: 5
    num_samples: 100
  - evaluation_platform: lm_harness
    task_name: arc_challenge
    num_fewshot: 25
  - evaluation_platform: lm_harness
    task_name: hellaswag
    num_fewshot: 10

# AlpacaEval Task
  - evaluation_platform: alpaca_eval
    version: 2.0  # or 1.0
    num_samples: 100

generation:
  batch_size: 16
  max_new_tokens: 512
  temperature: 0.0

enable_wandb: true
output_dir: "evaluation_results"
run_name: "phi3-evaluation"
```

#### Configuration Options

- `model`: Model-specific configuration
  - `model_name`: HuggingFace model identifier or local path
  - `trust_remote_code`: Whether to trust remote code (for custom models)
  - `adapter_model`: Path to adapter weights (optional)
  - `adapter_type`: Type of adapter ("lora" or "qlora")

- `tasks`: List of evaluation tasks
  - LM Harness Task Parameters:
    - `evaluation_platform`: "lm_harness"
    - `task_name`: Name of the LM Harness task
    - `num_fewshot`: Number of few-shot examples (0 for zero-shot)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

  - AlpacaEval Task Parameters:
    - `evaluation_platform`: "alpaca_eval"
    - `version`: AlpacaEval version (1.0 or 2.0)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

- `generation`: Generation parameters
  - `batch_size`: Batch size for inference ("auto" for automatic selection)
  - `max_new_tokens`: Maximum number of tokens to generate
  - `temperature`: Sampling temperature

- `enable_wandb`: Enable Weights & Biases logging
- `output_dir`: Directory for saving results
- `run_name`: Name of the evaluation run

## Supported Tasks

### HuggingFace Leaderboard Tasks

The `huggingface_leaderboard_v1` task suite includes:

- ARC (Challenge & Easy)
- HellaSwag
- MMLU
- TruthfulQA
- Winogrande
- GSM8K

### Additional Tasks

- BIG-bench
- LAMBADA
- PIQA
- SQuAD
- And many more...

To see all available tasks:

```bash
lm-eval --tasks list
```

## Results and Logging

### Evaluation Results

Results are saved in the specified `output_dir` with the following files:

- `lm_harness_{timestamp}_results.json`: Detailed evaluation metrics
- `lm_harness_{timestamp}_task_config.json`: Task configuration
- `lm_harness_{timestamp}_evaluation_config.yaml`: Evaluation configuration
- `lm_harness_{timestamp}_package_versions.json`: Package version information for reproducibility

### Weights & Biases Integration

When `enable_wandb` is true, results are automatically logged to W&B:

```python
# Environment variable for W&B project name
os.environ["WANDB_PROJECT"] = "my-evaluation-project"
```

## API Reference

- See the {py:class}`~oumi.core.configs.EvaluationConfig` class for complete configuration options.
- See {py:func}`~oumi.evaluate` function documentation for programmatic usage.
