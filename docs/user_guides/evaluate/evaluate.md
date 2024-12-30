# Evaluation

```{toctree}
:maxdepth: 2
:caption: Evaluation
:hidden:

generative
custom_lm_harness
```

## Overview

Oumi provides comprehensive evaluation capabilities through multiple benchmark types and frameworks, allowing you to assess language models across various dimensions and tasks. The framework is designed for reproducibility and extensibility, featuring batch processing optimization, flexible configuration system, and comprehensive experiment tracking through Weights & Biases integration.

All evaluations are automatically logged and versioned, capturing model configurations, evaluation parameters, and environmental details to ensure reproducible results. The framework supports both local execution and distributed evaluation for larger experiments.

### Benchmark Types

| Type | Description | When to Use | Get Started |
|------|-------------|-------------|-------------|
| **Generative Benchmarks** | Evaluate model's ability to generate contextual responses | Best for assessing instruction-following capabilities, response quality, and multi-turn dialogue performance | See [MT-bench notebook](/../notebooks/mt_bench.ipynb) and [AlpacaEval notebook](/../notebooks/alpaca_eval.ipynb) |
| **Multiple Choice (LM-Eval)** | Assess knowledge and reasoning through structured questions | Ideal for measuring factual knowledge, reasoning capabilities, and performance on established benchmarks | See [Basic Configuration](#basic-configuration) and [Supported Tasks](#supported-tasks) |
| **LLM as Judge** | Qualitative assessment using LLMs | Perfect for subjective evaluation of response quality, safety, and alignment with custom criteria | See {doc}`Judge documentation </user_guides/judge/judge>` |

```{tip}
When using LM-Eval harness, you can set `batch_size: "auto"` in your configuration to automatically optimize batch size based on your GPU memory.
```

## Evaluation Considerations

### Evaluating Models with Adapters

When evaluating models with adapters (LoRA, QLoRA, etc.), special considerations are needed:

```yaml
model:
  model_name: "microsoft/phi-2"
  adapter_model: "path/to/adapter"
  adapter_type: "lora"  # or "qlora"
  merge_adapters: true  # Set to true for inference optimization
```

### Impact of Quantization

It's crucial to evaluate models both before and after quantization to understand the performance impact:

1. Evaluate the base model first as a baseline
2. Evaluate after quantization to measure accuracy loss
3. Consider trade-offs between model size and performance

Example configuration for quantized model evaluation:

```yaml
model:
  model_name: "microsoft/phi-2"
  quantization:
    bits: 4
    group_size: 128

lm_harness_params:
  tasks: ["mmlu"]  # Use same tasks for fair comparison
```

### Distributed Evaluatio
For large-scale evaluations:

```bash
accelerate launch -m oumi evaluate -c config.yaml
```


## Quick Start

### Using the CLI

The simplest way to evaluate a model is through the Oumi CLI:

```bash
oumi evaluate -c configs/oumi/phi3.eval.lm_harness.yaml
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

lm_harness_params:
  tasks:
    - "huggingface_leaderboard_v1"
  num_fewshot: 0
  num_samples: 100
```

### Advanced Configuration

For more complex evaluations, you can specify additional parameters:

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True
  adapter_model: "path/to/adapter"  # Optional: For adapter-based models

lm_harness_params:
  tasks:
    - "huggingface_leaderboard_v1"
    - "mmlu"
    - "hellaswag"
  num_fewshot: 5
  num_samples: 1000

generation:
  batch_size: 16
  max_new_tokens: 512
  temperature: 0.0

enable_wandb: true
output_dir: "evaluation_results"
run_name: "phi3-evaluation"
tasks: # HuggingFace Leaderboard
  - evaluation_platform: lm_harness
    task_name: mmlu
    eval_kwargs:
      num_fewshot: 5
  - evaluation_platform: lm_harness
    task_name: arc_challenge
    eval_kwargs:
      num_fewshot: 25
  - evaluation_platform: lm_harness
    task_name: winogrande
    eval_kwargs:
      num_fewshot: 5
  - evaluation_platform: lm_harness
    task_name: hellaswag
    eval_kwargs:
      num_fewshot: 10
  - evaluation_platform: lm_harness
    task_name: truthfulqa_mc2
    eval_kwargs:
      num_fewshot: 0
  - evaluation_platform: lm_harness
    task_name: gsm8k
    eval_kwargs:
      num_fewshot: 5
```

```{note}
Adjust the parameters according to your specific evaluation needs. For a complete list of configuration options, refer to {py:class}`~oumi.core.configs.Evaluation` class.
```

#### Configuration Options

- `model`: Model-specific configuration
  - `model_name`: HuggingFace model identifier or local path
  - `trust_remote_code`: Whether to trust remote code (for custom models)
  - `adapter_model`: Path to adapter weights (optional)

- `lm_harness_params`: LM Evaluation Harness parameters
  - `tasks`: List of tasks to evaluate
  - `num_fewshot`: Number of few-shot examples (0 for zero-shot)
  - `num_samples`: Number of samples to evaluate

```{tip}
To use `lm_eval` with `oumi`:
1. Specify the tasks in your configuration file under `tasks`.
2. Adjust other parameters like `num_fewshot` and `num_samples` as needed.
```

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

### Local Results

To build a custom evaluation task, refer to the {doc}`Custom Evaluation Tasks </user_guides/evaluate/custom_lm_harness>`.

Results are saved in the specified `output_dir` with the following files:

- `lm_harness_{timestamp}_results.json`: Detailed evaluation results
- `lm_harness_{timestamp}_task_config.json`: Task configuration
- `lm_harness_{timestamp}_evaluation_config.yaml`: Evaluation configuration
- `lm_harness_{timestamp}_package_versions.json`: Package version information

### Weights & Biases Integration

When `enable_wandb` is true, results are automatically logged to W&B:

```python
# Environment variable for W&B project name
os.environ["WANDB_PROJECT"] = "my-evaluation-project"
```

## API Reference

See the {py:class}`~oumi.core.configs.Evaluation` class for complete configuration options.

For programmatic usage, refer to {py:func}`~oumi.evaluate` function documentation.
