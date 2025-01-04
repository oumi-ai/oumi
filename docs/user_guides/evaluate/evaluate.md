# Evaluation

```{toctree}
:maxdepth: 2
:caption: Evaluation
:hidden:

standardized_benchmarks
generative_benchmarks
leaderboards
```

## Overview

Oumi provides comprehensive evaluation capabilities through multiple benchmark types and frameworks, allowing you to assess language models across various dimensions and tasks. The framework is designed for reproducibility and extensibility, featuring batch processing optimization, a flexible configuration system, and comprehensive experiment tracking through Weights & Biases integration.

All evaluations are automatically logged and versioned, capturing model configurations, evaluation parameters, and environmental details to ensure reproducible results. The framework supports both local execution and distributed evaluation for larger experiments.

### Benchmark Types

| Type | Description | When to Use | Get Started |
|------|-------------|-------------|-------------|
| **Standardized Benchmarks** | Assess model knowledge and reasoning capability through structured questions with predefined answers | Ideal for measuring factual knowledge, reasoning capabilities, and performance on established benchmarks | See {doc}`Standardized benchmarks page </user_guides/evaluate/standardized_benchmarks>` |
| **Open-Ended Generation** | Evaluate model's ability to effectively respond to open-ended questions | Best for assessing instruction-following capabilities, response quality, and conciseness | See {doc}`Generative benchmarks page </user_guides/evaluate/generative_benchmarks>` |
| **LLM as Judge** | Qualitative assessment using LLMs | Suitable for subjective evaluation of response quality, safety, and alignment with custom criteria | See {doc}`Judge documentation </user_guides/judge/judge>` |

## Quick Start

### Using the CLI

The simplest way to evaluate a model is by authoring a [yaml](https://github.com/oumi-ai/oumi/blob/main/configs/recipes/phi3/evaluation/eval.yaml) file, and calling the Oumi CLI:

```bash
oumi evaluate -c configs/recipes/phi3/evaluation/eval.yaml
```

To run evaluation with multiple GPUs:
```bash
oumi distributed torchrun -m oumi evaluate -c configs/recipes/phi3/evaluation/eval.yaml
```

### Using the Python API

For more programmatic control, you can use the Python API to load the {py:class}`~oumi.core.configs.EvaluationConfig` class:

```python
from oumi import evaluate
from oumi.core.configs import EvaluationConfig

# Load configuration from YAML
config = EvaluationConfig.from_yaml("configs/recipes/phi3/evaluation/eval.yaml")

# Run evaluation
evaluate(config)
```

## Configuration

### Basic Configuration

A minimal evaluation configuration file looks as follows. The `model_name` can be a HuggingFace model name or a local path to a model.

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True

tasks:
  - evaluation_platform: lm_harness
    task_name: mmlu

output_dir: "my_evaluation_results"
```

### Advanced Configuration

For more complex evaluations, you can specify multiple tasks. We recommend to browse all available options of the overall {py:class}`~oumi.core.configs.EvaluationConfig` class, as well as {py:class}`~oumi.core.configs.params.model_params.ModelParams`, {py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`, and {py:class}`~oumi.core.configs.params.generation_params.GenerationParams` parameters that you can provide.

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True
  adapter_model: "path/to/adapter"  # Optional: For adapter-based models

tasks:
  # LM Harness Tasks
  - evaluation_platform: lm_harness
    task_name: mmlu
    num_samples: 100
    eval_kwargs:
      num_fewshot: 5
  - evaluation_platform: lm_harness
    task_name: arc_challenge
    eval_kwargs:
      num_fewshot: 25
  - evaluation_platform: lm_harness
    task_name: hellaswag
    eval_kwargs:
      num_fewshot: 10

  # AlpacaEval Task
  - evaluation_platform: alpaca_eval
    version: 2.0  # or 1.0
    num_samples: 805

generation:
  batch_size: 16
  max_new_tokens: 512
  temperature: 0.0

output_dir: "my_evaluation_results"
enable_wandb: true
run_name: "phi3-evaluation"
```

#### Configuration Options

- `model`: Model-specific configuration ({py:class}`~oumi.core.configs.params.model_params.ModelParams`)
  - `model_name`: HuggingFace model identifier or local path
  - `trust_remote_code`: Whether to trust remote code (for custom models)
  - `adapter_model`: Path to adapter weights (optional)
  - `adapter_type`: Type of adapter ("lora" or "qlora")

- `tasks`: List of evaluation tasks ({py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`)
  - LM Harness Task Parameters:   ({py:class}`~oumi.core.configs.params.evaluation_params.LMHarnessTaskParams`)
    - `evaluation_platform`: "lm_harness"
    - `task_name`: Name of the LM Harness task
    - `num_fewshot`: Number of few-shot examples (0 for zero-shot)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

  - AlpacaEval Task Parameters: ({py:class}`~oumi.core.configs.params.evaluation_params.AlpacaEvalTaskParams`)
    - `evaluation_platform`: "alpaca_eval"
    - `version`: AlpacaEval version (1.0 or 2.0)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

- `generation`: Generation parameters ({py:class}`~oumi.core.configs.params.generation_params.GenerationParams`)
  - `batch_size`: Batch size for inference ("auto" for automatic selection)
  - `max_new_tokens`: Maximum number of tokens to generate
  - `temperature`: Sampling temperature

- `enable_wandb`: Enable Weights & Biases logging
- `output_dir`: Directory for saving results
- `run_name`: Name of the evaluation run


## Results and Logging

### Evaluation Results

Results are saved under the specified `output_dir`, in a folder named `<platform>_<timestamp>`, which includes the following files:

- `platform_results.json`: Detailed evaluation metrics
- `platform_task_config.json`: Task configuration parameters of the underlying platform
- `task_params.json`: Task parameters for Oumi (see {py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`)
- `model_params.json`: Model parameters (see {py:class}`~oumi.core.configs.params.model_params.ModelParams`)
- `generation_params.json`: Generation parameters (see {py:class}`~oumi.core.configs.params.generation_params.GenerationParams`)
- `inference_config.json`: Inference configuration; only applicable to generative benchmarks (see {py:class}`~oumi.core.configs.inference_config.InferenceConfig`)
- `package_versions.json`: Package version information for reproducibility

### Weights & Biases Integration

When `enable_wandb` is true, results are automatically logged to W&B:

```python
# Environment variable for W&B project name
os.environ["WANDB_PROJECT"] = "my-evaluation-project"
```
