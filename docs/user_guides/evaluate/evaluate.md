# Evaluation

```{toctree}
:maxdepth: 2
:caption: Evaluation
:hidden:

evaluation_config
standardized_benchmarks
generative_benchmarks
leaderboards
custom_evals
```

## Overview

Oumi offers a flexible and unified framework designed to assess and benchmark **Large Language Models (LLMs)** and **Vision Language Models (VLMs)**. The framework allows researchers, developers, and organizations to easily evaluate the performance of their models across a variety of benchmarks, compare results, and track progress in a standardized and reproducible way.

Key features include:

- **Seamless Setup**: Single-step installation for all packages and dependencies, ensuring quick and conflict-free setup.
- **Consistency**: Platform ensures deterministic execution and [reproducible results](/user_guides/evaluate/evaluate.md#results-and-logging). Reproducibility is achieved by automatically logging and versioning all environmental parameters and experimental configurations.
- **Diversity**: Offering a [wide range of benchmarks](/user_guides/evaluate/evaluate.md#benchmark-types) across domains. Oumi enables a comprehensive evaluation of LLMs on tasks ranging from natural language understanding to creative text generation, providing holistic assessment across various real-world applications.
- **Scalability**: Supports [multi-GPU and multi-node evaluations](/user_guides/infer/infer.md#distributed-inference), along with the ability to shard large models across multiple GPUs/nodes. Incorporates batch processing optimizations to effectively manage memory constraints and ensure efficient resource utilization.
- **Multimodality**: Designed with multiple modalities in mind, Oumi already supports evaluating on {ref}`joint image-text <multi-modal-standardized-benchmarks>` inputs, assessing VLMs on cross-modal reasoning tasks, where visual and linguistic data are inherently linked.
<!-- Consider adding later:
**Extensibility**: Designed with simplicity and modularity in mind, Oumi offers a flexible framework that empowers the community to easily contribute new benchmarks and metrics. This facilitates continuous improvement and ensures the platform evolves alongside emerging research and industry trends.
-->

Oumi seamlessly integrates with leading evaluation frameworks such as [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), and (WIP) [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
For more specialized use cases not covered by these frameworks, Oumi also supports {doc}`custom evaluation functions </user_guides/evaluate/custom_evals>`, enabling you to tailor evaluations to your specific needs.

## Benchmark Types

| Type | Description | When to Use | Get Started |
|------|-------------|-------------|-------------|
| **Standardized Benchmarks** | Assess model knowledge and reasoning capability through structured questions with predefined answers | Ideal for measuring factual knowledge, reasoning capabilities, and performance on established text-based and multi-modal benchmarks | See {doc}`Standardized benchmarks page </user_guides/evaluate/standardized_benchmarks>` |
| **Open-Ended Generation** | Evaluate model's ability to effectively respond to open-ended questions | Best for assessing instruction-following capabilities and response quality | See {doc}`Generative benchmarks page </user_guides/evaluate/generative_benchmarks>` |
| **LLM as Judge** | Automated assessment using LLMs | Suitable for automated evaluation of response quality against predefined (helpfulness, honesty, safety) or custom criteria | See {doc}`Judge documentation </user_guides/judge/judge>` |
| **Custom Evaluations** | Fully custom evaluation functions | The most flexible option, allowing you to build any complex evaluation scenario | See {doc}`Custom evaluations documentation </user_guides/evaluate/custom_evals>` |

## Quick Start

### Using the CLI

The simplest way to evaluate a model is by authoring a `YAML` configuration, and calling the Oumi CLI:

````{dropdown} configs/recipes/phi3/evaluation/eval.yaml
```{literalinclude} /../configs/recipes/phi3/evaluation/eval.yaml
:language: yaml
```
````

```bash
oumi evaluate -c configs/recipes/phi3/evaluation/eval.yaml
```

To run evaluation with multiple GPUs, see {ref}`Multi-GPU Evaluation <multi-gpu-evaluation>`.

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

### Configuration File

A minimal evaluation configuration file looks as follows. The `model_name` can be a HuggingFace model name or a local path to a model. For more details on configuration settings, please visit the {doc}`evaluation configuration </user_guides/evaluate/evaluation_config>` page.

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True

tasks:
  - evaluation_backend: lm_harness
    task_name: mmlu

output_dir: "my_evaluation_results"
```

(multi-gpu-evaluation)=

### Multi-GPU Evaluation

Oumi supports multiple strategies for multi-GPU evaluation. **Choose the right strategy based on your model size and inference engine.**

##### Strategy 1: VLLM with Tensor Parallelism (Recommended)

**Best for:** Any model size with VLLM support
**Speed:** Fastest option for multi-GPU evaluation
**Use case:** Models that benefit from GPU acceleration

```{code-block} yaml
:emphasize-lines: 3-5,11
model:
  model_name: "Qwen/Qwen3-32B"
  model_kwargs:
    tensor_parallel_size: 4  # Split model across 4 GPUs
    # tensor_parallel_size: -1  # Or use -1 to auto-detect all GPUs (default)

tasks:
  - evaluation_backend: lm_harness
    task_name: mmlu_pro

inference_engine: VLLM

output_dir: "eval_results"
```

**Run:**

```shell
oumi evaluate -c config.yaml
```

The model will be automatically split across all specified GPUs using tensor parallelism.

##### Strategy 2: NATIVE with Data Parallelism (via Accelerate)

**Best for:** Models that fit on a single GPU
**Speed:** K× speedup with K GPUs (e.g., 4× faster with 4 GPUs)
**Use case:** Want faster evaluation of smaller models

```{code-block} yaml
:emphasize-lines: 8
model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"

tasks:
  - evaluation_backend: lm_harness
    task_name: mmlu_pro

inference_engine: NATIVE

output_dir: "eval_results"
```

**Run:**

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 oumi distributed accelerate launch -m oumi evaluate -c config.yaml
```

Each GPU evaluates different data in parallel. Results are automatically aggregated.

##### Strategy 3: NATIVE with Model Parallelism

**Best for:** Large models that don't fit on a single GPU
**Speed:** Enables evaluation of models that otherwise can't run
**Use case:** 70B+ models without VLLM support

```{code-block} yaml
:emphasize-lines: 3,9
model:
  model_name: "meta-llama/Llama-3.3-70B-Instruct"
  shard_for_eval: True  # Split model across available GPUs

tasks:
  - evaluation_backend: lm_harness
    task_name: mmlu

inference_engine: NATIVE

output_dir: "eval_results"
```

**Run:**

```shell
oumi evaluate -c config.yaml  # Do NOT use accelerate launch
```

The model weights will be automatically distributed across all available GPUs.

##### Choosing the Right Strategy

| Inference Engine | Strategy | GPU Required? | Command |
|---------------------------|----------|---------------|---------|
| VLLM | - | Yes | `oumi evaluate` |
| VLLM | Tensor Parallelism | Yes (multi-gpu) | `oumi evaluate` |
| NATIVE | - | No | `oumi evaluate` |
| NATIVE | Data Parallelism | Yes | `accelerate launch -m oumi evaluate` |
| NATIVE | Tensor Parallelism (`shard_for_eval=True`) | No | `oumi evaluate` |
| NATIVE | Tensor + Data Parallelism (`shard_for_eval=True`) | Yes (multi-gpu) | `accelerate launch -m oumi evaluate` |

```{note}
Only single node, multiple GPU configurations are currently supported. Multi-node evaluation is not yet available.
```

## Results and Logging

The evaluation outputs are saved under the specified `output_dir`, in a folder named `<backend>_<timestamp>`. This folder includes the evaluation results and all metadata required to reproduce the results.

### Evaluation Results

| File | Description |
|------|-------------|
| `task_result.json` | A dictionary that contains all evaluation metrics relevant to the benchmark, together with the execution duration, and date/time of execution.

**Schema**

```yaml
{
  "results": {
    <benchmark_name>: {
      <metric_1>: <metric_1_value>,
      <metric_2>: <metric_2_value>,
      etc.
    },
  },
  "duration_sec": <execution_duration>,
  "start_time": <start_date_and_time>,
}
```

### Reproducibility Metadata

To ensure that evaluations are fully reproducible, Oumi automatically logs all input configurations and environmental parameters, as shown below. These files provide a complete and traceable record of each evaluation, enabling users to reliably replicate results, ensuring consistency and transparency throughout the evaluation lifecycle.

| File | Description | Reference |
|------|-------------|-----------|
| `task_params.json` | Evaluation task parameters | {py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams` |
| `model_params.json` | Model parameters | {py:class}`~oumi.core.configs.params.model_params.ModelParams` |
| `generation_params.json` | Generation parameters | {py:class}`~oumi.core.configs.params.generation_params.GenerationParams` |
| `inference_config.json` | Inference configuration (for generative benchmarks) | {py:class}`~oumi.core.configs.inference_config.InferenceConfig` |
| `package_versions.json` | Package version information | N/A. Flat dictionary of all installed packages and their versions |

### Weights & Biases

To enhance experiment tracking and result visualization, Oumi integrates with [Weights and Biases](https://wandb.ai/site) (Wandb), a leading tool for managing machine learning workflows. Wandb enables users to monitor and log metrics, hyperparameters, and model outputs in real time, providing detailed insights into model performance throughout the evaluation process. When `enable_wandb` is set, Wandb results are automatically logged, empowering users to track experiments with greater transparency, and easily visualize trends across multiple runs. This integration streamlines the process of comparing models, identifying optimal configurations, and maintaining an organized, collaborative record of all evaluation activities.

To ensure Wandb results are logged:

- Enable Wandb in the {doc}`configuration file </user_guides/evaluate/evaluation_config>`

```yaml
enable_wandb: true
```

- Ensure the environmental variable `WANDB_PROJECT` points to your project name

```python
os.environ["WANDB_PROJECT"] = "my-evaluation-project"
```
