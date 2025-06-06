# Evaluation Configuration

Oumi allows users to define their evaluation configurations through a `YAML` file, providing a flexible, human-readable, and easily customizable format for setting up experiments. By using `YAML`, users can effortlessly configure model and generation parameters, and define a list of tasks to evaluate with. This approach not only streamlines the process of configuring evaluations but also ensures that configurations are easily versioned, shared, and reproduced across different environments and teams.

# Configuration Structure

The configuration `YAML` file is loaded into {py:class}`~oumi.core.configs.EvaluationConfig` class, and consists of {py:class}`~oumi.core.configs.params.model_params.ModelParams`, {py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`, and {py:class}`~oumi.core.configs.params.generation_params.GenerationParams`. If the evaluation benchmark is generative, meaning that the model responses need to be first generated (inferred) and then evaluated by a judge, you can also set the `inference_engine` ({py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`) for local inference or the `inference_remote_params` ({py:obj}`~oumi.core.configs.params.remote_params.RemoteParams`) for remote inference.

Here's an advanced configuration example, showing many of the available parameters:

```yaml
model:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  trust_remote_code: True
  adapter_model: "path/to/adapter"  # Optional: For adapter-based models

tasks:
  # LM Harness Tasks
  - evaluation_backend: lm_harness
    task_name: mmlu
    num_samples: 100
    eval_kwargs:
      num_fewshot: 5
  - evaluation_backend: lm_harness
    task_name: arc_challenge
    eval_kwargs:
      num_fewshot: 25
  - evaluation_backend: lm_harness
    task_name: hellaswag
    eval_kwargs:
      num_fewshot: 10

  # AlpacaEval Task
  - evaluation_backend: alpaca_eval
    version: 2.0  # or 1.0
    num_samples: 805

  # Custom Task
  - evaluation_backend: custom
    task_name: my_custom_evaluation

generation:
  batch_size: 16
  max_new_tokens: 512
  temperature: 0.0

inference_engine: NATIVE

output_dir: "my_evaluation_results"
enable_wandb: true
run_name: "phi3-evaluation"
```

# Configuration Options

- `model`: Model-specific configuration ({py:class}`~oumi.core.configs.params.model_params.ModelParams`)
  - `model_name`: HuggingFace model identifier or local path
  - `trust_remote_code`: Whether to trust remote code (for custom models)
  - `adapter_model`: Path to adapter weights (optional)
  - `adapter_type`: Type of adapter ("lora" or "qlora")
  - `shard_for_eval`: Enable multi-GPU parallelization on a single node

- `tasks`: List of evaluation tasks ({py:class}`~oumi.core.configs.params.evaluation_params.EvaluationTaskParams`)
  - LM Harness Task Parameters:   ({py:class}`~oumi.core.configs.params.evaluation_params.LMHarnessTaskParams`)
    - `evaluation_backend`: "lm_harness"
    - `task_name`: Name of the LM Harness task
    - `num_fewshot`: Number of few-shot examples (0 for zero-shot)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

  - AlpacaEval Task Parameters: ({py:class}`~oumi.core.configs.params.evaluation_params.AlpacaEvalTaskParams`)
    - `evaluation_backend`: "alpaca_eval"
    - `version`: AlpacaEval version (1.0 or 2.0)
    - `num_samples`: Number of samples to evaluate
    - `eval_kwargs`: Additional task-specific parameters

  - Custom Task Parameters:
    - `evaluation_backend`: "custom"
    - `task_name`: Name that the custom evaluation function was registered with

- `generation`: Generation parameters ({py:class}`~oumi.core.configs.params.generation_params.GenerationParams`)
  - `batch_size`: Batch size for inference ("auto" for automatic selection)
  - `max_new_tokens`: Maximum number of tokens to generate
  - `temperature`: Sampling temperature

- `inference_engine`: Inference engine for local inference ({py:obj}`~oumi.core.configs.inference_engine_type.InferenceEngineType`)
- `inference_remote_params`: Inference parameters for remote inference ({py:obj}`~oumi.core.configs.params.remote_params.RemoteParams`)

- `enable_wandb`: Enable Weights & Biases logging
- `output_dir`: Directory for saving results
- `run_name`: Name of the evaluation run
