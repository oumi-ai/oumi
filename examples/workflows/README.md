# Workflow Examples

This directory contains example workflow configurations for the Oumi workflow manager.

## What is a Workflow?

A workflow is a collection of oumi verbs (train, eval, infer, etc.) organized with dependencies and resource allocation. Workflows allow you to:

- Run multiple jobs in parallel on different GPUs
- Define dependencies between jobs (e.g., train → eval)
- Orchestrate complex ML pipelines
- Monitor all jobs in real-time with a TUI

## Available Examples

### 1. Simple Train & Evaluate (`simple_train_eval.yaml`)

The most basic workflow: train a model, then evaluate it.

```bash
oumi workflow run examples/workflows/simple_train_eval.yaml
```

### 2. Parallel Evaluation (`parallel_eval.yaml`)

Train once, then evaluate on multiple benchmarks in parallel across 4 GPUs.

```bash
oumi workflow run examples/workflows/parallel_eval.yaml
```

### 3. Hyperparameter Sweep (`hyperparameter_sweep.yaml`)

Run 8 training jobs in parallel with different hyperparameters (learning rates, batch sizes).

```bash
oumi workflow run examples/workflows/hyperparameter_sweep.yaml
```

### 4. Multi-Model Pipeline (`multi_model_pipeline.yaml`)

Train and evaluate multiple models (Llama and Mistral) in parallel.

```bash
oumi workflow run examples/workflows/multi_model_pipeline.yaml
```

### 5. Full ML Pipeline (`full_pipeline.yaml`)

Complete pipeline: data synthesis → training → evaluation → quantization → inference.

```bash
oumi workflow run examples/workflows/full_pipeline.yaml
```

## Workflow Configuration

### Basic Structure

```yaml
name: "my-workflow"
description: "What this workflow does"

resources:
  gpus: [0, 1, 2, 3]  # GPUs to use
  max_parallel: 4      # Max parallel jobs
  allocation: "dynamic"  # Resource allocation strategy

jobs:
  - name: "job-1"
    verb: "train"  # oumi verb to run
    config: "path/to/config.yaml"
    resources:
      gpu: 0  # Specific GPU or "auto"

  - name: "job-2"
    verb: "evaluate"
    config: "path/to/eval.yaml"
    depends_on: ["job-1"]  # Wait for job-1 to complete
    resources:
      gpu: auto  # Auto-assign to available GPU
```

### Resource Allocation

- **Fixed allocation**: Assign specific GPUs to jobs
  ```yaml
  resources: {gpu: 0}  # Use GPU 0
  ```

- **Auto allocation**: Let the workflow manager assign GPUs
  ```yaml
  resources: {gpu: auto}  # Use any available GPU
  ```

- **Dynamic queue**: More jobs than GPUs? They'll queue and start when resources free up
  ```yaml
  resources:
    gpus: [0, 1]
    max_parallel: 2  # Only 2 jobs at once
  # If you have 10 jobs, they'll queue and run 2 at a time
  ```

### Dependencies

Jobs can depend on other jobs completing successfully:

```yaml
jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"

  - name: "eval-1"
    verb: "evaluate"
    depends_on: ["train"]  # Waits for "train" to complete

  - name: "eval-2"
    verb: "evaluate"
    depends_on: ["train"]  # Also waits for "train"

  - name: "quantize"
    verb: "quantize"
    depends_on: ["eval-1", "eval-2"]  # Waits for BOTH evals
```

### Overriding Config Values

You can override config values from the command line:

```yaml
jobs:
  - name: "train-lr-high"
    verb: "train"
    config: "base.yaml"
    args:
      - "training.learning_rate=1e-3"
      - "training.run_name=experiment-1"
```

### Environment Variables

Set environment variables for all jobs or specific jobs:

```yaml
# Global env vars
env:
  WANDB_PROJECT: "my-project"
  HF_TOKEN: "${HF_TOKEN}"  # Use value from shell

jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    env:
      WANDB_RUN_NAME: "run-1"  # Job-specific env var
```

## CLI Commands

### Run a workflow
```bash
oumi workflow run workflow.yaml
```

### Run without TUI (headless)
```bash
oumi workflow run workflow.yaml --no-tui
```

### Specify GPUs
```bash
oumi workflow run workflow.yaml --gpus 0,1,2,3
```

### Limit parallelism
```bash
oumi workflow run workflow.yaml --max-parallel 2
```

### Validate before running
```bash
oumi workflow validate workflow.yaml
```

## TUI Features

When you run a workflow with the TUI enabled (default), you get:

- **Real-time job status**: See all jobs and their current state
- **Live progress bars**: Track training steps, eval progress
- **Metrics display**: Loss, learning rate, GPU usage, etc.
- **Job logs**: View logs for any job
- **Interactive controls**:
  - `j/k`: Scroll through jobs
  - `c`: Cancel selected job
  - `r`: Refresh display
  - `q`: Quit (workflows continue in background)

## Tips

1. **Start small**: Test with `simple_train_eval.yaml` first
2. **Validate first**: Always run `oumi workflow validate` before executing
3. **Use auto allocation**: Let the workflow manager handle GPU assignment for you
4. **Monitor in TUI**: The TUI makes it easy to track multiple jobs at once
5. **Parallel evaluation**: After training, run multiple evals in parallel to save time
6. **Hyperparameter sweeps**: Use workflows to run many experiments in parallel

## Creating Your Own Workflows

1. Start with an example that's close to what you need
2. Modify the jobs, configs, and dependencies
3. Validate: `oumi workflow validate my-workflow.yaml`
4. Run: `oumi workflow run my-workflow.yaml`

## Advanced Features

### Remote Execution (Coming Soon)

```yaml
resources:
  mode: "mixed"
  local_gpus: [0, 1]
  remote:
    - name: "aws-cluster"
      max_jobs: 4
```

### Retries

```yaml
jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    max_retries: 3
    retry_delay: 60  # seconds
```

### Timeouts

```yaml
# Global workflow timeout
timeout: 86400  # 24 hours

jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    timeout: 7200  # 2 hours for this job
```

## Need Help?

- Check the [workflow documentation](../../docs/workflow.md)
- Validate your workflow: `oumi workflow validate your-workflow.yaml`
- File an issue: https://github.com/oumi-ai/oumi/issues
