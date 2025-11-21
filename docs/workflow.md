# Workflow Manager

The Oumi Workflow Manager is a powerful system for orchestrating and managing complex ML pipelines. It allows you to define, execute, and monitor workflows that chain together multiple oumi verbs (train, evaluate, infer, etc.) with dependency management, parallel execution, and real-time progress tracking.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Workflow Configuration](#workflow-configuration)
- [Resource Management](#resource-management)
- [TUI (Terminal User Interface)](#tui-terminal-user-interface)
- [CLI Commands](#cli-commands)
- [Advanced Topics](#advanced-topics)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

A workflow is a collection of jobs organized into a directed acyclic graph (DAG) where:
- Each **job** is an execution of an oumi verb (train, eval, infer, etc.)
- Jobs can have **dependencies** on other jobs
- Jobs are assigned to **resources** (GPUs, remote machines)
- The workflow manager **schedules** and **executes** jobs efficiently
- A **TUI** provides real-time monitoring and control

### Why Use Workflows?

- **Automation**: Define complex pipelines once, run them repeatedly
- **Parallel execution**: Run multiple jobs simultaneously on different GPUs
- **Dependency management**: Automatically run jobs in the correct order
- **Resource efficiency**: Maximize GPU utilization with dynamic scheduling
- **Monitoring**: Track all jobs in real-time with a beautiful TUI
- **Reproducibility**: Configuration-as-code for your entire ML pipeline

## Key Features

### ✨ Core Capabilities

- **Multi-GPU parallelism**: Run 10+ jobs in parallel across GPUs
- **Dynamic resource allocation**: Jobs automatically queue and start when resources are available
- **Dependency management**: Define job dependencies for complex pipelines
- **Real-time TUI**: Monitor all jobs simultaneously with live progress updates
- **Async-first architecture**: Efficient non-blocking execution
- **Config overrides**: Override config values via CLI args
- **Retry logic**: Automatically retry failed jobs
- **Timeouts**: Set timeouts for jobs and entire workflows
- **Remote execution**: (Coming soon) Execute jobs on remote clusters

### 🎯 Supported Verbs

All oumi verbs are supported in workflows:
- `train` - Train models
- `evaluate` / `eval` - Evaluate models
- `infer` - Run inference
- `tune` - Hyperparameter tuning
- `quantize` - Model quantization
- `synth` / `synthesize` - Data synthesis
- `judge` - Judge datasets/conversations

## Quick Start

### 1. Create a workflow YAML

```yaml
# my_workflow.yaml
name: "my-first-workflow"

resources:
  gpus: [0, 1]  # Use GPU 0 and 1

jobs:
  - name: "train-model"
    verb: "train"
    config: "configs/recipes/llama/sft.yaml"
    resources:
      gpu: 0

  - name: "evaluate-model"
    verb: "evaluate"
    config: "configs/eval/mmlu.yaml"
    depends_on: ["train-model"]
    resources:
      gpu: 1
```

### 2. Validate the workflow

```bash
oumi workflow validate my_workflow.yaml
```

### 3. Run the workflow

```bash
oumi workflow run my_workflow.yaml
```

The TUI will launch, showing real-time progress of all jobs!

## Workflow Configuration

### Basic Structure

```yaml
name: "workflow-name"                # Required: Workflow identifier
description: "What this does"        # Optional: Human-readable description

resources:                           # Resource configuration
  gpus: [0, 1, 2, 3]                # GPUs to use
  max_parallel: 4                    # Max parallel jobs
  allocation: "dynamic"              # Allocation strategy

env:                                 # Global environment variables
  WANDB_PROJECT: "my-project"

output_dir: "./outputs/workflow"    # Base output directory

tui: true                            # Enable/disable TUI

timeout: 86400                       # Workflow timeout (seconds)

jobs:                                # List of jobs
  - name: "job-1"
    verb: "train"
    config: "path/to/config.yaml"
    # ... job configuration
```

### Job Configuration

Each job has the following fields:

```yaml
- name: "unique-job-name"            # Required: Job identifier
  verb: "train"                      # Required: Oumi verb to execute
  config: "path/to/config.yaml"     # Required: Config file path

  depends_on: ["job1", "job2"]      # Optional: Dependencies

  resources:                         # Optional: Resource requirements
    gpu: 0                           # Specific GPU (or "auto")
    # OR
    remote: "cluster-name"           # Remote execution

  args:                              # Optional: CLI args to override config
    - "training.learning_rate=1e-4"
    - "training.run_name=experiment-1"

  env:                               # Optional: Job-specific env vars
    CUDA_VISIBLE_DEVICES: "0,1"

  workdir: "./work"                  # Optional: Working directory

  max_retries: 3                     # Optional: Retry failed jobs
  retry_delay: 60                    # Optional: Seconds between retries

  timeout: 7200                      # Optional: Job timeout (seconds)
```

### Dependencies

Jobs can depend on other jobs. The workflow manager ensures jobs run only after their dependencies complete successfully.

```yaml
jobs:
  - name: "preprocess"
    verb: "synth"
    config: "prep.yaml"

  - name: "train"
    verb: "train"
    config: "train.yaml"
    depends_on: ["preprocess"]       # Waits for preprocess

  - name: "eval-mmlu"
    verb: "evaluate"
    config: "mmlu.yaml"
    depends_on: ["train"]             # Waits for train

  - name: "eval-gsm8k"
    verb: "evaluate"
    config: "gsm8k.yaml"
    depends_on: ["train"]             # Also waits for train

  - name: "quantize"
    verb: "quantize"
    config: "quant.yaml"
    depends_on: ["eval-mmlu", "eval-gsm8k"]  # Waits for BOTH evals
```

### Config Overrides

Override config values without modifying the config file:

```yaml
jobs:
  - name: "train-lr-high"
    verb: "train"
    config: "base.yaml"
    args:
      - "training.learning_rate=1e-3"           # Override learning rate
      - "training.num_epochs=10"                # Override epochs
      - "model.name_or_path=meta-llama/Llama-2-7b"  # Override model
```

## Resource Management

The workflow manager handles resource allocation intelligently.

### GPU Allocation Strategies

#### 1. Fixed GPU Assignment

Assign specific GPUs to jobs:

```yaml
resources:
  gpus: [0, 1, 2, 3]

jobs:
  - name: "train-on-gpu0"
    verb: "train"
    config: "train.yaml"
    resources:
      gpu: 0  # Always use GPU 0
```

#### 2. Auto Assignment

Let the workflow manager assign GPUs:

```yaml
jobs:
  - name: "train-1"
    verb: "train"
    config: "train.yaml"
    resources:
      gpu: auto  # Use any available GPU
```

#### 3. Dynamic Queueing

More jobs than GPUs? They'll queue automatically:

```yaml
resources:
  gpus: [0, 1]          # Only 2 GPUs
  max_parallel: 2       # Max 2 jobs at once

jobs:
  # 10 jobs defined...
  # First 2 start immediately
  # Remaining 8 queue and start as resources free
```

### Allocation Strategies

```yaml
resources:
  allocation: "dynamic"      # Default: assign resources as available
  # OR
  allocation: "fixed"        # Jobs wait for specific resources
  # OR
  allocation: "load_balanced"  # Assign to least busy GPU
```

### Parallelism Control

```yaml
resources:
  max_parallel: 4     # Run at most 4 jobs simultaneously
  # Even if more GPUs are available
```

## TUI (Terminal User Interface)

The TUI provides real-time monitoring and control of workflows.

### Layout

```
┌─ Workflow: my-workflow ──────────────────────────────────────────┐
│ Status: Running (3/10 completed)                                  │
├────────────────────────────────────────────────────────────────────┤
│ ┌─ Job: train-model [GPU:0] ────────────── RUNNING ──────┐       │
│ │ Train: 67% █████████████░░░░░░░ 670/1000 steps          │       │
│ │ Loss: 0.189 | LR: 0.0001 | Time: 1h 32m                 │       │
│ └──────────────────────────────────────────────────────────┘       │
│ ┌─ Job: eval-mmlu [GPU:1] ──────────────── COMPLETED ────┐       │
│ │ Accuracy: 0.752 | Duration: 15m                          │       │
│ └──────────────────────────────────────────────────────────┘       │
│                                                                    │
│ [Metrics Panel]                                                    │
│ [Log Viewer]                                                       │
└────────────────────────────────────────────────────────────────────┘
Status: Total: 10 | Running: 2 | Completed: 3 | Failed: 0
```

### Key Bindings

- `j` / `k` - Scroll through jobs
- `↑` / `↓` - Navigate
- `Enter` - View job details
- `c` - Cancel selected job
- `r` - Refresh display
- `q` - Quit (workflows continue in background)

### Features

- **Real-time updates**: Progress bars and metrics update live
- **Job cards**: Each job has a card showing status, progress, metrics
- **Metrics panel**: Aggregated workflow statistics
- **Log viewer**: View logs for any job
- **Color coding**: Visual indication of job status

## CLI Commands

### Run Workflow

```bash
# Basic usage
oumi workflow run workflow.yaml

# Specify GPUs
oumi workflow run workflow.yaml --gpus 0,1,2,3

# Limit parallelism
oumi workflow run workflow.yaml --max-parallel 2

# Disable TUI (headless mode)
oumi workflow run workflow.yaml --no-tui
```

### Validate Workflow

```bash
# Validate configuration
oumi workflow validate workflow.yaml

# Output:
# ✓ Workflow configuration is valid!
#   Name: my-workflow
#   Jobs: 5
#     - train: train (depends on: )
#     - eval-1: evaluate (depends on: train)
#     - eval-2: evaluate (depends on: train)
```

### Check Status

```bash
# Show running workflows (coming soon)
oumi workflow status
```

### Cancel Workflow

```bash
# Cancel a running workflow (coming soon)
oumi workflow cancel <workflow-id>
```

## Advanced Topics

### Environment Variables

#### Global Environment Variables

Apply to all jobs:

```yaml
env:
  WANDB_PROJECT: "my-project"
  WANDB_ENTITY: "my-team"
  HF_TOKEN: "${HF_TOKEN}"  # Use value from shell
```

#### Job-Specific Environment Variables

Override for specific jobs:

```yaml
jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    env:
      WANDB_RUN_NAME: "experiment-1"
      CUDA_VISIBLE_DEVICES: "0,1"
```

### Retry Logic

Automatically retry failed jobs:

```yaml
jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    max_retries: 3          # Retry up to 3 times
    retry_delay: 60         # Wait 60s between retries
```

### Timeouts

Set timeouts to prevent jobs from running forever:

```yaml
# Workflow-level timeout
timeout: 86400  # 24 hours

jobs:
  - name: "train"
    verb: "train"
    config: "train.yaml"
    timeout: 7200  # 2 hours for this specific job
```

### Output Directory

Organize workflow outputs:

```yaml
output_dir: "./outputs/my-workflow"

# Each job gets a subdirectory:
# ./outputs/my-workflow/train-model/
# ./outputs/my-workflow/eval-model/
```

### Remote Execution (Coming Soon)

Execute jobs on remote clusters:

```yaml
resources:
  mode: "mixed"
  local_gpus: [0, 1]
  remote:
    - name: "aws-cluster"
      max_jobs: 4

jobs:
  - name: "train-remote"
    verb: "train"
    config: "train.yaml"
    resources:
      remote: "aws-cluster"
```

## Examples

See the `examples/workflows/` directory for complete examples:

- `simple_train_eval.yaml` - Basic train → eval pipeline
- `parallel_eval.yaml` - Parallel evaluation on multiple benchmarks
- `hyperparameter_sweep.yaml` - 8 training jobs with different hyperparameters
- `multi_model_pipeline.yaml` - Train multiple models in parallel
- `full_pipeline.yaml` - Complete ML pipeline (synth → train → eval → quantize → infer)

## Troubleshooting

### Common Issues

#### "Job failed to start"

- Check that the config file exists and is valid
- Verify the verb name is correct
- Check GPU availability

#### "Circular dependency detected"

- Review your `depends_on` declarations
- Ensure there are no circular dependencies (A→B→A)

#### "Resource timeout"

- Not enough GPUs for your jobs?
- Increase `timeout` or reduce `max_parallel`

#### "TUI not available"

- Install textual: `pip install textual`
- Or run with `--no-tui` for headless mode

### Debug Tips

1. **Validate first**: Always run `oumi workflow validate` before executing
2. **Start simple**: Test with a small workflow before scaling up
3. **Check logs**: Each job has a log file in `<output_dir>/<job-name>/job.log`
4. **Use --no-tui**: Easier to debug without TUI
5. **Monitor resources**: Check GPU usage with `nvidia-smi`

### Getting Help

- Documentation: This file
- Examples: `examples/workflows/`
- Issues: https://github.com/oumi-ai/oumi/issues

## Architecture

### Components

```
WorkflowManager
├── WorkflowConfig      # YAML configuration
├── Workflow            # Workflow orchestration
├── WorkflowExecutor    # Async job execution
├── ResourceManager     # GPU/resource allocation
├── Job                 # Verb wrapper
└── TUI                 # Terminal interface
    ├── JobCard         # Job status widget
    ├── MetricsPanel    # Metrics display
    └── StatusBar       # Overall progress
```

### Execution Flow

1. Load workflow config from YAML
2. Validate configuration (dependencies, verbs, etc.)
3. Create Job instances for each job
4. Initialize ResourceManager with available GPUs
5. Start WorkflowExecutor
6. Launch TUI (if enabled)
7. Scheduler continuously:
   - Finds jobs ready to run (dependencies met)
   - Acquires resources for jobs
   - Starts jobs asynchronously
   - Monitors progress
   - Releases resources when complete
8. TUI updates in real-time
9. Workflow completes when all jobs finish

### Async Architecture

The workflow manager is built on asyncio:
- Jobs run as async tasks
- Non-blocking resource allocation
- Parallel job execution
- Real-time progress updates
- Responsive TUI

## Best Practices

1. **Start with validation**: `oumi workflow validate` catches errors early
2. **Use auto allocation**: `gpu: auto` is more flexible than fixed assignment
3. **Define clear dependencies**: Make job order explicit
4. **Set reasonable timeouts**: Prevent runaway jobs
5. **Use descriptive names**: Job names appear in TUI and logs
6. **Organize outputs**: Use `output_dir` to keep results organized
7. **Test locally first**: Validate workflows before remote execution
8. **Monitor with TUI**: Real-time monitoring helps catch issues early
9. **Leverage parallelism**: Run independent jobs in parallel
10. **Version control workflows**: Keep workflow YAML in git

## Future Enhancements

- Remote execution via `oumi launch`
- Workflow templates library
- Persistent state tracking
- Workflow resume/restart
- Web-based dashboard
- Workflow scheduling (cron-like)
- Advanced DAG visualization
- Workflow composition (nested workflows)
- Cost tracking and optimization

## Contributing

Contributions welcome! Areas to explore:
- Additional TUI features
- Remote execution backends
- Better progress parsing
- Workflow visualization
- Performance optimizations

---

**Version**: 1.0.0
**Last Updated**: 2025-01-20
