# Oumi Workflow Manager

> Orchestrate complex ML pipelines with parallel execution and real-time monitoring

## Quick Start

```bash
# 1. Create a workflow
cat > workflow.yaml << EOF
name: "my-workflow"
resources:
  gpus: [0, 1]
jobs:
  - name: "train"
    verb: "train"
    config: "configs/recipes/llama/sft.yaml"
    resources: {gpu: 0}
  - name: "eval"
    verb: "evaluate"
    config: "configs/eval/mmlu.yaml"
    depends_on: ["train"]
    resources: {gpu: 1}
EOF

# 2. Validate
oumi workflow validate workflow.yaml

# 3. Run with TUI
oumi workflow run workflow.yaml
```

## What is it?

The Workflow Manager orchestrates multiple oumi verbs (train, eval, infer, etc.) into complex pipelines:

- 🚀 **Run 10+ jobs in parallel** on different GPUs
- 📊 **Real-time TUI** with live progress and metrics
- 🔗 **Dependency management** - define job execution order
- ⚡ **Async-first** - efficient non-blocking execution
- 🎯 **Dynamic scheduling** - jobs queue and start automatically
- 📝 **YAML configuration** - declarative workflow definitions

## Key Concepts

### Jobs
A job is a single execution of an oumi verb:
```yaml
- name: "train-model"
  verb: "train"
  config: "train.yaml"
  resources: {gpu: 0}
```

### Dependencies
Jobs can depend on other jobs:
```yaml
- name: "eval"
  verb: "evaluate"
  config: "eval.yaml"
  depends_on: ["train"]  # Runs after "train" completes
```

### Resource Allocation
Assign jobs to GPUs automatically or explicitly:
```yaml
resources:
  gpus: [0, 1, 2, 3]    # Available GPUs
  allocation: "dynamic"  # Auto-assign

jobs:
  - name: "job1"
    resources: {gpu: auto}  # Use any available GPU
  - name: "job2"
    resources: {gpu: 2}     # Use specific GPU
```

## Architecture

```
┌─────────────────────────────────────────────┐
│           Workflow YAML Config              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          WorkflowConfig (validated)         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│     Workflow (job graph + orchestration)    │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐    ┌──────────────────┐
│ Resource     │◄───┤ WorkflowExecutor │
│ Manager      │    │ (async scheduler)│
└──────────────┘    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
            ┌──────────┐      ┌──────────┐
            │ Job 1    │      │ Job 2    │
            │ (train)  │      │ (eval)   │
            └──────────┘      └──────────┘
                    │                 │
                    └────────┬────────┘
                             ▼
                    ┌──────────────────┐
                    │   TUI Dashboard  │
                    │  (real-time UI)  │
                    └──────────────────┘
```

## Components

### Core (`src/oumi/workflow/`)
- **`config.py`** - YAML configuration and validation
- **`workflow.py`** - Workflow orchestration
- **`executor.py`** - Async job scheduler
- **`job.py`** - Job execution wrapper
- **`resource_manager.py`** - GPU allocation

### TUI (`src/oumi/workflow/tui/`)
- **`app.py`** - Textual TUI application
- **`widgets.py`** - Custom UI widgets

### CLI (`src/oumi/cli/`)
- **`workflow.py`** - CLI commands

## Examples

See `examples/workflows/` for complete examples:

1. **simple_train_eval.yaml** - Basic train → eval
2. **parallel_eval.yaml** - Parallel evaluation
3. **hyperparameter_sweep.yaml** - 8-way parameter sweep
4. **multi_model_pipeline.yaml** - Multiple models in parallel
5. **full_pipeline.yaml** - Complete ML pipeline

## Documentation

- **Full docs**: `docs/workflow.md`
- **Examples guide**: `examples/workflows/README.md`
- **Implementation details**: `WORKFLOW_IMPLEMENTATION.md`

## Testing

```bash
# Run all workflow tests
pytest tests/unit/workflow/

# Specific test
pytest tests/unit/workflow/test_config.py

# With coverage
pytest tests/unit/workflow/ --cov=oumi.workflow
```

## Requirements

- Python 3.9+
- `textual>=1.0` (for TUI)
- All standard oumi dependencies

## CLI Usage

```bash
# Run workflow with TUI
oumi workflow run workflow.yaml

# Run without TUI (headless)
oumi workflow run workflow.yaml --no-tui

# Specify GPUs
oumi workflow run workflow.yaml --gpus 0,1,2,3

# Limit parallelism
oumi workflow run workflow.yaml --max-parallel 2

# Validate before running
oumi workflow validate workflow.yaml
```

## Features

### ✅ Implemented
- Async job execution
- Multi-GPU parallelism
- Dynamic resource allocation
- Dependency resolution
- Real-time TUI
- Config validation
- Progress tracking
- Retry logic
- Timeouts
- YAML configuration
- Config overrides
- Environment variables

### 🔜 Coming Soon
- Remote execution (via oumi launch)
- Workflow persistence
- Advanced scheduling
- Web dashboard

## Contributing

The workflow manager is extensible. Areas to contribute:

- Better progress parsing
- Additional TUI features
- Remote execution backends
- Workflow visualization
- Performance optimizations

## License

Apache 2.0 - See LICENSE file
