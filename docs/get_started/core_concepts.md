# Core Concepts

## Overview

Oumi combines enterprise-grade reliability with research-friendly flexibility, supporting the complete foundation model lifecycle from pretraining to deployment.

This guide introduces the core concepts, and terminology used throughout Oumi, as well as the architecture and guiding design principles. Understanding these core terms will help you navigate Oumi's documentation and features effectively.

The following diagram illustrates the typical workflow in Oumi. You can either start from scratch (pre-training), or continue from a SOTA model (post-training or continued pre-training).

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#f5f5f5'}}}%%
graph LR
    %% Data stage connections
    DS[Datasets] --> |Existing Datasets| TR[Training]
    DS --> |Data Synthesis| TR

    %% Training methods
    TR --> |Pre-training| EV[Evaluation]
    TR --> |SFT| EV
    TR --> |DPO| EV

    %% Evaluation methods spread horizontally
    EV --> |Generative| INF[Inference]
    EV --> |Multi-choice| INF
    EV --> |LLM Judge| INF

    %% Style for core workflow
    style DS fill:#1565c0,color:#ffffff
    style TR fill:#1565c0,color:#ffffff
    style EV fill:#1565c0,color:#ffffff
    style INF fill:#1565c0,color:#ffffff
```

## Core Concepts

### Oumi CLI

The CLI is the entry point for all Oumi commands.

```bash
oumi <command> [options]
```

For detailed help on any command, you can use the `--help` option:

```bash
oumi --help            # for general help
oumi <command> --help  # for command-specific help
```

The available commands are:

| Command      | Purpose                                                               |
|--------------|-----------------------------------------------------------------------|
`train`        | Train a model.
`evaluate`     | Evaluate a model.
`infer`        | Run inference on a model.
`launch`       | Launch jobs remotely.
`judge`        | Judge datasets, models or conversations.
`env`          | Prints information about the current environment.
`distributed`  | A wrapper for torchrun/accelerate with reasonable default values for distributed training.

Any Oumi command which takes a config path as an argument (`train`, `evaluate`, `infer`, etc.) can override parameters from the command line. For example:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.max_steps 20 \
  --training.learning_rate 1e-4 \
  --data.train.datasets[0].shuffle true \
  --training.output_dir output/smollm-135m-sft
```

See {doc}`/cli/commands` for full CLI details, including more details about CLI overrides.

### Python API

The Python API allows you to use Oumi to `train`, `evaluate`, `infer`, `judge`, and more. You can use it in a notebook, a script, or any custom workflow.

For example, to train a model, you can use the `train` function:

```python
from oumi.train import train
from oumi.core.configs import TrainingConfig

config = TrainingConfig.from_yaml("path/to/config.yaml")
train(config)
```

See {doc}`/api/oumi` for full API details.

### Configs

To provide recordability and reproducibility for common workflows, Oumi uses exhaustive configs to define all the parameters for each step.

| Config Type | Purpose | Documentation |
|------------|---------|---------------|
| Training | Model training workflows | {doc}`/user_guides/train/configuration` |
| Evaluation | Benchmark configurations | {doc}`/user_guides/evaluate/evaluate` |
| Inference | Inference settings | {doc}`/user_guides/infer/infer` |
| Launcher | Deployment settings | {doc}`/user_guides/launch/launch` |

Example config structure:

```yaml
# Example training recipe
model:
  name: meta-llama/Llama-3.1-70B-Instruct
  trust_remote_code: true

data:
  train:
    datasets:
      - dataset_name: text_sft
        dataset_path: path/to/data
    stream: true

training:
  trainer_type: TRL_SFT
  learning_rate: 1e-4
```

For a full list of recipes, you can explore the {doc}`recipes page </resources/recipes>`.

### Other Key Concepts

| Term | Description | Documentation |
|------|-------------|---------------|
| Recipe | Predefined configurations in Oumi for common model training, evaluation and inference workflows | {doc}`/resources/recipes` |
| Launcher | Oumi's job orchestration system for running workloads across different cloud providers | {doc}`/user_guides/launch/launch` |
| Models | Model architectures and implementations. Oumi supports most models from HuggingFace's `transformers` library, as well as custom models. | {doc}`/resources/models/custom_models` |
| Datasets | Data loading and preprocessing pipelines | {doc}`/resources/datasets/datasets` |
| Trainers | Orchestrate training process and optimization. Oumi supports custom trainers, as well as trainers from HuggingFace's `transformers`, `TRL`, and many others in the future. | {doc}`/user_guides/train/training_methods` |
| Data Mixtures | Oumi's system for combining and weighting multiple datasets during training | {doc}`/resources/datasets/datasets` |
| Oumi Judge | Built-in system for evaluating model outputs based on customizable attributes (e.g. helpfulness, honesty, and safety) | {doc}`/user_guides/judge/judge` |

## Navigating the Repository

To contribute to Oumi or troubleshoot issues, it's helpful to understand how the repository is structured. Here's a breakdown of the key directories:

### Core Components

- `src/oumi/`: Main package directory
  - `core/`: Core functionality and base classes
  - `models/`: Model architectures and implementations
  - `datasets/`: Dataset loading and processing
  - `inference/`: Inference engines and serving
  - `evaluation/`: Evaluation pipelines and metrics
  - `judges/`: Implementation of Oumi Judge system
  - `launcher/`: Job orchestration and resource management
  - `cli/`: Command-line interface tools
  - `utils/`: Common utilities and helper functions

### Configuration and Examples

- `configs/`: YAML configuration files
  - `recipes/`: Predefined workflows for common tasks
- `notebooks/`: Example notebooks and tutorials
- `tests/`: Test suite (mirrors src/ structure)
- `docs/`: Documentation and guides

### Development Tools

- `pyproject.toml`: Project dependencies and build settings
- `Makefile`: Common development commands
- `scripts/`: Utility scripts for development
- `.github/`: CI/CD workflows and GitHub configurations

## Next Steps

1. **Get started with Oumi:** First {doc}`install Oumi </get_started/installation>`, then follow the {doc}`/get_started/quickstart` guide to run your first training job.
2. **Explore example recipes:**  Check out the {doc}`/resources/recipes` page and try running a few examples.
3. **Dive deeper with tutorials:** The {doc}`/get_started/tutorials` provide step-by-step guidance on specific tasks and workflows.
4. **Learn more about key functionalities:** Explore detailed guides on {doc}`training </user_guides/train/training_methods>`, {doc}`inference </user_guides/infer/infer>`, {doc}`evaluation </user_guides/evaluate/evaluate>`, and {doc}`model judging </user_guides/judge/judge>`.
