# Core Concepts

This guide walks you through the fundamental concepts in Oumi. By the end, you'll understand how Oumi's components fit together and be ready to train your first model.

## Prerequisites

Before diving in, you should have:

- **Oumi installed** — See the {doc}`installation guide </get_started/installation>` if you haven't set it up yet
- **Basic Python knowledge** — Familiarity with running scripts and using pip
- **ML fundamentals** — Understanding of what model training and fine-tuning mean at a high level

## Key Terminology

Oumi uses standard machine learning terminology. Here are the key terms you'll encounter:

| Term | What it means |
|------|---------------|
| **Pretraining** | Training a model from scratch on large amounts of text data |
| **Fine-tuning** | Adapting a pretrained model to a specific task or domain |
| **SFT** | Supervised Fine-Tuning — Teaching a model to follow instructions using input-output examples |
| **DPO** | Direct Preference Optimization — Aligning a model using pairs of preferred/rejected responses |
| **GRPO** | Group Relative Policy Optimization — A reinforcement learning method for model alignment |
| **LoRA** | Low-Rank Adaptation — A memory-efficient fine-tuning technique that trains small adapter layers |
| **Inference** | Using a trained model to generate predictions or text |

## The Oumi Workflow

The diagram below shows a typical workflow in Oumi. You can start from scratch with pretraining, or begin with an existing model and fine-tune it using SFT, DPO, or GRPO:

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#f5f5f5'}}}%%
graph LR
    %% Data stage connections
    DS[Datasets] --> |Existing Datasets| TR[Training]
    DS --> |Data Synthesis| TR

    %% Training methods
    TR --> |Pretraining| EV[Evaluation]
    TR --> |SFT| EV
    TR --> |DPO| EV
    TR --> |GRPO| EV

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

## Using Oumi

Oumi provides two ways to run workflows: the command-line interface (CLI) and the Python API. Most users start with the CLI for its simplicity, then move to the Python API when they need more control.

### Command-Line Interface (CLI)

The CLI is the quickest way to get started. All commands follow this pattern:

```bash
oumi <command> [options]
```

For detailed help on any command, you can use the `--help` option:

```bash
oumi --help            # for general help
oumi <command> --help  # for command-specific help
```

The available commands are:

| Command | Purpose |
|---------|---------|
| `train` | Train or fine-tune a model |
| `evaluate` | Evaluate a model on benchmarks |
| `infer` | Run inference on a model |
| `launch` | Launch jobs on cloud platforms |
| `judge` | Use LLM-as-a-Judge for evaluation |
| `synth` | Generate synthetic training data |
| `analyze` | Analyze and profile datasets |
| `tune` | Hyperparameter tuning with Optuna |
| `quantize` | Quantize models for efficient deployment |
| `distributed` | Distributed training wrapper for torchrun/accelerate |
| `env` | Display environment information |

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

The Python API gives you programmatic control over Oumi. Use it when you need to:
- Integrate training into a larger pipeline
- Modify configurations dynamically
- Run experiments in Jupyter notebooks

Here's a complete example that loads a recipe, modifies it, and runs training:

```python
from oumi.train import train
from oumi.core.configs import TrainingConfig

# Load a predefined recipe
config = TrainingConfig.from_yaml(
    "configs/recipes/smollm/sft/135m/quickstart_train.yaml"
)

# Optionally modify settings programmatically
config.training.max_steps = 100
config.training.output_dir = "output/my_experiment"

# Run training
train(config)
```

When you run this, you'll see output like:

```text
Loading model: HuggingFaceTB/SmolLM2-135M-Instruct
Starting training for 100 steps...
Step 10/100 | Loss: 2.45 | LR: 5.0e-05
Step 20/100 | Loss: 2.12 | LR: 5.0e-05
...
Training complete. Model saved to output/my_experiment
```

See {doc}`/api/oumi` for the full API reference.

### Configuration Files

Every Oumi workflow is defined by a YAML configuration file. This makes experiments reproducible—you can share a config file and someone else can run the exact same workflow.

Oumi has four types of configs:

| Config Type | What it controls | Learn more |
|------------|------------------|------------|
| Training | Model, data, hyperparameters, and training settings | {doc}`/user_guides/train/configuration` |
| Evaluation | Benchmarks and metrics to run | {doc}`/user_guides/evaluate/evaluate` |
| Inference | How to generate text from a model | {doc}`/user_guides/infer/infer` |
| Launcher | Where and how to run jobs (local, cloud, etc.) | {doc}`/user_guides/launch/launch` |

Here's what a training config looks like:

```yaml
# The model to train
model:
  name: meta-llama/Llama-3.1-70B-Instruct
  trust_remote_code: true

# Training data
data:
  train:
    datasets:
      - dataset_name: text_sft
        dataset_path: path/to/data
    stream: true

# Training hyperparameters
training:
  trainer_type: TRL_SFT
  learning_rate: 1e-4
  max_steps: 1000
```

Oumi comes with many ready-to-use configs called **recipes**. Browse them at {doc}`/resources/recipes`.

## Key Components

Now that you understand how to run Oumi, let's look at the main components you'll work with.

### Recipes

A **recipe** is a complete, ready-to-run configuration file. Oumi includes recipes for common workflows like fine-tuning Llama or training SmolLM. Think of recipes as starting points—you can use them as-is or customize them for your needs.

Browse available recipes: {doc}`/resources/recipes`

### Models

Oumi works with most models from HuggingFace's `transformers` library. You specify a model by its HuggingFace name (like `meta-llama/Llama-3.1-8B`) in your config file. You can also define custom model architectures.

Learn more: {doc}`/resources/models/custom_models`

### Datasets

Oumi provides a unified interface for loading and preprocessing training data. You can use datasets from HuggingFace, load local files, or create custom dataset classes.

**Data mixtures** let you combine multiple datasets with different weights—useful when you want to train on diverse data sources simultaneously.

Learn more: {doc}`/resources/datasets/datasets`

### Training Methods

Oumi supports multiple training approaches through different **trainers**:

- **SFT trainers** for supervised fine-tuning
- **DPO trainers** for preference-based alignment
- **GRPO trainers** for reinforcement learning

Each trainer handles the optimization loop, gradient updates, and checkpointing for its respective method.

Learn more: {doc}`/user_guides/train/training_methods`

### Oumi Judge

**Oumi Judge** uses an LLM to evaluate model outputs on attributes like helpfulness, honesty, and safety. It's useful for automated quality assessment when you don't have ground-truth labels.

Learn more: {doc}`/user_guides/judge/judge`

### Launcher

The **launcher** lets you run Oumi jobs on different platforms—your local machine, a GPU cluster, or cloud providers like AWS and GCP. You define where to run in a launcher config, keeping your training config portable.

Learn more: {doc}`/user_guides/launch/launch`

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
