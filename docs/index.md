<div class="oumi-hero-stats">
  <h1 class="oumi-hero-title">Open Universal Machine Intelligence</h1>
  <p class="oumi-hero-tagline">Everything you need to build state-of-the-art foundation models, end-to-end.</p>
  <div class="oumi-stats-bar">
    <a href="resources/recipes.html" class="oumi-stat">
      <span class="oumi-stat-value">200+</span>
      <span class="oumi-stat-label">Recipes</span>
    </a>
    <span class="oumi-stat-divider"></span>
    <a href="resources/models/models.html" class="oumi-stat">
      <span class="oumi-stat-value">100+</span>
      <span class="oumi-stat-label">Models</span>
    </a>
    <span class="oumi-stat-divider"></span>
    <a href="https://github.com/oumi-ai/oumi" class="oumi-stat">
      <span class="oumi-stat-value">8.8k</span>
      <span class="oumi-stat-label">GitHub Stars</span>
    </a>
  </div>
  <a href="get_started/quickstart.html" class="oumi-hero-cta">Get Started →</a>
</div>

## What is Oumi?

Oumi is an open-source platform designed for ML engineers and researchers who want to train, fine-tune, evaluate, and deploy foundation models. Whether you're fine-tuning a small language model on a single GPU or training a 405B parameter model across a cluster, Oumi provides a unified interface that scales with your needs.

**Who is Oumi for?**

- **ML Engineers** building production AI systems who need reliable training pipelines and deployment options
- **Researchers** experimenting with new training methods, architectures, or datasets
- **Teams** who want a consistent workflow from local development to cloud-scale training

**What problems does Oumi solve?**

- **Fragmented tooling**: Instead of stitching together different libraries for training, evaluation, and deployment, Oumi provides one cohesive platform
- **Scaling complexity**: The same configuration works locally and on cloud infrastructure (AWS, GCP, Azure, Lambda Labs)
- **Reproducibility**: YAML-based configs make experiments easy to track, share, and reproduce

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

Home <self>
get_started/quickstart
get_started/installation
get_started/core_concepts
get_started/tutorials
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guides

user_guides/train/train
user_guides/infer/infer
user_guides/evaluate/evaluate
user_guides/analyze/analyze
user_guides/judge/judge
user_guides/launch/launch
user_guides/synth
user_guides/tune
user_guides/quantization
user_guides/customization
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Resources

resources/models/models
resources/datasets/datasets
resources/recipes

```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

API Reference <api/oumi>
CLI Reference <cli/commands>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: FAQ

faq/troubleshooting
faq/oom
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Development

development/dev_setup
development/contributing
development/code_of_conduct
development/style_guide
development/docs_guide
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: About

about/acknowledgements
about/license
about/citations
```

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 🚀 Quickstart
:link: get_started/quickstart
:link-type: doc

Install and run your first training job in **5 minutes**
:::

:::{grid-item-card} 📚 Core Concepts
:link: get_started/core_concepts
:link-type: doc

Understand configs, models, and workflows
:::

:::{grid-item-card} 🎯 Training Guide
:link: user_guides/train/train
:link-type: doc

Deep dive into training options
:::

::::

## Quick Start

**Prerequisites:** Python 3.10+, pip. GPU recommended for larger models (CPU works for small models like SmolLM-135M).

Install Oumi and start training in minutes:

```bash
# Install with GPU support (or use `pip install oumi` for CPU-only)
pip install oumi[gpu]

# Train a model
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

# Run inference
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

For detailed setup instructions including virtual environments and cloud setup, see the {doc}`installation guide <get_started/installation>`.

## Hands on Notebooks

| **Notebook** | **Try in Colab** | **Goal** |
|----------|--------------|-------------|
| **🎯 Getting Started: A Tour** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - A Tour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Quick tour of core features: training, evaluation, inference, and job management |
| **🔧 Model Finetuning Guide** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Finetuning Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | End-to-end guide to LoRA tuning with data prep, training, and evaluation |
| **📚 Model Distillation** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Distill a Large Model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Guide to distilling large models into smaller, efficient ones |
| **📋 Model Evaluation** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Evaluation with Oumi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Comprehensive model evaluation using Oumi's evaluation framework |
| **☁️ Remote Training** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Running Jobs Remotely.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Launch and monitor training jobs on cloud (AWS, Azure, GCP, Lambda, etc.) platforms |
| **📈 LLM-as-a-Judge** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Simple Judge.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Filter and curate training data with built-in judges |

## Documentation Guide

A complete map of the documentation to help you find what you need:

| Category | Description | Links |
|----------|-------------|-------|
| **Getting Started** | Installation, quickstart, and core concepts | {doc}`Quickstart <get_started/quickstart>` · {doc}`Installation <get_started/installation>` · {doc}`Core Concepts <get_started/core_concepts>` |
| **User Guides** | In-depth guides for each capability | {doc}`Training <user_guides/train/train>` · {doc}`Inference <user_guides/infer/infer>` · {doc}`Evaluation <user_guides/evaluate/evaluate>` · {doc}`Analysis <user_guides/analyze/analyze>` |
| **Resources** | Models, datasets, and ready-to-use recipes | {doc}`Models <resources/models/models>` · {doc}`Datasets <resources/datasets/datasets>` · {doc}`Recipes <resources/recipes>` |
| **Reference** | API and CLI documentation | {doc}`Python API <api/oumi>` · {doc}`CLI Reference <cli/commands>` |
| **Development** | Contributing to Oumi | {doc}`Dev Setup <development/dev_setup>` · {doc}`Contributing <development/contributing>` · {doc}`Style Guide <development/style_guide>` |

## Feature Highlights

Explore Oumi's core capabilities:

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} Training
:link: user_guides/train/train
:link-type: doc

Train models from 10M to 405B parameters with SFT, LoRA, QLoRA, DPO, GRPO, and more.
:::

:::{grid-item-card} Inference
:link: user_guides/infer/infer
:link-type: doc

Deploy models with vLLM, SGLang, or native inference. Local and remote engines supported.
:::

:::{grid-item-card} Evaluation
:link: user_guides/evaluate/evaluate
:link-type: doc

Evaluate across standard benchmarks with LM Evaluation Harness integration.
:::

:::{grid-item-card} Analysis
:link: user_guides/analyze/analyze
:link-type: doc

Profile datasets, identify outliers, and filter data before training.
:::

:::{grid-item-card} Data Synthesis
:link: user_guides/synth
:link-type: doc

Generate synthetic training data with LLM-powered pipelines.
:::

:::{grid-item-card} Cloud Deployment
:link: user_guides/launch/launch
:link-type: doc

Launch jobs on AWS, GCP, Azure, Lambda, and other cloud providers.
:::

::::

## Join the Community

Oumi is a community-first effort. Whether you are a developer, a researcher, or a non-technical user, all contributions are very welcome!

- To contribute to the `oumi` repository, please check the [`CONTRIBUTING.md`](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md) for guidance on how to contribute to send your first Pull Request.
- Make sure to join our [Discord community](https://discord.gg/oumi) to get help, share your experiences, and contribute to the project!
- If you are interested by joining one of the community's open-science efforts, check out our [open collaboration](https://oumi.ai/community) page.

## Need Help?

If you encounter any issues or have questions, please don't hesitate to:

1. Check our {doc}`FAQ section <faq/troubleshooting>` for common questions and answers.
2. Open an issue on our [GitHub Issues page](https://github.com/oumi-ai/oumi/issues) for bug reports or feature requests.
3. Join our [Discord community](https://discord.gg/oumi) to chat with the team and other users.
