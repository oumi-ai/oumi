<div align="center">
<img src="_static/logo/oumi_logo_dark.png" alt="Oumi Logo" width="150"/>
<h1> Oumi: Open Universal Machine Intelligence </h1>
</div>

[![Github](https://img.shields.io/badge/Github-oumi-blue.svg)](https://github.com/oumi-ai/oumi)
[![Blog](https://img.shields.io/badge/Blog-oumi-blue.svg)](https://oumi.ai/blog)
[![Discord](https://img.shields.io/discord/1286348126797430814?label=Discord)](https://discord.gg/oumi)
[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Repo stars](https://img.shields.io/github/stars/oumi-ai/oumi)](https://github.com/oumi-ai/oumi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![About](https://img.shields.io/badge/About-oumi-blue.svg)](https://oumi.ai)

<h4> Everything you need to build state-of-the-art foundation models, end-to-end. </h4>

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

Oumi is a fully open-source platform that streamlines the entire lifecycle of foundation models - from data preparation and training to evaluation and deployment.

## Quick Start

Install Oumi and start training in minutes:

```bash
# Install with GPU support
pip install oumi[gpu]

# Train a model
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

# Run inference
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

For detailed setup instructions, see the {doc}`installation guide <get_started/installation>`.

## 🚀 Go Deeper

| **Notebook** | **Try in Colab** | **Goal** |
|----------|--------------|-------------|
| **🎯 Getting Started: A Tour** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - A Tour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Quick tour of core features: training, evaluation, inference, and job management |
| **🔧 Model Finetuning Guide** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Finetuning Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | End-to-end guide to LoRA tuning with data prep, training, and evaluation |
| **📚 Model Distillation** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Distill a Large Model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Guide to distilling large models into smaller, efficient ones |
| **📋 Model Evaluation** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Evaluation with Oumi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Comprehensive model evaluation using Oumi's evaluation framework |
| **☁️ Remote Training** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Running Jobs Remotely.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Launch and monitor training jobs on cloud (AWS, Azure, GCP, Lambda, etc.) platforms |
| **📈 LLM-as-a-Judge** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Simple Judge.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Filter and curate training data with built-in judges |

## Documentation Guide

| Category | Description | Links |
|----------|-------------|-------|
| **Getting Started** | Installation, quickstart, and core concepts | {doc}`Quickstart <get_started/quickstart>` · {doc}`Installation <get_started/installation>` · {doc}`Core Concepts <get_started/core_concepts>` |
| **User Guides** | In-depth guides for each capability | {doc}`Training <user_guides/train/train>` · {doc}`Inference <user_guides/infer/infer>` · {doc}`Evaluation <user_guides/evaluate/evaluate>` · {doc}`Analysis <user_guides/analyze/analyze>` |
| **Resources** | Models, datasets, and ready-to-use recipes | {doc}`Models <resources/models/models>` · {doc}`Datasets <resources/datasets/datasets>` · {doc}`Recipes <resources/recipes>` |
| **Reference** | API and CLI documentation | {doc}`Python API <api/oumi>` · {doc}`CLI Reference <cli/commands>` |
| **Development** | Contributing to Oumi | {doc}`Dev Setup <development/dev_setup>` · {doc}`Contributing <development/contributing>` · {doc}`Style Guide <development/style_guide>` |

## Key Capabilities

::::{grid} 2
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

## 🤝 Join the Community

Oumi is a community-first effort. Whether you are a developer, a researcher, or a non-technical user, all contributions are very welcome!

- To contribute to the `oumi` repository, please check the [`CONTRIBUTING.md`](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md) for guidance on how to contribute to send your first Pull Request.
- Make sure to join our [Discord community](https://discord.gg/oumi) to get help, share your experiences, and contribute to the project!
- If you are interested by joining one of the community's open-science efforts, check out our [open collaboration](https://oumi.ai/community) page.

## ❓ Need Help?

If you encounter any issues or have questions, please don't hesitate to:

1. Check our {doc}`FAQ section <faq/troubleshooting>` for common questions and answers.
2. Open an issue on our [GitHub Issues page](https://github.com/oumi-ai/oumi/issues) for bug reports or feature requests.
3. Join our [Discord community](https://discord.gg/oumi) to chat with the team and other users.
