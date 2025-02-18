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
user_guides/judge/judge
user_guides/launch/launch
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

about/changelog
about/acknowledgements
about/license
about/citations
```

<p align="center">
    <a href="https://trendshift.io/repositories/12865">
        <img alt="GitHub trending" src="https://trendshift.io/api/badge/repositories/12865" />
    </a>
</p>

Oumi is a fully open-source platform that streamlines the entire lifecycle of foundation models - from data preparation and training to evaluation and deployment. Whether you're developing on a laptop, launching large scale experiments on a cluster, or deploying models in production, Oumi provides the tools and workflows you need.

With Oumi, you can:

- 🚀 Train and fine-tune models from 10M to 405B parameters using state-of-the-art techniques (SFT, LoRA, QLoRA, DPO, and more)
- 🤖 Work with both text and multimodal models (Llama, DeepSeek, Qwen, Phi, and others)
- 🔄 Synthesize and curate training data with LLM judges
- ⚡️ Deploy models efficiently with popular inference engines (vLLM, SGLang)
- 📊 Evaluate models comprehensively across standard benchmarks
- 🌎 Run anywhere - from laptops to clusters to clouds (AWS, Azure, GCP, Lambda, and more)
- 🔌 Integrate with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI, Parasail, ...)

All with one consistent API, production-grade reliability, and all the flexibility you need for research. Oumi is currently in <ins>beta</ins> and under active development.

## 🚀 Getting Started

| **Notebook** | **Try in Colab** | **Goal** |
|----------|--------------|-------------|
| **🎯 Getting Started: A Tour** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - A Tour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Quick tour of core features: training, evaluation, inference, and job management |
| **🔧 Model Finetuning Guide** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Finetuning Tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | End-to-end guide to LoRA tuning with data prep, training, and evaluation |
| **📚 Model Distillation** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Distill a Large Model.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Guide to distilling large models into smaller, efficient ones |
| **📋 Model Evaluation** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Evaluation with Oumi.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Comprehensive model evaluation using Oumi's evaluation framework |
| **☁️ Remote Training** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Running Jobs Remotely.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Launch and monitor training jobs on cloud (AWS, Azure, GCP, Lambda, etc.) platforms |
| **📈 LLM-as-a-Judge** | <a target="_blank" href="https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi - Oumi Judge.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | Filter and curate training data with built-in judges |

## 💻 Why use Oumi?

If you need a comprehensive platform for training, evaluating, or deploying models, Oumi is a great choice.

Here are some of the key features that make Oumi stand out:

- 🔧 **Zero Boilerplate**: Get started in minutes with ready-to-use recipes for popular models and workflows. No need to write training loops or data pipelines.
- 🏢 **Enterprise-Grade**: Built and validated by teams training models at scale
- 🎯 **Research Ready**: Perfect for ML research with easily reproducible experiments, and flexible interfaces for customizing each component.
- 🌐 **Broad Model Support**: Works with most popular model architectures - from tiny models to the largest ones, text-only to multimodal.
- 🚀 **SOTA Performance**: Native support for distributed training techniques (FSDP, DDP) and optimized inference engines (vLLM, SGLang).
- 🤝 **Community First**: 100% open source with an active community. No vendor lock-in, no strings attached.


## 📖 Where to go next?

While you can dive directly into any section that interests you, we recommend following the suggested path below to get the most out of Oumi.

| Category | Description | Links |
|----------|-------------|-------|
| 🚀 Getting Started | Get up and running quickly with Oumi | [→ Quickstart](get_started/quickstart)<br>[→ Installation](get_started/installation)<br>[→ Core Concepts](get_started/core_concepts) |
| 📚 User Guides | Learn how to use Oumi effectively | [→ Training](user_guides/train/train)<br>[→ Inference](user_guides/infer/infer)<br>[→ Evaluation](user_guides/evaluate/evaluate) |
| 🤖 Models | Explore available models and recipes | [→ Overview](resources/models/models)<br>[→ Recipes](resources/recipes)<br>[→ Custom Models](resources/models/custom_models) |
| 🔧 Development | Contribute to Oumi | [→ Dev Setup](development/dev_setup)<br>[→ Contributing](development/contributing)<br>[→ Style Guide](development/style_guide) |
| 📖 API Reference | Documentation of all modules | [→ Python API](api/oumi)<br>[→ CLI](cli/commands) |


## 🤝 Join the Community!

Oumi is a community-first effort. Whether you are a developer, a researcher, or a non-technical user, all contributions are very welcome!

- To contribute to the `oumi` repository, please check the [`CONTRIBUTING.md`](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md) for guidance on how to contribute to send your first Pull Request.
- Make sure to join our [Discord community](https://discord.gg/oumi) to get help, share your experiences, and contribute to the project!
- If you are interested by joining one of the community's open-science efforts, check out our [open collaboration](https://oumi.ai/community) page.


## ❓ Need Help?

If you encounter any issues or have questions, please don't hesitate to:

1. Check our {doc}`FAQ section <faq/troubleshooting>` for common questions and answers.
2. Open an issue on our [GitHub Issues page](https://github.com/oumi-ai/oumi/issues) for bug reports or feature requests.
3. Join our [Discord community](https://discord.gg/oumi) to chat with the team and other users.
