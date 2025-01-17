<div align="center">
<img src="_static/logo/oumi_logo_dark.png" alt="Oumi Logo" width="150"/>

# Oumi: Open Universal Machine Intelligence

E2E Foundation Model Research Platform - Community-first & Enterprise-grade
</div>

[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Github](https://img.shields.io/badge/Github-oumi-blue.svg)](https://github.com/oumi-ai/oumi)
[![Discord](https://img.shields.io/discord/1286348126797430814?label=Discord)](https://discord.gg/oumi)
[![GitHub Repo stars](https://img.shields.io/github/stars/oumi-ai/oumi)](https://github.com/oumi-ai/oumi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


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

Oumi is a community-first, end-to-end platform for advanced AI research and development. It provides comprehensive support for foundation model workflows - from pre-training and post-training, to data curation, synthesis, and evaluation. Built with enterprise-grade quality and reliability, Oumi serves both researchers pushing the boundaries of AI and organizations building production-ready solutions.

## Features

Oumi is designed to be fully flexible yet easy to use:

### 🤖 Model Support

- **Comprehensive Model Coverage**: Support for leading models including Llama 3.1/3.2/3.3 (up to 405B), Mistral, Gemma, Qwen2, {doc}`and more <resources/models/supported_models>`
- **Vision-Language Models**: First-class support for multimodal models like Llama 3.2 Vision, LLaVA-1.5, and Qwen2-VL
- **Unified Interface**: One consistent API for all models, whether from HuggingFace Hub or custom implementations

### 🚀 Training & Optimization

- **End-to-End Training**: Support for the full training lifecycle (pre-training, SFT, DPO, guardrails, and more)
- **Full Parameter Training**: Full fine-tuning with DDP, FSDP support and long context capabilities
- **Efficient Fine-tuning**: LoRA, QLoRA for parameter-efficient fine-tuning
- **Built for Scale**: Efficiently handle models up to 405B parameters with distributed training and inference on large clusters

### 📊 Inference & Deployment

- **Multiple Backends**: Native PyTorch, vLLM/SG-Lang for optimized serving
- **Flexible Deployment**: Run anywhere from local machines to cloud clusters
- **Production Ready**: Enterprise-grade reliability and comprehensive testing

### 🔧 Development Tools

- **Rich Configuration**: Version-controlled YAML configs for reproducible experiments
- **Extensible Platform**: Easy model & dataset registration and customization
- **Comprehensive Documentation**: Detailed guides, examples, and API reference

## Where to go next?

While you can dive directly into any section that interests you, we recommend following the suggested path below to get the most out of Oumi.

| Category | Description | Links |
|----------|-------------|-------|
| 🚀 Getting Started | Get up and running quickly with Oumi | [→ Quickstart](get_started/quickstart)<br>[→ Installation](get_started/installation)<br>[→ Core Concepts](get_started/core_concepts) |
| 📚 User Guides | Learn how to use Oumi effectively | [→ Training](user_guides/train/train)<br>[→ Inference](user_guides/infer/infer)<br>[→ Evaluation](user_guides/evaluate/evaluate) |
| 🤖 Models | Explore available models and recipes | [→ Overview](resources/models/models)<br>[→ Recipes](resources/recipes)<br>[→ Custom Models](resources/models/custom_models) |
| 🔧 Development | Contribute to Oumi | [→ Dev Setup](development/dev_setup)<br>[→ Contributing](development/contributing)<br>[→ Style Guide](development/style_guide) |
| 📖 API Reference | Documentation of all modules | [→ Python API](api/oumi)<br>[→ CLI](cli/commands) |

## 🔗 Contributing

We welcome contributions! See our {doc}`Contributing Guide <development/contributing>` for information on how to get involved, including guidelines for code style, testing, and submitting pull requests.

If there's a feature that you think is missing, let us know or join us in making it a reality by sending a [feature request](https://github.com/oumi-ai/oumi/issues/new?assignees=&labels=Enhancement&projects=&template=feature-request.yaml&title=%5BFeature%5D%3A+), or {doc}`contributing directly <development/contributing>`!

## ❓ Need Help?

If you encounter any issues or have questions, please don't hesitate to:

1. Check our {doc}`FAQ section <faq/troubleshooting>` for common questions and answers.
2. Open an issue on our [GitHub Issues page](https://github.com/oumi-ai/oumi/issues) for bug reports or feature requests.
3. Join our [Discord community](https://discord.gg/oumi) to chat with the team and other users.
