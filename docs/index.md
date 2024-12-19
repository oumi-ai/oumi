<div align="center">
<img src="_static/logo/oumi_logo_dark.png" alt="Oumi Logo" width="150"/>

# Oumi: Open Universal Machine Intelligence

E2E Foundation Model Research Platform - Community-first & Enterprise-grade
</div>

[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

```{toctree}
:maxdepth: 2
:hidden:
:caption: Get started

get_started/installation
get_started/quickstart
get_started/tutorials
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide

user_guides/train/train
user_guides/infer/infer
user_guides/evaluate/evaluate
user_guides/judge/judge
user_guides/launch/launch
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Advanced Topics

advanced/customization
advanced/quantization
advanced/performance_optimization
advanced/distributed_training
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Models

models/recipes
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Datasets

datasets/local_datasets
datasets/pretraining
datasets/sft
datasets/preference_tuning
datasets/vl_sft
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: FAQ

faq/troubleshooting
faq/oom
faq/gpu_sizing
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API Reference

API Reference <api/oumi>
CLI Reference <cli/commands>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Development

development/dev_setup
development/contributing
development/code_of_conduct
development/style_guide
development/git_workflow
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

Oumi is a community-first, end-to-end platform for advanced AI research and development. It provides comprehensive support for foundation model workflows - from pretraining and post-training, to data curation, synthesis, and evaluation. Built with enterprise-grade quality and reliability, Oumi serves both researchers pushing the boundaries of AI and organizations building production-ready solutions.

## Features

Oumi is designed to be fully flexible yet easy to use:

- **Run Anywhere**: Train and evaluate models seamlessly across environments - from local machines to remote clusters, with native support for Jupyter notebooks and VS Code debugging.

- **Comprehensive Training**: Support for the full ML lifecycle - from pretraining to fine-tuning (SFT, LoRA, QLoRA, DPO) to evaluation. Built for both research exploration and production deployment.

- **Built for Scale**: First-class support for distributed training with PyTorch DDP and FSDP. Efficiently handle models up to 405B parameters.

- **Reproducible Research**: Version-controlled configurations via YAML files and CLI arguments ensure fully reproducible experiments across training and evaluation pipelines.

- **Unified Interface**: One consistent interface for everything - data processing, training, evaluation, and inference. Seamlessly work with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI).

- **Extensible Architecture**: Easily add new models, datasets, training approaches and evaluation metrics. Built with modularity in mind.

- **Production Ready**: Comprehensive test coverage, detailed documentation, and enterprise-grade support make Oumi reliable for both research and production use cases.

If there's a feature that you think is missing, let us know or join us in making it a reality by sending a [feature request](https://github.com/oumi-ai/oumi/issues/new?template=feature_request.md), or [contributing directly](development/contributing)!

## Getting Started

If you're new to Oumi, we recommend starting with the following sections:

1. [Installation](get_started/installation) - Install Oumi on your system.
2. [Quickstart](get_started/quickstart) - Quickstart guide to get you up and running with training, evaluation and inference in no time.
3. [Recipes](get_started/tutorials) - Tutorials and recipes to get you started with Oumi with various models, datasets and workflows.

## API Reference

For detailed information about the Oumi library, check out the [API Reference](api/oumi) section.

This includes comprehensive documentation for all modules, classes, and functions in the Oumi library.

## Contributing

We welcome contributions! See our [Contributing Guide](development/contributing) for information on how to get involved, including guidelines for code style, testing, and submitting pull requests.

## Need Help?

If you encounter any issues or have questions, please don't hesitate to:

1. Check our [FAQ section](faq/troubleshooting) for common questions and answers.
2. Open an issue on our [GitHub Issues page](https://github.com/oumi-ai/oumi/issues) for bug reports or feature requests.
3. Send us a message on [Discord](https://discord.gg/S74NxTDh7v) to chat with the team and other users.
