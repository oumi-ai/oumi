<table border="0">
 <tr>
    <td width="150">
      <img src="docs/_static/logo/oumi_logo_dark.png" alt="Oumi Logo" width="150"/>
    </td>
    <td>
      <h1>Oumi: Open Universal Machine Intelligence</h1>
      <p>E2E multimodal frontier AI Platform - Community-first & Enterprise-grade</p>
    </td>
 </tr>
</table>

[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pre-review Tests](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml)
[![Documentation](https://img.shields.io/badge/docs-oumi-blue.svg)](https://oumi.ai/docs/latest/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Oumi is a community-first, end-to-end platform for advanced AI research and development. It provides comprehensive support for foundation model workflows - from pretraining and post-training, to data curation, synthesis and evaluation. Built with enterprise-grade quality and reliability, Oumi serves both researchers pushing the boundaries of AI and organizations building production-ready solutions.

<p align="center">
   <b>Check out our docs!</b>
   <br>
   ↓↓↓↓↓↓
   <br>
   https://oumi.ai/docs
   <br>
   <b>Password:</b> c155c7d02520
   <br>
   ↑↑↑↑↑↑
</p>

## Features

Oumi is designed to be fully flexible and yet easy to use:

- **Run Anywhere**: Train and evaluate models seamlessly across environments - from local machines to remote clusters, with native support for Jupyter notebooks and VS Code debugging.

- **Comprehensive Training**: Support for the full ML lifecycle - from pretraining to fine-tuning (SFT, LoRA, QLoRA, DPO) and evaluation. Built for both research exploration and production deployment.

- **Built for Scale**: First-class support for distributed training with PyTorch DDP and FSDP. Efficiently handle models up to 405B parameters.

- **Reproducible Research**: Version-controlled configurations via YAML files and CLI arguments ensure fully reproducible experiments across training and evaluation pipelines.

- **Unified Interface**: One consistent API for everything - data processing, training, evaluation, and inference. Seamlessly work with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI).

- **Extensible Architecture**: Easily add new models, datasets, training approaches and evaluation metrics. Built with modularity in mind.

- **Production Ready**: Comprehensive test coverage, detailed documentation, and enterprise-grade support make Oumi reliable for both research and production use cases.

We're just getting started on this journey, and we can't wait to build Oumi together! If there's a feature that you think is missing, let us know or join us in making it a reality:

- [Request a feature](https://github.com/oumi-ai/oumi/issues/new?template=feature_request.md)
- [Contribute](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md)

For a full tour of what Oumi can do, dive into our [documentation](https://oumi.ai/docs).

## Getting Started

With just a couple commands you can install Oumi, train, infer, and evaluate. All it would take is something like the following:

### Installation

1. Install Oumi:

   ```shell
   pip install 'oumi'
   ```

### Usage

1. Run training locally:

   ```shell
   oumi train -c path/to/your/config.yaml
   ```

   For more advanced training options, see the [training guide](/docs/user_guides/train/train.md) and [distributed training](docs/advanced/distributed_training.md).

### Configurations

These configurations demonstrate how to setup and run full training for different model architectures using Oumi.

| Model | Type | Configuration | Cluster | Status |
|-------|------|---------------|---------|--------|
| **Llama Instruction Finetuning** | | | | |
| Llama3.1 8b | LoRA | [polaris_job.yaml](/configs/recipes/llama3_1/sft/8b_lora/polaris_job.yaml) | Polaris | ✨ Supported ✨ |
| Llama3.1 8b | SFT | [polaris_job.yaml](/configs/recipes/llama3_1/sft/8b_full/polaris_job.yaml) | Polaris | ✨ Supported ✨ |
| Llama3.1 70b | LoRA | [polaris_job.yaml](/configs/recipes/llama3_1/sft/70b_lora/polaris_job.yaml) | Polaris | ✨ Supported ✨ |
| Llama3.1 70b | SFT | [polaris_job.yaml](/configs/recipes/llama3_1/sft/70b_full/polaris_job.yaml) | Polaris | ✨ Supported ✨ |
| **Example Models** | | | | |
| Aya | SFT | [train.yaml](configs/projects/aya/sft/train.yaml) | GCP | ✨ Supported ✨ |
| Zephyr |QLoRA | [qlora_train.yaml](/configs/projects/zephyr/sft/qlora_train.yaml) | GCP | ✨ Supported ✨ |
| ChatQA | SFT | [chatqa_stage1_train.yaml](/configs/projects/chatqa/sft/chatqa_stage1_train.yaml) | GCP | ✨ Supported ✨ |
| **Pre-training** | | | | |
| GPT-2 | Pre-training | [gpt2.pt.mac.yaml](/configs/recipes/gpt2/pretraining/mac_train.yaml) | Mac (mps) | ✨ Supported ✨ |
| Llama2 2b | Pre-training | [fineweb.pt.yaml](/configs/examples/fineweb_ablation_pretraining/ddp/train.yaml) | Polaris | ✨ Supported ✨ |

## Tutorials

We provide several Jupyter notebooks to help you get started with Oumi. Here's a list of available examples:

| Notebook | Description |
|----------|-------------|
| [A Tour](/notebooks/Oumi%20-%20A%20Tour.ipynb) | A comprehensive tour of the Oumi repository and its features |
| [Finetuning Tutorial](/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb) | Step-by-step guide on how to finetune models using Oumi |
| [Tuning Llama](/notebooks/Oumi%20-%20Tuning%20Llama.ipynb) | Detailed tutorial on tuning Llama models with Oumi |
| [Multinode Inference on Polaris](/notebooks/Oumi%20-%20Multinode%20Inference%20on%20Polaris.ipynb) | Guides you through running inference with trained models |
| [Datasets Tutorial](/notebooks/Oumi%20-%20Datasets%20Tutorial.ipynb) | Explains how to work with datasets in Oumi |
| [Deploying a Job](/notebooks/Oumi%20-%20Deploying%20a%20Job.ipynb) | Instructions on how to deploy a training job using Oumi |

## Documentation

View our API documentation [here](https://oumi.ai/docs/latest/index.html).

Reach out to <matthew@oumi.ai> if you have problems with access.

## Contributing

Contributions are welcome! After all, this is a community-based effort. Please check the `CONTRIBUTING.md` file for guidelines on how to contribute to the project.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Troubleshooting

1. Pre-commit hook errors with VS Code
   - When committing changes, you may encounter an error with pre-commit hooks related to missing imports.
   - To fix this, make sure to start your vscode instance after activating your conda environment.

     ```shell
     conda activate oumi
     code .  # inside the Oumi directory
     ```
