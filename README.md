# Open Unified Machine Intelligence (Oumi)

[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pre-review Tests](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml)
[![Documentation](https://img.shields.io/badge/docs-oumi-blue.svg)](https://learning-machines.ai/docs/latest/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

The Oumi Platform enables the end-to-end development of foundation and specialized models including data curation, data synthesis, pretraining, tuning, and evaluation.

## Features

- **Run Anywhere**: Train and evaluate models seamlessly across local environments, Jupyter notebooks, VS Code debugger, or remote clusters.
- **Any Training**: Pretraining and comprehensive instruction fine-tuning capabilities, including FFT, LoRA, DPO, and more.
- **Scalability**: Built-in support for multi-node distributed training using PyTorch's DistributedDataParallel (DDP) or Fully Sharded Data Parallel (FSDP). Inference support for Llama 405B and beyond.
- **Cloud Flexibility**: Compatible with major cloud providers (GCP, AWS, Azure, ...) and specialized platforms like DOE ALCF Polaris.
- **Reproducibility**: Flexible configuration system using YAML files and command-line arguments.
- **Unified Interface**: Streamlined processes for data preprocessing, model training, and evaluation.
- **Customizable**: Easily extendable to incorporate new models, datasets, and evaluation metrics.

## Getting Started

For an overview of the Oumi features and usage, checkout the [user guide](/USAGE.md) and the [hands-on tour of the repository](/notebooks/Oumi%20-%20A%20Tour.ipynb).

### Quickstart

0. (Optional) Set up Git and Conda:

   For new developers, we highly recommend that you follow the [installation guide](/docs/DEV_SETUP.md) to help set up Git and a local conda environment.

1. Install Oumi:

   ```shell
   pip install 'oumi[all]'
   ```

2. Set up your configuration file (example configs are provided in the [configs](/configs) directory).

3. Run training locally:

   ```shell
   oumi-train -c path/to/your/config.yaml
   ```

   For more advanced training options, see [cloud training guide](/docs/CLOUD_TRAINING.md) and [distributed training](/docs/DISTRIBUTED_TRAINING.md).

### Configurations

These configurations demonstrate how to setup and run full training for different model architectures using Oumi.

| Model | Type | Configuration | Cluster | Status |
|-------|------|---------------|---------|--------|
| **Llama Instruction Finetuning** | | | | |
| Llama3.1 8b | LoRA | [llama8b_lora.yaml](/configs/oumi/jobs/polaris/llama8b_lora.yaml) | Polaris | ✨ Supported ✨ |
| Llama3.1 8b | SFT | [llama8b_sft.yaml](/configs/oumi/jobs/polaris/llama8b_sft.yaml) | Polaris | ✨ Supported ✨ |
| Llama3.1 70b | LoRA | [llama70b_lora.yaml](/configs/oumi/jobs/polaris/llama70b_lora.yaml) | Polaris | ✨ Supported ✨ |
| Llama3.1 70b | SFT | [llama70b_sft.yaml](/configs/oumi/jobs/polaris/llama70b_sft.yaml) | Polaris | ✨ Supported ✨ |
| **Example Models** | | | | |
| Aya | SFT | [llama3.8b.aya.sft.yaml](/configs/oumi/llama3.8b.aya.sft.yaml) | GCP | ✨ Supported ✨ |
| Zephyr |QLoRA | [zephyr.7b.qlora.yaml](/configs/oumi/zephyr.7b/sft/qlora.yaml) | GCP | ✨ Supported ✨ |
| ChatQA | SFT | [chatqa.stage1.yaml](/configs/oumi/chatqa/chatqa.stage1.yaml) | GCP | ✨ Supported ✨ |
| **Pre-training** | | | | |
| GPT-2 | Pre-training | [gpt2.pt.mac.yaml](/configs/oumi/gpt2.pt.mac.yaml) | Mac (mps) | ✨ Supported ✨ |
| Llama2 2b | Pre-training | [llama2b.pt.yaml](/configs/oumi/llama2b.pt.yaml) | Polaris | ✨ Supported ✨ |

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

View our API documentation [here](https://learning-machines.ai/docs/latest/index.html).

Reach out to <matthew@learning-machines.ai> if you have problems with access.

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
