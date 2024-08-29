# Learning Machines (LeMa)

LeMa is a learning machines modeling platform that allows you to build foundation models end-to-end including data curation/synthesis, pretraining, tuning, and evaluation.

## Features

- **Run anywhere**: Run training and evaluation seamlessly across local environments, Jupyter notebooks, vscode debugger, or remote clusters
- **Instruction Fine-tuning**: Comprehensive instruction fine-tuning capabilities, including SFT, DPO, LoRA, and more
- **Scalable Training**: Built-in support for distributed training using PyTorch's DistributedDataParallel (DDP) or Fully Sharded Data Parallel (FSDP).
- **Cloud Flexibility**: Compatible with major cloud providers (GCP, AWS, Azure, ...) and specialized platforms like DOE ALCF Polaris*
- **Reproducibility**: Flexible configuration system using YAML files and command-line arguments
- **Unified Interface**: Streamlined processes for data preprocessing, model training, and evaluation
- **Customizable**: Easily extendable to incorporate new models, datasets, and evaluation metrics

## Getting Started

For an overview of the LeMa features and usage, checkout the [user guide](/USAGE.md) and the [hands on tour of the repository](/notebooks/LeMa%20-%20A%20Tour.ipynb).

### Quickstart

1. Install the package:

   ```shell
   pip install 'lema[cloud,dev,train,gpu]'
   ```

   For detailled instructions to setup your environment, see [installation guide](/docs/DEV_SETUP.md).

2. Set up your configuration file (example configs are provided in the [configs](/configs) directory).

3. Run training locally:

   ```shell
   lema-train -c path/to/your/config.yaml
   ```

   For more advanced training options, see [cloud training guide](/docs/CLOUD_TRAINING.md) and [distributed training](/docs/DISTRIBUTED_TRAINING.md).

### Configurations

These configurations demonstrate how to set up and run full training for different model architectures using LeMa.

| Model | Type | Configuration | Cluster | Status |
|-------|------|---------------|---------|--------|
| **Llama Instruction Finetuning** | | | | |
| Llama3.1 8b | LoRA | [llama8b_lora.yaml](/configs/lema/jobs/polaris/llama8b_lora.yaml) | Polaris | ✨ |
| Llama3.1 8b | SFT | [llama8b_sft.yaml](/configs/lema/jobs/polaris/llama8b_sft.yaml) | Polaris | ✨ |
| Llama3.1 70b | LoRA | [llama70b_lora.yaml](/configs/lema/jobs/polaris/llama70b_lora.yaml) | Polaris | ✨ |
| Llama3.1 70b | SFT | [llama70b_sft.yaml](/configs/lema/jobs/polaris/llama70b_sft.yaml) | Polaris | ✨ |
| **Example Models** | | | | |
| Aya | SFT | [llama3.8b.aya.sft.yaml](/configs/lema/llama3.8b.aya.sft.yaml) | GCP | ✨ |
| Zephyr |QLoRA | [zephyr.7b.qlora.yaml](/configs/lema/zephyr.7b/sft/qlora.yaml) | GCP | ✨ |
| ChatQA | SFT | [chatqa.stage1.yaml](/configs/lema/chatqa/chatqa.stage1.yaml) | GCP | ✨ |
| **Pre-training** | | | | |
| GPT-2 | Pre-training | [gpt2.pt.mac.yaml](/configs/lema/gpt2.pt.mac.yaml) | Mac (mps) | ✨ |
| Llama2 7b | Pre-training | [llama2b.pt.yaml](/configs/lema/llama2b.pt.yaml) | Polaris | ✨ |

## Tutorials

We provide several example notebooks to help you get started with LeMa. Here's a list of available notebooks:

| Notebook | Description |
|----------|-------------|
| [A Tour](/notebooks/LeMa%20-%20A%20Tour.ipynb) | A comprehensive tour of the LeMa repository and its features |
| [Finetuning Tutorial](/notebooks/LeMa%20-%20Finetuning%20Tutorial.ipynb) | Step-by-step guide on how to finetune models using LeMa |
| [Tuning Llama](/notebooks/LeMa%20-%20Tuning%20Llama.ipynb) | Detailed tutorial on tuning Llama models with LeMa |
| [Multinode Inference on Polaris](/notebooks/LeMa%20-%20Multinode%20Inference%20on%20Polaris.ipynb) | Guides you through running inference with trained models |
| [Datasets Tutorial](/notebooks/Lema%20-%20Datasets%20Tutorial.ipynb) | Explains how to work with datasets in LeMa |
| [Deploying a Job](/notebooks/LeMa%20-%20Deploying%20a%20Job.ipynb) | Instructions on how to deploy a training job using LeMa |

## Documentation

View our API documentation [here](https://learning-machines.ai/docs/latest/index.html).

Reach out to <matthew@learning-machines.ai> if you have problems with access.

## Contributing

Contributions are welcome! Please check the `CONTRIBUTING.md` file for guidelines on how to contribute to the project.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Troubleshooting

1. Pre-commit hook errors with vscode
   - When committing changes, you may encounter an error with pre-commit hooks related to missing imports.
   - To fix this, make sure to start your vscode instance after activating your conda environment.

     ```shell
     conda activate lema
     code .  # inside the LeMa directory
     ```
