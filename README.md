# Learning Machines (LeMa)

LeMa is a learning machines modeling platform that allows you to build foundation models end-to-end including data curation/synthesis, pretraining, tuning, and evaluation.

## Features

- [x] Easily run training and evaluation locally, in a jupyter notebook, vscode debugger, or remote cluster
- [x] Instruction finetuning support: full finetuning using SFT, DPO, LoRA, etc.
- [x] Support for distributed training and multiple GPUs
- [x] Support for multiple cloud providers (GCP, AWS, Azure), and DOE ALCF Polaris
- [x] Flexible configuration system using YAML files and command-line arguments
- [x] Easy-to-use interface for data preprocessing, model training, and evaluation
- [x] Extensible architecture allowing easy addition of new models, datasets, and evaluation metrics

## Getting Started

For an overview of the LeMa features and usage, checkout the [user guide](/USAGE.md) and the [hands on tour of the repository](/notebooks/LeMa%20-%20A%20Tour.ipynb).

### Quickstart

### Configurations

These configurations demonstrate how to set up and run full training for different model architectures using LeMa.

| Model | Type | Configuration | Cluster | Status |
|-------|------|---------------|---------|--------|
| **Llama Instruction Finetuning** | | | | |
| Llama3.1 8b | LoRA | [llama8b_lora.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/llama8b_lora.yaml) | Polaris | ✨ |
| Llama3.1 8b | SFT | [llama8b_sft.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/llama8b_sft.yaml) | Polaris | ✨ |
| Llama3.1 70b | LoRA | [llama70b_lora.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/llama70b_lora.yaml) | Polaris | ✨ |
| Llama3.1 70b | SFT | - | Polaris | 🚀 COMING SOON! |
| **Example Models** | | | | |
| Aya | Full Training | [aya_full.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/aya_full.yaml) | Polaris | ✨ |
| Zephir | Full Training | [zephir_full.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/zephir_full.yaml) | Polaris | ✨ |
| ChatQA | Full Training | [chatqa_full.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/chatqa_full.yaml) | Polaris | ✨ |
| **Pre-training** | | | | |
| GPT-2 | Full Pre-training | [gpt2_pretrain.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/gpt2_pretrain.yaml) | Polaris | ✨ |
| Llama2 7b | Full Pre-training | [llama2_7b_pretrain.yaml](https://github.com/openlema/lema/blob/main/configs/lema/jobs/polaris/llama2_7b_pretrain.yaml) | Polaris | ✨ |

## Tutorials

We provide several example notebooks to help you get started with LeMa. Here's a list of available notebooks:

| Notebook | Description |
|----------|-------------|
| [LeMa - A Tour](https://github.com/openlema/lema/blob/main/notebooks/LeMa%20-%20A%20Tour.ipynb) | A comprehensive tour of the LeMa repository and its features |
| [Data Processing](https://github.com/openlema/lema/blob/main/notebooks/Data%20Processing.ipynb) | Demonstrates how to load, preprocess, and prepare data for training |
| [Model Training](https://github.com/openlema/lema/blob/main/notebooks/Model%20Training.ipynb) | Shows how to configure and train a model using LeMa |
| [Evaluation](https://github.com/openlema/lema/blob/main/notebooks/Evaluation.ipynb) | Explains how to evaluate trained models using various metrics |
| [Inference](https://github.com/openlema/lema/blob/main/notebooks/Inference.ipynb) | Guides you through running inference with trained models |
| [Custom Model](https://github.com/openlema/lema/blob/main/notebooks/Custom%20Model.ipynb) | Demonstrates how to create and use custom model architectures |

## Main Components

1. **Data Processing**: Supports various dataset formats and provides utilities for data loading, preprocessing, and batching.

2. **Model Building**: Includes a flexible system for building and configuring models, with support for custom architectures.

3. **Training**: Offers a robust training pipeline with support for different optimizers, learning rate schedules, and training strategies.

4. **Evaluation**: Provides comprehensive evaluation tools, including support for multiple metrics and integration with the LM Evaluation Harness.

5. **Inference**: Allows for both batch and interactive inference using trained models.

6. **Logging and Monitoring**: Includes detailed logging and optional integration with wandb and TensorBoard for experiment tracking.

## Documentation

View our API documentation [here](https://learning-machines.ai/docs/latest/index.html).

Reach out to <matthew@learning-machines.ai> if you have problems with access.

## Getting Started

1. Install the package:

   ```shell
   pip install 'lema[cloud,dev,train]'
   ```

2. Set up your configuration file (example configs are provided in the `configs` directory).

3. Run training:

   ```shell
   python -m lema.train -c path/to/your/config.yaml
   ```

4. Evaluate your model:

   ```shell
   python -m lema.evaluate -c path/to/your/eval_config.yaml
   ```

5. Run inference:

   ```shell
   python -m lema.infer -c path/to/your/inference_config.yaml
   ```

## Advanced Usage

- **Distributed Training**: LeMa supports multi-GPU training using PyTorch's DistributedDataParallel (DDP) or Fully Sharded Data Parallel (FSDP).
- **Cloud Training**: Integrated with SkyPilot for easy deployment to cloud GPU clusters.
- **Custom Models and Datasets**: The framework is designed to be easily extensible with custom models and datasets.

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
