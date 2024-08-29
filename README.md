# Learning Machines (LeMa)

LeMa is a learning machines modeling platform that allows you to build foundation models end-to-end including data curation/synthesis, pretraining, tuning, and evaluation.

- Easy-to-use interface for data preprocessing, model training, and evaluation.
- Support for various machine learning algorithms and techniques.
- Visualization tools for model analysis and interpretation.
- Integration with popular libraries and frameworks.

## Features

- [x] Easily run in a locally, jupyter notebook, vscode debugger, or remote cluster
- [x] Full finetuning using SFT, DPO
- [x] Flexible configuration system using YAML files and command-line arguments
- [x] Support for distributed training and multiple GPUs
- [x] Support for multiple cloud providers (GCP, AWS, Azure), and DOE ALCF Polaris
- [x] Easy-to-use interface for data preprocessing, model training, and evaluation
- [x] Extensible architecture allowing easy addition of new models, datasets, and evaluation metrics

Take a [tour of our repository](https://github.com/openlema/lema/blob/main/notebooks/LeMa%20-%20A%20Tour.ipynb) to learn more!

Ready to get started? Check out our [Getting Started Guide](https://learning-machines.ai/docs/latest/getting_started.html).

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
