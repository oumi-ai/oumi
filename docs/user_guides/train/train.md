# Training

```{toctree}
:maxdepth: 2
:caption: Training
:hidden:

finetuning
llama
training_config
trainers
distributed_training
```

## Overview

1. **Model Selection**: Choose from a variety of pre-trained models or define your own. See the {doc}`../../models/recipes` page for available model recipes.

2. **Dataset**: Prepare your dataset for training. Oumi supports various dataset formats and provides tools for custom dataset creation. You have several options:

   - Local dataset: Load your own data in supported formats. See {doc}`../../datasets/local_datasets` for details.
   - Existing dataset classes: Utilize pre-defined dataset classes for common tasks:
      - Supervised fine-tuning (SFT): {doc}`../../datasets/sft`
      - Vision-Language SFT: {doc}`../../datasets/vl_sft`
      - Pre-training: {doc}`../../datasets/pretraining`
      - Preference tuning: {doc}`../../datasets/preference_tuning`
   - Define a new dataset adapter class: Create a custom dataset class for specific needs. Learn how in {doc}`../../advanced/custom_datasets`.

3. **Training Configuration**: Set up your training parameters using YAML configuration files. For details, refer to the {doc}`training_config` page.

4. **Trainers**: Oumi offers different trainers for various training scenarios. Explore available options in the {doc}`trainers` page.

5. **Distributed Training**: For large-scale training, Oumi supports distributed training across multiple GPUs and nodes. See the {doc}`distributed_training` page for more information.

## Training Process

1. **Preparing our Configuration**: Create or modify a YAML configuration file with your desired training settings.

2. **Training**: Use the Oumi CLI to start the training process:

   ```bash
   oumi train -c path/to/your/config.yaml
   ```

3. **Monitoring**: Track the training progress using TensorBoard or Weights & Biases (if configured).

4. **Next Steps**: After training, we can:
   - Evaluate the model's performance using the `oumi evaluate` command. See the {doc}`../evaluate/evaluate` page for details.
   - Run inference with `oumi infer` command. Learn more on the {doc}`../infer/infer` page.
   - Judge your model's outputs using the `oumi judge` command. Refer to the {doc}`../judge/judge` page for guidance.

## Advanced Topics

- {doc}`finetuning`: Learn about fine-tuning techniques and best practices.
- {doc}`llama`: Specific guide for training LLaMA models.
- {doc}`../../advanced/performance_optimization`: Tips for optimizing training performance.
- {doc}`../../advanced/custom_models`: Guide on implementing custom model architectures.

## Troubleshooting

If you encounter issues during training, check the {doc}`../../faq/troubleshooting` page for common problems and solutions.
