# Training

```{toctree}
:maxdepth: 2
:caption: Training
:hidden:

finetuning
training_config
trainers
```

## Training Process

1. **Model Selection**: Choose from a variety of pre-trained models or define your own.

   ```{seealso}
   Explore available model recipes in the {doc}`Model Recipes <../../models/recipes>` page.
   ```

2. **Dataset Preparation**: Prepare your dataset for training. Oumi supports various dataset formats and provides tools for custom dataset creation.

   ```{admonition} Dataset Options
   :class: note

   - **Local dataset**: Load your own data in supported formats. See {doc}`Local Datasets <../../datasets/local_datasets>` for details.
   - **Existing dataset classes**: Utilize pre-defined dataset classes for common tasks:
     - Supervised fine-tuning (SFT): {doc}`SFT Datasets <../../datasets/sft>`
     - Vision-Language SFT: {doc}`VL-SFT Datasets <../../datasets/vl_sft>`
     - Pre-training: {doc}`Pretraining Datasets <../../datasets/pretraining>`
     - Preference tuning: {doc}`Preference Tuning Datasets <../../datasets/preference_tuning>`
   - **Custom dataset**: Create a custom dataset class for specific needs. Learn how in {doc}`Custom Datasets <../../advanced/custom_datasets>`.
   ```

3. **Training Configuration**: Set up your training parameters using YAML configuration files.

   ```{tip}
   For detailed information on configuration options, refer to the {doc}`Training Configuration <training_config>` page.
   ```

4. **Trainer Selection**: Oumi offers different trainers for various training scenarios.

   ```{seealso}
   Explore available trainers and their use cases in the {doc}`Trainers <trainers>` page.
   ```

5. **Distributed Training**: For large-scale training, Oumi supports distributed training across multiple GPUs and nodes.

   ```{note}
   Learn more about distributed training setups in the {doc}`Distributed Training <../../advanced/distributed_training>` guide.
   ```

## Running a Training Job

1. **Preparing Configuration**: Create or modify a YAML configuration file with your desired training settings.

   ```{code-block} yaml
   model:
     model_name: "gpt2"

   data:
     train:
       datasets:
         - dataset_name: "tatsu-lab/alpaca"
           split: "train"

   training:
     output_dir: "output/my_training_run"
     num_train_epochs: 3
     learning_rate: 5e-5
   ```

2. **Starting Training**: Use the Oumi CLI to start the training process:

   ```{code-block} bash
   oumi train -c path/to/your/config.yaml
   ```

   ```{tip}
   You can override configuration parameters directly from the command line:

   ```{code-block} bash
   oumi train -c path/to/your/config.yaml --training.learning_rate 1e-4
   ```

3. **Monitoring Progress**: Track the training progress using TensorBoard or Weights & Biases (if configured).

   ```{code-block} bash
   tensorboard --logdir output/my_training_run/tensorboard
   ```

4. **Next Steps**: After training, we can:
    - **Evaluate** the model's performance using the `oumi evaluate` command. See the {doc}`Evaluation Guide <../evaluate/evaluate>` for details.
    - **Run inference** with the `oumi infer` command. Learn more on the {doc}`Inference Guide <../infer/infer>` page.
    - **Judge** your model's outputs using the `oumi judge` command. Refer to the {doc}`Judge Guide <../judge/judge>` for guidance.

## Advanced Topics

- {doc}`finetuning`: Learn about fine-tuning techniques and best practices.
- {doc}`llama`: Specific guide for training LLaMA models.
- {doc}`../../advanced/performance_optimization`: Tips for optimizing training performance.
- {doc}`../../advanced/custom_models`: Guide on implementing custom model architectures.

## Troubleshooting

If you encounter issues during training, check the {doc}`Troubleshooting Guide <../../faq/troubleshooting>` for common problems and solutions.

For more help, don't hesitate to open an issue on the [Oumi GitHub repository](https://github.com/oumi-ai/oumi/issues).
