# Configuration

## 1. Configuration

Oumi uses YAML configuration files to specify model and training parameters. Create a file named `config.yaml` with your desired settings. Here's a basic example:

```yaml
model:
  model_name: "gpt2"  # Specifies the model architecture to use

data:
  train:
    datasets:
      - dataset_name: "tatsu-lab/alpaca"  # Training dataset
        split: "train"
  validation:
    datasets:
      - dataset_name: "tatsu-lab/alpaca"  # Validation dataset
        split: "validation"

training:
  output_dir: "output/quickstart"  # Directory to save outputs
  per_device_train_batch_size: 8  # Batch size per GPU
  num_train_epochs: 3  # Number of training epochs
  learning_rate: 5e-5  # Learning rate for optimization
```

Each section in the configuration file controls different aspects of the training process:

- `model`: Specifies the model architecture and related parameters
- `data`: Defines the datasets for training and validation
- `training`: Sets training hyperparameters and output locations

## 2. Model Selection

Oumi supports various model architectures. To use a different model, simply change the `model_name` in your configuration:

```yaml
model:
  model_name: "bert-base-uncased"  # Uses BERT instead of GPT-2
```

Popular choices include:

- `gpt2`: For general language generation tasks
- `bert-base-uncased`: For classification or token-level tasks
- `t5-small`: For sequence-to-sequence tasks

The choice of model depends on your specific task and computational resources.

## 3. Data Preprocessing

Oumi handles most data preprocessing automatically. However, for custom datasets or specific requirements, you can implement your own preprocessing logic. Here's an example of how to preprocess data for a text classification task:

```python
from oumi.core.datasets import BaseSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role

@register_dataset("my_text_classification_dataset")
class MyTextClassificationDataset(BaseSftDataset):
    def transform_conversation(self, raw_example):
        text = raw_example['text']
        label = raw_example['label']

        # Create a conversation with user input and expected output
        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content=f"Classify the following text: {text}"),
                Message(role=Role.ASSISTANT, content=f"The classification is: {label}")
            ]
        )
        return conversation
```

Then, use your custom dataset in the configuration:

```yaml
data:
  train:
    datasets:
      - dataset_name: "my_text_classification_dataset"
        split: "train"
```

## 4. Training

To train a model using Oumi, use the `train` script:

```bash
oumi train -c config.yaml
```

This will:

1. Download the specified model and tokenizer
2. Load and preprocess the datasets
3. Initialize the trainer
4. Start the training process

Monitor the training progress in the console output. Checkpoints and logs will be saved in the specified `output_dir`.

To resume training from a checkpoint, add the following to your configuration:

```yaml
training:
  resume_from_checkpoint: "output/quickstart/checkpoint-1000"
```

## 5. Evaluation

After training, evaluate your model using the `evaluate` script:

```bash
oumi evaluate -c eval_config.yaml
```

Create an `eval_config.yaml` file with evaluation-specific settings:

```yaml
model:
  model_name: "output/quickstart/checkpoint-1000"  # Path to your trained model

lm_harness_params:
  tasks:
    - "mmlu"  # Multiple-choice grade-school tasks
output_dir: "output/quickstart/eval_results"
```

This evaluates your model on the specified tasks and saves the results in `output/quickstart/eval_results`. Common metrics include accuracy, perplexity, and F1 score, depending on the task.

## 6. Inference

To run inference on your trained model:

```bash
oumi infer -c infer_config.yaml
```

Create an `infer_config.yaml` file with inference settings:

```yaml
model:
  model_name: "output/quickstart/checkpoint-1000"  # Path to your trained model

generation:
  max_new_tokens: 100  # Maximum number of tokens to generate
  batch_size: 1  # Batch size for inference
```

This allows you to input prompts and get responses from your model interactively.

## 7. Multi-GPU and Distributed Training

For multi-GPU training on a single machine:

```bash
torchrun --standalone --nproc-per-node=4 -m oumi train -c config.yaml
```

For distributed training across multiple nodes, use `oumi launch`:

```bash
oumi launch -c launch_config.yaml
```

See the [Running Jobs Remotely](https://github.com/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Running%20Jobs%20Remotely.ipynb) notebook for detailed examples.

## 8. Monitoring and Logging

Oumi supports Weights & Biases (wandb) and TensorBoard for logging.

To enable wandb logging:

```yaml
training:
  enable_wandb: true
```

TensorBoard logging is enabled by default. View logs with:

```bash
tensorboard --logdir output/quickstart/tensorboard
```
