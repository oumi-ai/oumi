# LeMa Usage Overview

LeMa is a framework for training and evaluating large language models. This tutorial will guide you through the process of setting up, training, and evaluating a model using LeMa.

## 1. Installation

First, you'll need to install LeMa and its dependencies:

```bash
git clone https://github.com/openlema/lema.git
cd lema
pip install -e "[dev,train]"
```

## 2. Configuration

LeMa uses configuration files to specify model and training parameters. Create a YAML file (e.g., `config.yaml`) with your desired settings. Here's a basic example:

```yaml
model:
  model_name: "gpt2"

data:
  train:
    datasets:
      - dataset_name: "tatsu-lab/alpaca"
        split: "train"
  validation:
    datasets:
      - dataset_name: "tatsu-lab/alpaca"
        split: "validation"

training:
  output_dir: "output"
  per_device_train_batch_size: 8
  num_train_epochs: 3
  learning_rate: 5e-5
```

For more advanced use cases, and examples of other configuration options, see the [Configuration](https://github.com/openlema/lema/tree/main/configs/lema) directory.

## 3. Training

To train a model using LeMa, use the `train` script:

```bash
lema-train -c config.yaml

# Or, if the script is not in your PATH:
python -m lema.train -c config.yaml

# On the other hand, this does not work!
python src/lema/train.py -c config.yaml
```

This will start the training process using the configuration specified in `config.yaml`. The script will:

1. Download the specified model and tokenizer (if not already cached)
2. Download the datasets (if not already cached)
3. Initialize the trainer
4. Start the training process

You can monitor the training progress in the console output. Checkpoints and logs will be saved in the specified `output_dir`.

## 4. Evaluation

After training, you can evaluate your model using the `evaluate` script:

```bash
lema-evaluate -c eval_config.yaml

# Alternatively:
python -m lema.evaluate -c eval_config.yaml
```

Create an `eval_config.yaml` file with evaluation-specific settings:

```yaml
model:
  model_name: "output/checkpoint-1000"  # Path to your trained model

data:
  datasets:
    - dataset_name: "cais/mmlu"
      split: "test"

evaluation_framework: "lm_harness"
num_shots: 5
output_dir: "eval_results"
```

This will evaluate your model on the specified dataset(s) and save the results in the `eval_results` directory.

## 5. Inference

To run inference on your trained model, use the `infer` script:

```bash
lema-infer -c infer_config.yaml

# Alternatively:
python -m lema.infer -c infer_config.yaml
```

Create an `infer_config.yaml` file with inference settings:

```yaml
model:
  model_name: "output/checkpoint-1000"  # Path to your trained model

generation:
  max_new_tokens: 100
  batch_size: 1
```

You can also use the `-i` flag for interactive mode:

```bash
lema-infer -c infer_config.yaml -i
```

This will allow you to input prompts and get responses from your model interactively.

## 6. Custom Datasets

LeMa supports custom datasets. To use your own SFT dataset, create a new class that inherits from `BaseLMSftDataset` and implement the required methods. Then, register your dataset using the `@register_dataset` decorator:

```python
from lema.core.datasets.base_dataset import BaseLMSftDataset
from lema.core.registry import register_dataset

@register_dataset("my_custom_dataset")
class MyCustomDataset(BaseLMSftDataset):
    def transform_conversation(self, raw_example):
        # Implement your data transformation logic here
        pass
```

You can then use your custom dataset in the configuration file:

```yaml
data:
  train:
    datasets:
      - dataset_name: "my_custom_dataset"
        split: "train"
```

For more details, see this notebook [Custom Datasets](https://github.com/openlema/lema/blob/main/notebooks/Lema%20-%20Datasets%20Tutorial.ipynb). You can also find the list of datasets already implemented in lema [here](https://github.com/openlema/lema/tree/main/src/lema/datasets).

## 7. Multi-GPU Training

LeMa supports distributed training. To use multiple GPUs, you can use the `torch.distributed.launch` module:

```bash
torchrun --standalone --nproc_per_node=4 -m lema.train -c config.yaml
```

This will launch the training script on 4 GPUs.

## 8.  Distributed Training

To scale up to multiple nodes, or to use GPUs on a remote cluster, you can use the `lema-launcher`, which makes it straightforward to run jobs on remote machines.

You can find a detailled example here: [notebook](https://github.com/openlema/lema/blob/main/notebooks/LeMa%20-%20Running%20Jobs%20Remotely.ipynb)

## 9. Monitoring and Logging

LeMa supports Weights & Biases (wandb) and TensorBoard for logging.

To enable wandb logging, set `enable_wandb: true` in your config file.  logging is enabled by default.
To enable tensorboard logging, set `enable_tensorboard: true` in your config file. TensorBoard logging is enabled by default.

You can view TensorBoard logs by running:

```bash
tensorboard --logdir output/runs
```
