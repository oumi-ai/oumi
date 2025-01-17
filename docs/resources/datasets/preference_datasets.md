# Preference Tuning

Preference tuning is a technique for training language models to align with human preferences. It involves presenting the model with pairs of outputs (e.g., two different responses to the same prompt) and training it to prefer the output that humans prefer.

This guide covers datasets used for RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) in Oumi.

## Supported Datasets

```{include} /api/summary/preference_tuning_datasets.md
```

## Configuration

```yaml
training:
  data:
    train:
      datasets:
        - dataset_name: your_preference_dataset
          split: train
          dataset_kwargs:
            max_length: 512
```

## Usage Example

```python
from oumi.builders import build_dataset
from oumi.core.configs import DatasetSplit

# Build the dataset
dataset = build_dataset(
    dataset_name="your_preference_dataset",
    tokenizer=tokenizer,
    dataset_split=DatasetSplit.TRAIN
)

# Use in DPO training
for batch in dataset:
    # batch contains 'prompt', 'chosen', and 'rejected' responses
    ...
```

## Creating Custom Preference Dataset

```python
from oumi.core.datasets import BaseExperimentalDpoDataset
from oumi.core.registry import register_dataset

@register_dataset("custom_dpo_dataset")
class CustomDpoDataset(BaseExperimentalDpoDataset):
    """A custom DPO dataset."""

    default_dataset = "custom_dpo_name"

    def transform_preference(self, example: dict) -> dict:
        return {
            "prompt": example["instruction"],
            "chosen": example["better_response"],
            "rejected": example["worse_response"]
        }
```

### Using Custom Datasets via the CLI

See {doc}`/user_guides/customization` to quickly enable your dataset when using the CLI.
