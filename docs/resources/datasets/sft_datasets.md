# Supervised Fine-Tuning

This guide covers datasets used for instruction tuning and supervised learning in Oumi.

## SFT Datasets

Out-of-the box, we support multiple popular SFT datasets:

```{include} /api/summary/sft_datasets.md
```

## Usage

### Configuration

To use a specific SFT dataset in your Oumi configuration, specify it in the {py:class}`~oumi.core.configs.TrainingConfig`.

Here's an example:

```yaml
training:
  data:
    train:
      datasets:
        - dataset_name: your_sft_dataset_name
          split: train
          stream: false
      collator_name: text_with_padding
```

In this configuration:

- {py:attr}`~oumi.core.configs.DatasetParams.dataset_name` specifies the name of your SFT dataset
- {py:attr}`~oumi.core.configs.DatasetParams.split` selects a specific dataset split (e.g., train, validation, test)
- {py:attr}`~oumi.core.configs.DatasetParams.stream` enables streaming mode for large datasets
- {py:attr}`~oumi.core.configs.DatasetSplitParams.collator_name` specifies the collator to use for batching

### Python API

To use a specific SFT dataset in your code, you can use the {py:func}`~oumi.builders.data.build_dataset` function:

```python
from oumi.builders import build_dataset
from oumi.core.configs import DatasetSplit
from torch.utils.data import DataLoader

# Assume you have your tokenizer initialized
tokenizer = ...

# Build the dataset
dataset = build_dataset(
    dataset_name="your_sft_dataset_name",
    tokenizer=tokenizer,
    dataset_split=DatasetSplit.TRAIN
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now you can use the dataset in your training loop
for batch in loader:
    # Process your batch
    ...
```

## Adding a New SFT Dataset

All SFT datasets in Oumi are subclasses of {py:class}`~oumi.core.datasets.BaseLMSftDataset`.

```{important}
The {py:class}`~oumi.core.datasets.BaseLMSftDataset` is an abstract base class. Implement your specific dataset by subclassing it and overriding the {py:meth}`~oumi.core.datasets.BaseLMSftDataset.transform_conversation` method.
```

To add a new SFT dataset:

1. Subclass {py:class}`~oumi.core.datasets.BaseLMSftDataset`
2. Implement the {py:meth}`~oumi.core.datasets.BaseLMSftDataset.transform_conversation` method to define the dataset-specific transformation logic.

For example:

```python
from oumi.core.datasets import BaseSftDataset
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.registry import register_dataset

@register_dataset("custom_sft_dataset")
class CustomSftDataset(BaseLMSftDataset):
    def __init__(self, config: TrainingConfig,
                 tokenizer: BaseTokenizer,
                 dataset_split: DatasetSplit):
        super().__init__(config, tokenizer, dataset_split)
        # Initialize your dataset here

    def transform_conversation(self, example: Dict[str, Any]) -> Conversation:
        # Transform the raw example into a Conversation object
        # 'example' represents one row of the raw dataset
        # Structure of 'example':
        # {
        #     'input': str,  # The user's input or question
        #     'output': str  # The assistant's response
        # }
        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content=example['input']),
                Message(role=Role.ASSISTANT, content=example['output'])
            ]
        )

        return conversation
```

```{tip}
For more advanced SFT dataset implementations, explore the `oumi.datasets` module, which contains implementations of several [open source datasets](https://github.com/oumi-ai/oumi/tree/main/src/oumi/datasets).
```
