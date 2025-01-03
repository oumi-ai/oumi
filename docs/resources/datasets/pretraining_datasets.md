# Pretraining

This guide covers pretraining datasets used for training language models from scratch or continuing pretraining in Oumi.

## Supported Datasets

Out of the box, we support multiple popular pretraining datasets:

```{include} /api/summary/pretraining_datasets.md
```

## Usage

### Configuration

To use a specific pretraining dataset in your Oumi configuration, you need to specify it in the {py:class}`~oumi.core.configs.TrainingConfig`. Here's an example of how to configure a pretraining dataset:

```yaml
training:
  data:
    train:
      datasets:
        - dataset_name: your_pretraining_dataset
          subset: optional_subset
          split: train
          stream: true  # Recommended for large datasets
          pack: true    # Enable sequence packing
          dataset_kwargs:
            seq_length: 4096  # packing sequence length
          stream: true  # Recommended for large datasets
          pack: true    # Enable sequence packing
```

In this configuration:

- {py:attr}`~oumi.core.configs.DatasetParams.dataset_name` specifies the name of your dataset
- {py:attr}`~oumi.core.configs.DatasetParams.subset` and {py:attr}`~oumi.core.configs.DatasetParams.split` allow you to select a specific dataset split (e.g. train, validation, test) or a dataset subset (if defined by the dataset)
- {py:attr}`~oumi.core.configs.DatasetParams.stream` enables streaming mode, which is essential for large datasets
- {py:attr}`~oumi.core.configs.DatasetParams.pack` activates sequence packing
- {py:attr}`~oumi.core.configs.DatasetParams.dataset_kwargs` allows you to pass additional parameters specific to your dataset

```{important}
Setting {py:attr}`~oumi.core.configs.DatasetSplitParams.experimental_use_async_dataset` to `True` enables the use of {py:class}`~oumi.datasets.pretraining_async_text_dataset.PretrainingAsyncTextDataset`, which is optimized for asynchronous data processing.
```

### Python API

To use a specific pretraining dataset in your code, you can leverage the {py:func}`~oumi.builders.data.build_dataset_mixture` function. Here's an example:

```python
from oumi.builders import build_dataset_mixture
from oumi.core.configs import TrainingConfig, DatasetSplit
from oumi.core.tokenizers import BaseTokenizer

# Assume you have your config and tokenizer initialized
config: TrainingConfig = ...
tokenizer: BaseTokenizer = ...

# Build the dataset
dataset = build_dataset_mixture(
    config=config,
    tokenizer=tokenizer,
    dataset_split=DatasetSplit.TRAIN
)

# Now you can use the dataset in your training loop
for batch in dataset:
    # Process your batch
    ...
```

The {py:func}`~oumi.builders.data.build_dataset_mixture` function takes care of creating the appropriate dataset based on your configuration. It handles the complexities of dataset initialization, including:

- Applying the correct tokenizer
- Setting up streaming if enabled
- Configuring sequence packing if specified
- Handling dataset mixtures if multiple datasets are defined

## Adding a New Pretraining Dataset

All pretraining datasets in Oumi are subclasses of {py:class}`~oumi.core.datasets.iterable_dataset.BasePretrainingIterableDataset`.

This class extends {py:class}`~oumi.core.datasets.iterable_dataset.BaseIterableDataset` to offer functionality specific to pretraining tasks.

```{note}
The {py:class}`~oumi.core.datasets.iterable_dataset.BasePretrainingIterableDataset` is an abstract base class. You should implement your specific dataset by subclassing it and overriding the {py:meth}`~oumi.core.datasets.iterable_dataset.BasePretrainingIterableDataset.transform` method.
```

To add a new pretraining dataset, you have to:

1. Subclass {py:class}`~oumi.core.datasets.iterable_dataset.BasePretrainingIterableDataset`
2. Implement the {py:meth}`~oumi.core.datasets.iterable_dataset.BasePretrainingIterableDataset.transform` method to define the dataset-specific transformation logic.

For example:

```python
from oumi.core.datasets import BasePretrainingDataset
from oumi.core.registry import register_dataset

@register_dataset("custom_pretraining_dataset")
class CustomPretrainingDataset(BasePretrainingDataset):
    """A custom pretraining dataset."""

    default_dataset = "custom_pretraining_name"

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Transform raw data for pretraining
        tokens = self.tokenizer(
            data["text"],
            max_length=self.max_length,
            truncation=True
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": tokens["input_ids"].copy()
        }
```
