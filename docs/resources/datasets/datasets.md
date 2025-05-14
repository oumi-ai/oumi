# Datasets

```{toctree}
:maxdepth: 2
:caption: Datasets
:hidden:

data_formats
sft_datasets
pretraining_datasets
preference_datasets
vl_sft_datasets
other_datasets
```

## Overview

Oumi provides a dataset framework designed to handle everything from small custom datasets to web-scale pre-training datasets. Our goal is to make it easy to work with any type of data while maintaining consistent interfaces and optimal performance.

Key features include:

- **Multiple Dataset Types**: Support for Supervised Fine-Tuning (SFT), Pre-training, Preference Tuning, Vision-Language datasets, and more.
- **Flexible Data Formats**: Work with standard formats like ChatML, or implement custom data processing
- **Scalable Processing**: Handle datasets of any size through streaming and efficient data loading
- **Pre-built Datasets**: Access to a curated collection of popular open-source datasets ready for immediate use

## Quick Start

### Using Pre-built Datasets

The fastest way to get started is using one of our pre-built datasets. These datasets are ready to use and require minimal setup. You can load them directly using either the Python API or configure them through YAML files.

::::{tab-set}
:::{tab-item} YAML Config

```yaml
data:
  train:
    datasets:
      - dataset_name: tatsu-lab/alpaca
        split: train
```

:::

:::{tab-item} Python API

```python
from oumi.builders import build_dataset
from oumi.core.configs import DatasetSplit

# Load a pre-built dataset
dataset = build_dataset(
    dataset_name="tatsu-lab/alpaca"
)

# Access the training sample at index 0
print(dataset[0])

# Use in your training loop
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in dataloader:
    # Your training code here
    pass
```

:::
::::

### Working with Dataset Mixtures

For more complex training scenarios, you might want to combine multiple datasets. Oumi makes it easy to create and configure dataset mixtures, allowing you to train on multiple datasets simultaneously with configurable mixing strategies.

::::{tab-set}
:::{tab-item} YAML Config

```yaml
data:
  train:
    datasets:
      - dataset_name: tatsu-lab/alpaca
        split: train
      - dataset_name: databricks/dolly
        split: train
    mixture_strategy: first_exhausted  # Strategy for combining multiple datasets
    collator_name: text_with_padding
```

:::

:::{tab-item} Python API

```python
from oumi.builders import build_dataset_mixture
from oumi.core.configs import DataParams, DatasetParams, DatasetSplit, DatasetSplitParams
from oumi.core.tokenizers import BaseTokenizer

tokenizer: BaseTokenizer = ...
# Build a mixture of datasets
data_params = DataParams(
    train=DatasetSplitParams(
        datasets=[
            DatasetParams(dataset_name="tatsu-lab/alpaca"),
            DatasetParams(dataset_name="databricks/dolly"),
        ],
        mixture_strategy="first_exhausted",
    )
)

dataset = build_dataset_mixture(
    data_params=data_params,
    tokenizer=tokenizer,
    dataset_split=DatasetSplit.TRAIN
)
```

:::
::::

## Core Concepts

### How Datasets Work

At its core, each dataset in Oumi consists of two main components:

1. **Dataset Class**: Specified by `dataset_name`, this defines how the data should be processed. Dataset classes are registered in the codebase and map to specific Python classes. For example:
   - `"tatsu-lab/alpaca"` maps to the {py:class}`~oumi.datasets.sft.alpaca.AlpacaDataset` class, which handles JSON Lines data in Alpaca format
   - `"text_sft"` maps to the {py:class}`~oumi.datasets.sft.sft_jsonlines.TextSftJsonLinesDataset` class, which handles generic SFT data
   - Each class knows how to parse its input format and convert examples into the right format for training

2. **Dataset Path**: Specified by `dataset_path`, this points to where the actual data is stored. It can be:
   - A local file path (e.g., `"data/my_dataset.jsonl"`), or path to a cached dataset
   - Left empty to use the default data source for pre-built datasets

Here are two examples showing how this works in configuration:

```yaml
data:
  train:
    datasets:
      # Example 1: Using pre-built Alpaca dataset
      - dataset_name: tatsu-lab/alpaca  # Uses AlpacaDataset class
        # No dataset_path needed - will use default Alpaca data from
        # https://huggingface.co/tatsu-lab/alpaca

      # Example 2: Using custom data with text_sft format
      - dataset_name: text_sft  # Uses TextSFTDataset class
        dataset_path: path/to/my_custom_data.jsonl  # Your data in text_sft format
```

This separation between the dataset class and data source makes it easy to:

- Use the same processing logic with different data sources.
  - For example, the {py:class}`~oumi.datasets.sft.alpaca.AlpacaDataset` class can be used with both the default Alpaca data (`"tatsu-lab/alpaca"`), or one of the cleaned versions (`yahma/alpaca-cleaned`), or any other file that follows the same format.
- Apply consistent formatting across your own datasets
- Mix and match different dataset types in training

### Dataset Types

Our dataset collection covers various training objectives and tasks:

| Dataset Type | Key Features | Documentation |
|--------------|--------------|---------------|
| **Pretraining** | • Large-scale corpus training for foundational models<br>• Efficient sequence packing and streaming | [→ Pretraining guide](pretraining_datasets.md) |
| **Supervised Fine-Tuning (SFT)** | • Instruction-following datasets<br>• Conversation format support for chat models<br>• Task-specific fine-tuning capabilities | [→ SFT guide](sft_datasets.md) |
| **Preference Tuning** | • Human preference data for RLHF or DPOtraining | [→ Preference learning guide](preference_datasets.md) |
| **Vision-Language** | • Image-text pairs for multi-modal training <br>• Conversation format support for chat models| [→ Vision-language guide](vl_sft_datasets.md) |

It's also possible to define custom datasets for new types of data not covered above. See [→ Other Datasets](other_datasets.md).

## Next Steps

1. **New to Oumi Datasets?**
   - Start with our [Data Formats Guide](/resources/datasets/data_formats) to understand basic concepts and structures

2. **Using Existing Datasets?**
   - Explore the available [SFT Datasets](/resources/datasets/sft_datasets), [Pretraining Datasets](/resources/datasets/pretraining_datasets), [Preference Datasets](/resources/datasets/preference_datasets), and [Vision-Language Datasets](/resources/datasets/vl_sft_datasets)
