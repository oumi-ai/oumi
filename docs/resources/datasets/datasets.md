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
custom_datasets
```

This guide will help you integrate datasets into your AI inference and training pipelines, whether you're using our pre-built datasets or creating custom ones.

### Working with Your Data

1. **Standard Format Datasets**
  If your data matches our [supported formats](/resources/datasets/data_formats), use it directly with minimal setup.

2. **Custom Processing Needs**
  For datasets requiring special handling, follow our [custom dataset guide](/resources/datasets/custom_datasets).

3. **Pre-built Datasets** We have a collection of pre-built, popular open source datasets for various training objectives and tasks. These are available for download and use in your training and inference pipelines, and can be used to complement your own datasets. See [Pre-built Datasets](#pre-built-datasets) for more information.


### Pre-Built Datasets

Our dataset collection covers various training objectives and tasks:

| Dataset Type | Key Features | Documentation |
|--------------|--------------|---------------|
| **Pretraining** | • Large-scale corpus training for foundational models<br>• Domain adaptation through continued pretraining<br>• Efficient sequence packing and streaming | [→ Pretraining guide](pretraining_datasets.md) |
| **Supervised Fine-Tuning (SFT)** | • Instruction-following datasets<br>• Conversation format support for chat models<br>• Task-specific fine-tuning capabilities | [→ SFT guide](sft_datasets.md) |
| **Preference** | • Human preference data for RLHF training<br>• Direct preference optimization (DPO) support<br>• Quality and alignment tuning | [→ Preference learning guide](preference_datasets.md) |
| **Vision-Language** | • Image-text pairs for multi-modal training<br>• Visual question answering datasets<br>• Image captioning collections | [→ Vision-language guide](vl_sft_datasets.md) |

## Quick Start

Let's begin with a simple example using the python API:

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

You can also build a mixture of datasets, to train on multiple datasets at once:

```python
from oumi.core.configs import DataParams, DatasetParams
from oumi.builders import build_dataset_mixture

config = DataParams(
    train=DatasetSplitParams(
        datasets=[
            DatasetParams(dataset_name="tatsu-lab/alpaca"),
            DatasetParams(dataset_name="databricks/dolly"),
        ],
        mixture_strategy="first_exhausted",
    )
)

dataset = build_dataset_mixture(
    config=config,
    split=DatasetSplit.TRAIN
)
```

Configuration can be done via YAML:

```yaml
training:
  data:
    train:
      datasets:
        - dataset_name: oumi/sft-basic
          split: train
          stream: true  # Enable for large datasets
      collator_name: text_with_padding
```

## Next Steps

Start with our pre-built datasets for common use cases, and move to custom implementations when you need more control over data processing and loading.

1. **New to Oumi Datasets?**
   - Start with our [Data Formats Guide](/resources/datasets/data_formats) to understand basic concepts and structures

2. **Using Existing Datasets?**
   - Explore the available [SFT Datasets](/resources/datasets/sft_datasets), [Pretraining Datasets](/resources/datasets/pretraining_datasets), [Preference Datasets](/resources/datasets/preference_datasets), and [Vision-Language Datasets](/resources/datasets/vl_sft_datasets)

3. **Building Datasets with Custom Processing?**
   - Follow our [Custom Dataset Guide](/resources/datasets/custom_datasets)
