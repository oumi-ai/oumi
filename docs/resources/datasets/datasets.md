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

## Quick Start Guide

Let's begin with a simple example:

```python
from oumi.builders import build_dataset
from oumi.core.configs import DatasetSplit

# Load a pre-built dataset
dataset = build_dataset(
    dataset_name="oumi/sft-basic",
    split=DatasetSplit.TRAIN
)

# Use in your training loop
for batch in dataset:
    # Your training code here
    pass
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
   - Start with our [Data Formats Guide](/resources/datasets/data_formats)
   - Understand basic concepts and structures
   - Try working with a pre-built dataset

2. **Using Existing Datasets?**
   - Explore the available [SFT Datasets](/resources/datasets/sft_datasets), [Pretraining Datasets](/resources/datasets/pretraining_datasets), and [Preference Datasets](/resources/datasets/preference_datasets)
   - Check out [Vision-Language Datasets](/resources/datasets/vl_sft_datasets)
   - Review performance optimization tips

3. **Building Custom Datasets?**
   - Follow our [Custom Dataset Guide](/resources/datasets/custom_datasets)
   - Understand the base classes
   - Learn about optimization strategies

## Support and Resources

- Check our example notebooks in the `examples/` directory
- Visit our [GitHub repository](https://github.com/oumi-ai/oumi) for updates
- Join our community discussions for help and tips
