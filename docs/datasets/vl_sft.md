# Vision-Language Supervised Fine-Tuning

## VL-SFT Datasets

```{include} ../api/summary/vl_sft_datasets.md
```

## Usage

### Using a Specific VL Dataset in Configuration

The configuration for VL-SFT datasets is similar to regular SFT datasets, with some additional parameters for image processing. Here's an example:

```yaml
training:
  data:
    train:
      datasets:
        - dataset_name: your_vl_sft_dataset_name
          split: train
          stream: false
      collator_name: vision_language_with_padding
```

### Using a Specific VL Dataset in Code

Using a VL-SFT dataset in code is similar to using a regular SFT dataset, with the main difference being in the batch contents:

```python
from oumi.builders import build_dataset
from oumi.core.configs import DatasetSplit
from torch.utils.data import DataLoader

# Assume you have your tokenizer and image processor initialized
tokenizer = ...

# Build the dataset
dataset = build_dataset(
    dataset_name="your_vl_sft_dataset_name",
    tokenizer=tokenizer,
    dataset_split=DatasetSplit.TRAIN
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now you can use the dataset in your training loop
for batch in loader:
    # Process your batch
    # Note: batch will contain both text and image data
    ...
```

```{tip}
VL-SFT batches typically include additional keys for image data, such as `pixel_values` or `image_embeddings`, depending on the specific dataset and model architecture.
```

## Custom VL-SFT Datasets

### VisionLanguageSftDataset Base Class

All VL-SFT datasets in Oumi are subclasses of {py:class}`~oumi.core.datasets.VisionLanguageSftDataset`. This class extends the functionality of {py:class}`~oumi.core.datasets.BaseLMSftDataset` to handle image data alongside text.

### Adding a New VL-SFT Dataset

To add a new VL-SFT dataset, follow these steps:

1. Subclass {py:class}`~oumi.core.datasets.VisionLanguageSftDataset`
2. Implement the {py:meth}`~oumi.core.datasets.VisionLanguageSftDataset.transform_conversation` method to handle both text and image data.

Here's a basic example:

```python
from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.types.turn import ContentItem, Conversation, Message, Role, Type

class MyVLSftDataset(VisionLanguageSftDataset):
    def transform_conversation(self, example: Dict[str, Any]) -> Conversation:
        # Transform the raw example into a Conversation object
        # 'example' represents one row of the raw dataset
        # Structure of 'example':
        # {
        #     'image_path': str,  # Path to the image file
        #     'question': str,    # The user's question about the image
        #     'answer': str       # The assistant's response
        # }
        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content=[
                    ContentItem(type=Type.IMAGE_PATH, content=example['image_path']),
                    ContentItem(type=Type.TEXT, content=example['question']),
                ]),
                Message(role=Role.ASSISTANT, content=example['answer'])
            ]
        )

        return conversation
```

```{note}
The key difference in VL-SFT datasets is the inclusion of image data, typically represented as an additional `ContentItem` with `type=Type.IMAGE_BINARY`, `type=Type.IMAGE_PATH` or `Type.IMAGE_URL`.
```

For more advanced VL-SFT dataset implementations, explore the {py:mod}`oumi.datasets.vision_language` module.
