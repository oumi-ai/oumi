# Vision-Language Datasets

Vision-Language Supervised Finetuning (VL-SFT) extends the concept of Supervised Fine-Tuning (SFT) to handle both images and text. This enables the model to understand and reason about visual information, opening up a wide range of multimodal applications.

This guide covers Vision-Language datasets used for instruction tuning and supervised learning in Oumi.

(vl-sft-datasets)=
## VL-SFT Datasets

```{include} /api/summary/vl_sft_datasets.md
```

## Usage

### Configuration

The configuration for VL-SFT datasets is similar to regular SFT datasets, with some additional parameters for image processing. Here's an example:

```yaml
training:
  data:
    train:
      collator_name: vision_language_with_padding
      collator_kwargs: {}  # Optional: Additional collator parameters
      datasets:
        - dataset_name: "your_vl_sft_dataset_name"
          split: "train"
          trust_remote_code: False # Set to true if needed for model-specific processors
          transform_num_workers: "auto"
          dataset_kwargs:
            processor_name: "meta-llama/Llama-3.2-11B-Vision-Instruct" # Model-specific processor
            return_tensors: True
```
In this configuration:

- `dataset_name`: Name of the vision-language dataset
- `collator_kwargs`: Optional additional parameters for the collator (e.g., `allow_multi_image_inputs: false`)
- `trust_remote_code`: Enable for model-specific processors that use downloaded scripts
- `transform_num_workers`: Number of workers for image processing
- `processor_name`: Vision model processor to use

### Python API

Using a VL-SFT dataset in code is similar to using a regular SFT dataset, with the main difference being in the batch contents:

```python
from oumi.builders import build_dataset, build_processor, build_tokenizer
from oumi.core.configs import DatasetSplit, ModelParams
from torch.utils.data import DataLoader

# Assume you have your tokenizer and image processor initialized
model_params: ModelParams = ...
trust_remote_code: bool = False # `True` if model-specific processor requires it
tokenizer: BaseTokenizer = build_tokenizer(model_params)
processor: BaseProcessor = build_processor(
        model_params.model_name, tokenizer, trust_remote_code=trust_remote_code
)

# Build the dataset
dataset = build_dataset(
    dataset_name="your_vl_sft_dataset_name",
    tokenizer=tokenizer,
    split=DatasetSplit.TRAIN,
    dataset_kwargs=dict(processor=processor),
    trust_remote_code=trust_remote_code,
)

# Create dataloader
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Now you can use the dataset in your training loop
for batch in loader:
    # Process your batch
    # Note: batch will contain both text and image data
    ...
```

### Batch Contents

Vision-language batches typically include:

- `input_ids`: Text token IDs
- `attention_mask`: Text attention mask
- `pixel_values`: Processed image tensors
- `image_attention_mask`: Image attention mask
- Additional model-specific keys

```{tip}
VL-SFT batches typically include additional keys for image data, such as `pixel_values` or `cross_attention_mask`, depending on the specific dataset and model architecture.
```

## Custom VL-SFT Datasets

### VisionLanguageSftDataset Base Class

All VL-SFT datasets in Oumi are subclasses of {py:class}`~oumi.core.datasets.VisionLanguageSftDataset`. This class extends the functionality of {py:class}`~oumi.core.datasets.BaseSftDataset` to handle image data alongside text.

### Adding a New VL-SFT Dataset

To add a new VL-SFT dataset, follow these steps:

1. Subclass {py:class}`~oumi.core.datasets.VisionLanguageSftDataset`
2. Implement the {py:meth}`~oumi.core.datasets.VisionLanguageSftDataset.transform_conversation` method to handle both text and image data.

Here's a basic example, which loads data from the hypothetical `example/foo` HuggingFace dataset (image + text),
and formats the data as Oumi `Conversation`-s for SFT tuning:

```python
from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type

@register_dataset("your_vl_sft_dataset_name")
class CustomVLDataset(VisionLanguageSftDataset):
    """Dataset class for the `example/foo` dataset."""
    default_dataset = "example/foo" # Name of the original HuggingFace dataset (image + text)

    def transform_conversation(self, example: Dict[str, Any]) -> Conversation:
        """Transform raw data into a conversation with images."""
        # Transform the raw example into a Conversation object
        # 'example' represents one row of the raw dataset
        # Structure of 'example':
        # {
        #     'image_bytes': bytes,  # PNG bytes of the image
        #     'question': str,       # The user's question about the image
        #     'answer': str          # The assistant's response
        # }
        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content=[
                    ContentItem(type=Type.IMAGE_BINARY, binary=example['image_bytes']),
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

### Using Custom Datasets via the CLI

See {doc}`/user_guides/customization` to quickly enable your dataset when using the CLI.
