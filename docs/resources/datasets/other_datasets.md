(other-datasets)=
# Other Datasets

In addition to the common LLM dataset formats ([Pretraining](pretraining_datasets.md), [SFT](sft_datasets.md), [VL-SFT](vl_sft_datasets.md), etc),
Oumi infrastructure also allows users to define arbitrary ad-hoc dataset formats,
which can be used not just for text-centric LLM models, but for alternative model types
and applications such as Vison models (e.g., convolutional networks), scientific computing, etc.

This can be accomplished by defining a subclass of {py:class}`~oumi.core.datasets.BaseMapDataset` or of {py:class}`~oumi.core.datasets.BaseIterableDataset`.

## NumPy Dataset

The popular `numpy` library defines `.npy` and `.npz` file formats [[details](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html)],
which can be used to [save](https://numpy.org/doc/2.1/reference/generated/numpy.save.html) arbitrary multi-dimensional arrays ([`np.ndarray`](https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.html)):

1. `.npy` file contains a single `np.ndarray`.
2. `.npz` is an archive that contains a collection of multiple `np.ndarray`-s, with optional support for [data compression](https://numpy.org/doc/2.1/reference/generated/numpy.savez_compressed.html).


### Adding a New Numpy (.npz) Dataset

To add a new dataset that can load data from `.npz` files, follow these steps:

1. Subclass {py:class}`~oumi.core.datasets.BaseMapDataset`
2. Implement the {py:meth}`~oumi.core.datasets.BaseMapDataset.__init__`, {py:meth}`~oumi.core.datasets.BaseMapDataset._load_data`, {py:meth}`~oumi.core.datasets.BaseMapDataset.transform` methods to handle initialization, data loading, and data transforms respectively.

Here's a basic example, which shows how to do that:

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
