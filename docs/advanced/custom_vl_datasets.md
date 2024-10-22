# Custom Vision-Language Datasets

This tutorial will guide you through the process of implementing a new Vision-Language dataset. We'll create a generic dataset called "NewVisionLanguageDataset" that can be adapted for various vision-language tasks.

**Note:**
If your dataset is already in the `oumi` jsonl format, you can skip this step and simply use the `vision_language_jsonl` dataset.

## Step 1: Set Up the Dataset File

1. Create a new file in the `src/oumi/datasets/vision_language/` directory. Let's call it `new_vision_language_dataset.py`.

2. Import the necessary modules:

```python
from oumi.core.datasets import VisionLanguageSftDataset
from oumi.core.registry import register_dataset
from oumi.core.types.turn import Conversation, Message, Role, Type
```

## Step 2: Define the Dataset Class

1. Create a new class that inherits from `VisionLanguageSftDataset`:

    ```python
    @register_dataset("new_vision_language_dataset")
    class NewVisionLanguageDataset(VisionLanguageSftDataset):
        default_dataset = "your_org/new_vision_language_dataset"  # Replace with actual dataset name
    ```

2. Implement the `transform_conversation` method:

```python
def transform_conversation(self, example: dict) -> Conversation:
    """Transform a single conversation example into a Conversation object."""
    text_input = example['text_input']
    text_output = example['text_output']
    image_path = example['image_path']

    messages = [
        Message(role=Role.USER, content=text_input),
        Message(role=Role.USER, content=image_path, type=Type.IMAGE_PATH),  # Assuming image path for now
        Message(role=Role.ASSISTANT, content=text_output),
    ]

    return Conversation(messages=messages)
```

## Step 3: Handle Different Image Types

Depending on how your dataset stores images, you might need to adjust the `Message` creation for the image. Here are examples for different scenarios:

```python
# For image paths:
Message(role=Role.USER, content=image_source, type=Type.IMAGE_PATH)

# For image URLs:
Message(role=Role.USER, content=image_source, type=Type.IMAGE_URL)

# For binary image data:
Message(role=Role.USER, binary=image_source, type=Type.IMAGE_BINARY)
```

## Step 4: Register the Dataset

Ensure your dataset is registered in `src/oumi/datasets/__init__.py`:

```python
from oumi.datasets.vision_language.new_vision_language_dataset import NewVisionLanguageDataset

__all__ = [
    # ... other datasets ...
    "NewVisionLanguageDataset",
]
```

## Step 5: Use the Dataset in Configuration

Now you can use your dataset in a configuration file. Create or modify a YAML file in the `configs/` directory:

```yaml
data:
  train:
    datasets:
      - dataset_name: "new_vision_language_dataset"
        split: "train"
        dataset_kwargs:
          # any other kwargs here will be passed to the dataset constructor
          processor_name: "openai/clip-vit-base-patch32"  # or any other suitable processor
```

## Step 6: Custom Dataset Loading (Optional)

If your dataset isn't available through the Hugging Face `datasets` library, you may need to override the `_load_data` method in your dataset class:

```python
import pandas as pd

class NewVisionLanguageDataset(VisionLanguageSftDataset):
    # ... other methods ...

    def _load_data(self) -> pd.DataFrame:
        # Load your data here, e.g., from a CSV file
        data = pd.read_csv('path/to/your/data.csv')
        return data
```

## Step 7: Testing Your Dataset

Create a test file in the `tests/datasets/` directory, e.g., `test_new_vision_language_dataset.py`:

```python
import pytest
from oumi.datasets import NewVisionLanguageDataset

def test_new_vision_language_dataset():
    dataset = NewVisionLanguageDataset(split="train")
    assert len(dataset) > 0

    sample = dataset[0]
    assert 'input_ids' in sample
    assert 'labels' in sample
    # Add more assertions based on your dataset's structure
```
