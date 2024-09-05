import json
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from lema.core.types.turn import Role
from lema.datasets.vision_language import VisionLanguageSftDataset


@pytest.fixture(scope="module")
def temp_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        image_dir = Path(temp_dir) / "images"
        image_dir.mkdir(exist_ok=True)

        # Create a sample image
        sample_image = Image.new("RGB", (100, 100), color="red")
        image_filename = "test_image.jpg"
        sample_image.save(image_dir / image_filename)

        # Create a sample dataset in jsonlines format
        sample_data = [
            {
                "input": "Describe this image",
                "output": "This is a red square image",
                "image_filename": image_filename,
            }
        ]

        # Save the sample dataset as a jsonlines file
        dataset_path = Path(temp_dir) / "test_dataset.jsonl"
        with open(dataset_path, "w") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        yield str(dataset_path), str(image_dir)


@pytest.fixture
def vision_language_dataset(temp_dataset):
    dataset_path, image_dir = temp_dataset
    return VisionLanguageSftDataset(
        dataset_name_or_path=dataset_path, image_dir=image_dir, image_size=224
    )


def test_dataset_initialization(vision_language_dataset, temp_dataset):
    _, image_dir = temp_dataset
    assert len(vision_language_dataset) == 1
    assert vision_language_dataset.image_dir == image_dir


def test_transform_conversation(vision_language_dataset):
    example = vision_language_dataset.data.iloc[0].to_dict()
    conversation = vision_language_dataset.transform_conversation(example)

    assert len(conversation.messages) == 2
    assert isinstance(conversation.image, str)


def test_transform(vision_language_dataset):
    example = vision_language_dataset.data.iloc[0].to_dict()
    conversation = vision_language_dataset.transform(example)

    assert isinstance(conversation["text"], str)
    assert isinstance(conversation["image"], torch.Tensor)
    assert conversation.image.shape == (3, 224, 224)


def test_getitem(vision_language_dataset):
    item = vision_language_dataset[0]

    assert "conversation" in item
    assert "image" in item
    assert isinstance(item["conversation"], list)
    assert len(item["conversation"]) == 2
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)


def test_message_content(vision_language_dataset):
    item = vision_language_dataset[0]
    messages = item["conversation"]

    assert messages[0].role == Role.USER
    assert messages[0].content == "Describe this image"
    assert messages[1].role == Role.ASSISTANT
    assert messages[1].content == "This is a red square image"


def test_image_normalization(vision_language_dataset):
    item = vision_language_dataset[0]
    image = item["image"]

    # Check if the image is normalized
    assert torch.allclose(image.mean(), torch.tensor([0.0, 0.0, 0.0]), atol=1e-1)
    assert torch.allclose(image.std(), torch.tensor([1.0, 1.0, 1.0]), atol=1e-1)
