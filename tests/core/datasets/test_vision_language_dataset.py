from unittest.mock import Mock, patch

import pytest
from pandas.core.api import DataFrame as DataFrame
from PIL import Image

from oumi.core.datasets.vision_language_dataset import VisionLanguageSftDataset
from oumi.core.types.turn import Conversation, Message, Role, Type


@pytest.fixture
def mock_processor():
    processor = Mock()
    processor.tokenizer = Mock()
    processor.image_processor = Mock()
    processor.chat_template = None
    processor.side_effect = lambda images, text, return_tensors, padding: {
        "input_ids": [[1]],
        "attention_mask": Mock(),
        "pixel_values": [[1]],
    }
    return processor


@pytest.fixture
def sample_conversation():
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Describe this image:", type=Type.TEXT),
            Message(role=Role.USER, content="path/to/image.jpg", type=Type.IMAGE_PATH),
            Message(
                role=Role.ASSISTANT,
                content="A beautiful sunset over the ocean.",
                type=Type.TEXT,
            ),
        ]
    )


@pytest.fixture
def test_dataset(mock_processor, sample_conversation):
    class TestDataset(VisionLanguageSftDataset):
        default_dataset = "custom"

        def transform_conversation(self, example):
            return sample_conversation

        def _load_data(self):
            pass

    return TestDataset(processor=mock_processor)


def test_transform_image(test_dataset):
    with patch("PIL.Image.open") as mock_open:
        mock_image = Mock(spec=Image.Image)
        mock_open.return_value.convert.return_value = mock_image

        test_dataset.transform_image("path/to/image.jpg")

        mock_open.assert_called_once_with("path/to/image.jpg")
        test_dataset._image_processor.assert_called_once()


def test_transform_simple_model(test_dataset):
    with patch.object(test_dataset, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = test_dataset.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "pixel_values" in result


def test_transform_instruct_model(test_dataset, mock_processor):
    mock_processor.chat_template = "Template"
    mock_processor.apply_chat_template = Mock(return_value="Processed template")

    with patch.object(test_dataset, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = test_dataset.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert "pixel_values" in result
    mock_processor.apply_chat_template.assert_called_once()
