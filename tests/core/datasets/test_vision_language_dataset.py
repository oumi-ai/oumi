import functools
import io
from typing import Optional, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pandas.core.api import DataFrame as DataFrame
from PIL import Image

from oumi.core.datasets.vision_language_dataset import VisionLanguageSftDataset
from oumi.core.types.turn import Conversation, Message, Role, Type


class EqBytesIO:
    def __init__(self, bytes_io: io.BytesIO):
        self._byte_io = bytes_io

    def __eq__(self, other):
        return (
            isinstance(other, io.BytesIO)
            and other.getvalue() == self._byte_io.getvalue()
        )


@pytest.fixture
def mock_processor():
    processor = Mock()
    processor.tokenizer = Mock()
    processor.image_processor = Mock()
    processor.chat_template = None
    processor.side_effect = lambda images, text, return_tensors, padding: {
        "input_ids": [[101, 102, 103, 104]],
        "attention_mask": [[1, 1, 1, 1]],
        "pixel_values": [
            [
                np.ones(shape=(3, 2, 8)),
                np.zeros(shape=(3, 2, 8)),
                np.ones(shape=(3, 2, 8)) * 0.5,
                np.ones(shape=(3, 2, 8)) * 0.7,
            ]
        ],
    }
    return processor


@functools.lru_cache(maxsize=None)  # same as @cache added in Python 3.9
def _get_test_png_image_bytes(image_size: Optional[Tuple[int, int]] = None) -> bytes:
    if image_size is None:
        image_size = (80, 40)
    image = Image.new(mode="RGBA", size=image_size)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format="PNG")
    return bytes_io.getvalue()


@pytest.fixture
def sample_conversation_using_image_path():
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
def sample_conversation_using_image_binary():
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Describe this image:", type=Type.TEXT),
            Message(
                role=Role.USER,
                binary=_get_test_png_image_bytes(),
                type=Type.IMAGE_BINARY,
            ),
            Message(
                role=Role.ASSISTANT,
                content="A beautiful sunset over the ocean.",
                type=Type.TEXT,
            ),
        ]
    )


@pytest.fixture
def test_dataset_using_image_path(
    mock_processor: Mock, sample_conversation_using_image_path: Conversation
):
    class TestDatasetImagePath(VisionLanguageSftDataset):
        default_dataset = "custom"

        def transform_conversation(self, example):
            return sample_conversation_using_image_path

        def _load_data(self):
            pass

    return TestDatasetImagePath(processor=mock_processor)


@pytest.fixture
def test_dataset_using_image_binary(
    mock_processor: Mock, sample_conversation_using_image_binary: Conversation
):
    class TestDatasetImageBinary(VisionLanguageSftDataset):
        default_dataset = "custom"

        def transform_conversation(self, example):
            return sample_conversation_using_image_binary

        def _load_data(self):
            pass

    return TestDatasetImageBinary(processor=mock_processor)


def test_transform_image_using_image_path(test_dataset_using_image_path):
    with patch("PIL.Image.open") as mock_open:
        mock_image = Mock(spec=Image.Image)
        mock_open.return_value.convert.return_value = mock_image

        test_dataset_using_image_path.transform_image("path/to/image.jpg")

        mock_open.assert_called_once_with("path/to/image.jpg")
        test_dataset_using_image_path._image_processor.assert_called_once()


def test_transform_image_using_image_binary(test_dataset_using_image_binary):
    with patch("PIL.Image.open") as mock_open:
        mock_image = Mock(spec=Image.Image)
        mock_open.return_value.convert.return_value = mock_image

        test_image_bytes = _get_test_png_image_bytes()
        test_dataset_using_image_binary.transform_image(
            Message(type=Type.IMAGE_BINARY, binary=test_image_bytes, role=Role.USER)
        )

        mock_open.assert_called_once_with(EqBytesIO(io.BytesIO(test_image_bytes)))
        test_dataset_using_image_binary._image_processor.assert_called_once()


def test_transform_simple_model_using_image_path(test_dataset_using_image_path):
    with patch.object(test_dataset_using_image_path, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = test_dataset_using_image_path.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert np.array(result["input_ids"]).shape == (4,)
    assert "attention_mask" in result
    assert np.array(result["attention_mask"]).shape == (4,)
    assert "labels" in result
    assert np.array(result["labels"]).shape == (4,)
    assert np.all(np.array(result["labels"]) == np.array(result["input_ids"]))
    assert "pixel_values" in result
    assert np.array(result["pixel_values"]).shape == (4, 3, 2, 8)


def test_transform_simple_model_using_image_binary(test_dataset_using_image_binary):
    with patch.object(
        test_dataset_using_image_binary, "_load_image"
    ) as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = test_dataset_using_image_binary.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert np.array(result["input_ids"]).shape == (4,)
    assert "attention_mask" in result
    assert np.array(result["attention_mask"]).shape == (4,)
    assert "labels" in result
    assert np.array(result["labels"]).shape == (4,)
    assert np.all(np.array(result["labels"]) == np.array(result["input_ids"]))
    assert "pixel_values" in result
    assert np.array(result["pixel_values"]).shape == (4, 3, 2, 8)


def test_transform_instruct_model_using_image_path(
    test_dataset_using_image_path, mock_processor: Mock
):
    mock_processor.chat_template = "Template"
    mock_processor.apply_chat_template = Mock(return_value="Processed template")

    with patch.object(test_dataset_using_image_path, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = test_dataset_using_image_path.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert np.array(result["input_ids"]).shape == (4,)
    assert "attention_mask" in result
    assert np.array(result["attention_mask"]).shape == (4,)
    assert "labels" in result
    assert np.array(result["labels"]).shape == (4,)
    assert np.all(np.array(result["labels"]) == np.array(result["input_ids"]))
    assert "pixel_values" in result
    assert np.array(result["pixel_values"]).shape == (4, 3, 2, 8)
    mock_processor.apply_chat_template.assert_called_once()


def test_transform_instruct_model_using_image_binary(
    test_dataset_using_image_binary, mock_processor: Mock
):
    mock_processor.chat_template = "Template"
    mock_processor.apply_chat_template = Mock(return_value="Processed template")

    with patch.object(
        test_dataset_using_image_binary, "_load_image"
    ) as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = test_dataset_using_image_binary.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert np.array(result["input_ids"]).shape == (4,)
    assert "attention_mask" in result
    assert np.array(result["attention_mask"]).shape == (4,)
    assert "labels" in result
    assert np.array(result["labels"]).shape == (4,)
    assert np.all(np.array(result["labels"]) == np.array(result["input_ids"]))
    assert "pixel_values" in result
    assert np.array(result["pixel_values"]).shape == (4, 3, 2, 8)
    mock_processor.apply_chat_template.assert_called_once()
