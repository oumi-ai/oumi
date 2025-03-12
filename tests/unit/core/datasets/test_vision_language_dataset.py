import copy
import functools
import io
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import transformers
from pandas.core.api import DataFrame as DataFrame
from PIL import Image
from typing_extensions import override

from oumi.builders import build_chat_template
from oumi.core.datasets.vision_language_dataset import VisionLanguageSftDataset
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)


class EqBytesIO:
    def __init__(self, bytes_io: io.BytesIO):
        self._byte_io = bytes_io

    def __eq__(self, other):
        return (
            isinstance(other, io.BytesIO)
            and other.getvalue() == self._byte_io.getvalue()
        )


_IMAGE_TOKEN = "<image_token>"
_IMAGE_TOKEN_ID = 32001


@pytest.fixture
def mock_image_tokenizer() -> MagicMock:
    def _convert_tokens_to_ids(token: str) -> int:
        if token == _IMAGE_TOKEN:
            return _IMAGE_TOKEN_ID
        return 101

    mock = MagicMock(spec=BaseTokenizer)
    mock.pad_token_id = 0
    mock.chat_template = build_chat_template("llava")
    mock.convert_tokens_to_ids = MagicMock(side_effect=_convert_tokens_to_ids)
    return mock


def create_mock_processor(label_ignore_index: Optional[int]):
    processor = Mock()
    processor.processor_name = "llava-hf/llava-1.5-7b-hf"
    processor.tokenizer = Mock()
    processor.image_processor = Mock()
    processor.chat_template = None
    processor.image_token = _IMAGE_TOKEN
    processor.image_token_id = _IMAGE_TOKEN_ID
    processor.label_ignore_index = label_ignore_index
    processor.side_effect = (
        lambda images, text, return_tensors, padding: transformers.BatchEncoding(
            data={
                "input_ids": [[101, 102, _IMAGE_TOKEN_ID, 104]],
                "attention_mask": [[1, 1, 1, 1]],
                "pixel_values": [
                    [
                        np.ones(shape=(3, 2, 8)),
                        np.zeros(shape=(3, 2, 8)),
                        np.ones(shape=(3, 2, 8)) * 0.5,
                        np.ones(shape=(3, 2, 8)) * 0.7,
                    ]
                ],
            },
            tensor_type=return_tensors,
        )
    )
    return processor


@pytest.fixture
def mock_processor():
    return create_mock_processor(label_ignore_index=-100)


@pytest.fixture
def mock_processor_no_label_ignore_index():
    return create_mock_processor(label_ignore_index=None)


@functools.cache  # same as @cache added in Python 3.9
def _get_test_png_image_bytes(image_size: Optional[tuple[int, int]] = None) -> bytes:
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
            Message(
                role=Role.USER,
                content=[
                    ContentItem(content="Describe this image:", type=Type.TEXT),
                    ContentItem(content="path/to/image.jpg", type=Type.IMAGE_PATH),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content="A beautiful sunset over the ocean.",
            ),
        ]
    )


@pytest.fixture
def sample_conversation_using_image_binary():
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(content="Describe this image:", type=Type.TEXT),
                    ContentItem(
                        binary=_get_test_png_image_bytes(), type=Type.IMAGE_BINARY
                    ),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content="A beautiful sunset over the ocean.",
            ),
        ]
    )


@pytest.fixture
def sample_dataset_image_path_no_label_ignore_index(
    mock_processor_no_label_ignore_index: Mock,
    sample_conversation_using_image_path: Conversation,
    mock_image_tokenizer: MagicMock,
):
    class TestDatasetImagePath(VisionLanguageSftDataset):
        default_dataset = "custom"

        @override
        def transform_conversation(self, example):
            return sample_conversation_using_image_path

        @override
        def _load_data(self):
            pass

    return TestDatasetImagePath(
        processor=mock_processor_no_label_ignore_index,
        tokenizer=mock_image_tokenizer,
    )


@pytest.fixture
def sample_dataset_image_binary_label_ignore_index(
    mock_processor: Mock,
    sample_conversation_using_image_binary: Conversation,
    mock_image_tokenizer: MagicMock,
):
    class TestDatasetImageBinary(VisionLanguageSftDataset):
        default_dataset = "custom"

        @override
        def transform_conversation(self, example):
            return sample_conversation_using_image_binary

        @override
        def _load_data(self):
            pass

    return TestDatasetImageBinary(
        processor=mock_processor, tokenizer=mock_image_tokenizer
    )


@pytest.fixture
def sample_dataset_image_binary_return_conversations_barebones(
    sample_conversation_using_image_binary: Conversation,
):
    class TestDatasetImageBinary(VisionLanguageSftDataset):
        default_dataset = "custom"

        @override
        def transform_conversation(self, example):
            return sample_conversation_using_image_binary

        @override
        def _load_data(self):
            pass

    return TestDatasetImageBinary(
        # No processor/tokenizer
        return_conversations=True,
    )


@pytest.fixture
def sample_dataset_image_binary_return_conversations_with_processor(
    mock_processor: Mock,
    sample_conversation_using_image_binary: Conversation,
    mock_image_tokenizer: MagicMock,
):
    class TestDatasetImageBinary(VisionLanguageSftDataset):
        default_dataset = "custom"

        @override
        def transform_conversation(self, example):
            return sample_conversation_using_image_binary

        @override
        def _load_data(self):
            pass

    return TestDatasetImageBinary(
        processor=mock_processor,
        tokenizer=mock_image_tokenizer,
        return_conversations=True,
    )


@pytest.fixture
def return_conversation_fixtures(
    sample_dataset_image_binary_return_conversations_barebones,
    sample_dataset_image_binary_return_conversations_with_processor,
):
    return {
        "barebones": (sample_dataset_image_binary_return_conversations_barebones),
        "with_processor": (
            sample_dataset_image_binary_return_conversations_with_processor
        ),
    }


def test_transform_simple_model_using_image_path(
    sample_dataset_image_path_no_label_ignore_index,
):
    my_dataset = sample_dataset_image_path_no_label_ignore_index
    with patch.object(
        my_dataset._feature_generator,
        "_load_image",
    ) as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = my_dataset.transform({"example": "data"})

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


def test_transform_simple_model_using_image_binary(
    sample_dataset_image_binary_label_ignore_index,
):
    my_dataset = sample_dataset_image_binary_label_ignore_index
    with patch.object(my_dataset._feature_generator, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = my_dataset.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert np.array(result["input_ids"]).shape == (4,)
    assert np.all(
        np.array(result["input_ids"]) == np.array([101, 102, _IMAGE_TOKEN_ID, 104])
    )
    assert "attention_mask" in result
    assert np.array(result["attention_mask"]).shape == (4,)
    assert "labels" in result
    assert np.array(result["labels"]).shape == (4,)
    assert np.all(np.array(result["labels"]) == np.array([101, 102, -100, 104]))
    assert "pixel_values" in result
    assert np.array(result["pixel_values"]).shape == (4, 3, 2, 8)


def test_transform_instruct_model_using_image_path(
    sample_dataset_image_path_no_label_ignore_index,
    mock_processor_no_label_ignore_index: Mock,
):
    my_dataset = sample_dataset_image_path_no_label_ignore_index
    mock_processor_no_label_ignore_index.chat_template = "Template"
    mock_processor_no_label_ignore_index.apply_chat_template = Mock(
        return_value="Processed template"
    )

    with patch.object(my_dataset._feature_generator, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = my_dataset.transform({"example": "data"})

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
    mock_processor_no_label_ignore_index.apply_chat_template.assert_called_once()


def test_transform_instruct_model_using_image_binary(
    sample_dataset_image_binary_label_ignore_index, mock_processor: Mock
):
    my_dataset = sample_dataset_image_binary_label_ignore_index
    mock_processor.chat_template = "Template"
    mock_processor.apply_chat_template = Mock(return_value="Processed template")

    with patch.object(my_dataset._feature_generator, "_load_image") as mock_load_image:
        mock_image = Mock(spec=Image.Image)
        mock_load_image.return_value = mock_image

        result = my_dataset.transform({"example": "data"})

    assert isinstance(result, dict)
    assert "input_ids" in result
    assert np.array(result["input_ids"]).shape == (4,)
    assert np.all(
        np.array(result["input_ids"]) == np.array([101, 102, _IMAGE_TOKEN_ID, 104])
    )
    assert "attention_mask" in result
    assert np.array(result["attention_mask"]).shape == (4,)
    assert "labels" in result
    assert np.array(result["labels"]).shape == (4,)
    assert np.all(np.array(result["labels"]) == np.array([101, 102, -100, 104]))
    assert "pixel_values" in result
    assert np.array(result["pixel_values"]).shape == (4, 3, 2, 8)
    mock_processor.apply_chat_template.assert_called_once()


@pytest.mark.parametrize(
    "fixture_name",
    ["barebones", "with_processor"],
)
def test_return_conversations(
    fixture_name: str,
    return_conversation_fixtures,
    mock_processor: Mock,
    sample_conversation_using_image_binary: Conversation,
):
    my_dataset = return_conversation_fixtures[fixture_name]
    mock_processor.chat_template = "Template"
    mock_processor.apply_chat_template = Mock(return_value="Processed template")

    assert my_dataset._feature_generator is None

    result = my_dataset.transform({"example": "data"})

    assert isinstance(result, dict)
    assert set({"conversation_json"}) == set(result.keys())
    assert "conversation_json" in result
    conversation = Conversation.from_json(result["conversation_json"])
    assert conversation == sample_conversation_using_image_binary

    mock_processor.apply_chat_template.assert_not_called()


def test_return_conversations_with_max_images(
    mock_processor,
    mock_image_tokenizer,
    sample_conversation_using_image_binary: Conversation,
):
    class TestDatasetImageBinary(VisionLanguageSftDataset):
        default_dataset = "custom"

        @override
        def transform_conversation(self, example):
            convo = sample_conversation_using_image_binary
            messages = copy.deepcopy(convo.messages)
            last_message = messages[-1]
            assert isinstance(last_message.content, str)
            messages[-1] = Message(
                role=last_message.role,
                content=[
                    ContentItem(content=last_message.content, type=Type.TEXT),
                    ContentItem(content="/anotherimage.jpg", type=Type.IMAGE_PATH),
                    ContentItem(
                        content="http://oumi.ai/yet_another_image.png",
                        type=Type.IMAGE_URL,
                    ),
                ],
            )
            return Conversation(
                conversation_id=convo.conversation_id,
                messages=messages,
                metadata=convo.metadata,
            )

        @override
        def _load_data(self):
            pass

    my_dataset = TestDatasetImageBinary(
        processor=mock_processor,
        tokenizer=mock_image_tokenizer,
        return_conversations=True,
        max_images=1,
    )

    mock_processor.chat_template = "Template"
    mock_processor.apply_chat_template = Mock(return_value="Processed template")

    assert my_dataset._feature_generator is None

    result = my_dataset.transform({"example": "data"})

    assert isinstance(result, dict)
    assert set({"conversation_json"}) == set(result.keys())
    assert "conversation_json" in result
    conversation = Conversation.from_json(result["conversation_json"])
    assert conversation == sample_conversation_using_image_binary

    mock_processor.apply_chat_template.assert_not_called()


def test_dataset_no_tokenizer(
    mock_processor: Mock,
    sample_conversation_using_image_binary: Conversation,
):
    class FooDataset(VisionLanguageSftDataset):
        default_dataset = "custom"

        @override
        def transform_conversation(self, example):
            return sample_conversation_using_image_binary

        @override
        def _load_data(self):
            pass

    with pytest.raises(ValueError, match="Tokenizer must be provided"):
        FooDataset(processor=mock_processor)

    with pytest.raises(ValueError, match="Tokenizer must be provided"):
        FooDataset(processor=mock_processor, tokenizer=None)
