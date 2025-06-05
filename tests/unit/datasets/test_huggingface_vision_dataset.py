"""Tests for HuggingFaceVisionDataset."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from oumi.core.types.conversation import ContentItem, Conversation, Role, Type
from oumi.datasets.vision_language.huggingface import HuggingFaceVisionDataset

_IMAGE_TOKEN = "<image_token>"
_IMAGE_TOKEN_ID = 32001


def create_mock_processor():
    processor = Mock()
    processor.processor_name = "Salesforce/blip2-opt-2.7b"
    processor.tokenizer = Mock()
    processor.image_processor = Mock()
    processor.chat_template = None
    processor.image_token = _IMAGE_TOKEN
    processor.image_token_id = _IMAGE_TOKEN_ID
    processor.label_ignore_index = -100
    return processor


@pytest.fixture
def mock_processor():
    return create_mock_processor()


@pytest.fixture
def sample_dataset_example():
    mock_image = Mock()
    mock_image.bytes = b"fake_image_bytes"

    return {
        "image": mock_image,
        "question": "What is in this image?",
        "answer": "A cat sitting on a table.",
    }


@pytest.fixture
def sample_dataset_example_with_url():
    return {
        "image": "https://example.com/image.jpg",
        "question": "Describe this image.",
        "answer": "A beautiful landscape.",
    }


@pytest.fixture
def sample_dataset_example_with_path():
    return {
        "image": "/path/to/image.jpg",
        "question": "What do you see?",
        "answer": "A dog playing in the park.",
    }


@pytest.fixture
def sample_dataset_example_no_answer():
    return {
        "image": "https://example.com/image.jpg",
        "question": "What is in this image?",
    }


@pytest.fixture
def sample_dataset_example_with_system_prompt():
    return {
        "image": "https://example.com/image.jpg",
        "question": "What is in this image?",
        "answer": "A cat sitting on a table.",
        "system_prompt": "You are a helpful vision assistant.",
    }


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_init_success(mock_load_data, mock_tokenizer, mock_processor):
    mock_load_data.return_value = pd.DataFrame()

    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    assert dataset.image_column == "image"
    assert dataset.question_column == "question"
    assert dataset.answer_column == "answer"
    assert dataset.system_prompt is None
    assert dataset.system_prompt_column is None


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_init_with_custom_columns(mock_load_data, mock_tokenizer, mock_processor):
    mock_load_data.return_value = pd.DataFrame()

    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="img",
        question_column="query",
        answer_column="response",
        system_prompt="You are helpful assistant.",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    assert dataset.image_column == "img"
    assert dataset.question_column == "query"
    assert dataset.answer_column == "response"
    assert dataset.system_prompt == "You are helpful assistant."
    assert dataset.system_prompt_column is None


def test_init_missing_hf_dataset_path(mock_tokenizer, mock_processor):
    with pytest.raises(
        ValueError, match="The `hf_dataset_path` parameter must be provided"
    ):
        with patch.object(HuggingFaceVisionDataset, "_load_data") as mock_load_data:
            mock_load_data.return_value = pd.DataFrame()
            HuggingFaceVisionDataset(
                hf_dataset_path="",
                image_column="image",
                question_column="question",
                tokenizer=mock_tokenizer,
                processor=mock_processor,
            )


def test_init_missing_image_column(mock_tokenizer, mock_processor):
    with pytest.raises(
        ValueError, match="The `image_column` parameter must be provided"
    ):
        with patch.object(HuggingFaceVisionDataset, "_load_data") as mock_load_data:
            mock_load_data.return_value = pd.DataFrame()
            HuggingFaceVisionDataset(
                hf_dataset_path="test/dataset",
                image_column="",
                question_column="question",
                tokenizer=mock_tokenizer,
                processor=mock_processor,
            )


def test_init_missing_question_column(mock_tokenizer, mock_processor):
    with pytest.raises(
        ValueError, match="The `question_column` parameter must be provided"
    ):
        with patch.object(HuggingFaceVisionDataset, "_load_data") as mock_load_data:
            mock_load_data.return_value = pd.DataFrame()
            HuggingFaceVisionDataset(
                hf_dataset_path="test/dataset",
                image_column="image",
                question_column="",
                tokenizer=mock_tokenizer,
                processor=mock_processor,
            )


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_get_image_content_item_with_bytes_attribute(
    mock_load_data, mock_tokenizer, mock_processor
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    mock_image = Mock()
    mock_image.bytes = b"fake_image_bytes"

    content_item = dataset._get_image_content_item(mock_image)

    assert content_item.type == Type.IMAGE_BINARY
    assert content_item.binary == b"fake_image_bytes"


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_get_image_content_item_with_raw_bytes(
    mock_load_data, mock_tokenizer, mock_processor
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    content_item = dataset._get_image_content_item(b"raw_bytes")

    assert content_item.type == Type.IMAGE_BINARY
    assert content_item.binary == b"raw_bytes"


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_get_image_content_item_with_url(
    mock_load_data, mock_tokenizer, mock_processor
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    content_item = dataset._get_image_content_item("https://example.com/image.jpg")

    assert content_item.type == Type.IMAGE_URL
    assert content_item.content == "https://example.com/image.jpg"


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_get_image_content_item_unsupported_type(
    mock_load_data, mock_tokenizer, mock_processor
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    with pytest.raises(ValueError, match="Unsupported image data type"):
        dataset._get_image_content_item(123)


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_success(
    mock_load_data, mock_tokenizer, mock_processor, sample_dataset_example
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    conversation = dataset.transform_conversation(sample_dataset_example)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 2

    # Check user message
    user_message = conversation.messages[0]
    assert user_message.role == Role.USER
    assert isinstance(user_message.content, list)
    assert len(user_message.content) == 2

    # Check image content item
    image_item = user_message.content[0]
    assert image_item.type == Type.IMAGE_BINARY
    assert image_item.binary == b"fake_image_bytes"

    # Check text content item
    text_item = user_message.content[1]
    assert text_item.type == Type.TEXT
    assert text_item.content == "What is in this image?"

    # Check assistant message
    assistant_message = conversation.messages[1]
    assert assistant_message.role == Role.ASSISTANT
    assert assistant_message.content == "A cat sitting on a table."


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_with_static_system_prompt(
    mock_load_data, mock_tokenizer, mock_processor, sample_dataset_example
):
    """Test conversation transformation with static system prompt."""
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        system_prompt="You are a helpful vision assistant.",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    conversation = dataset.transform_conversation(sample_dataset_example)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 3

    # Check system message
    system_message = conversation.messages[0]
    assert system_message.role == Role.SYSTEM
    assert system_message.content == "You are a helpful vision assistant."

    # Check user message
    user_message = conversation.messages[1]
    assert user_message.role == Role.USER

    # Check assistant message
    assistant_message = conversation.messages[2]
    assert assistant_message.role == Role.ASSISTANT


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_with_system_prompt_column(
    mock_load_data,
    mock_tokenizer,
    mock_processor,
    sample_dataset_example_with_system_prompt,
):
    """Test conversation transformation with system prompt from column."""
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        system_prompt_column="system_prompt",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    conversation = dataset.transform_conversation(
        sample_dataset_example_with_system_prompt
    )

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 3

    # Check system message
    system_message = conversation.messages[0]
    assert system_message.role == Role.SYSTEM
    assert system_message.content == "You are a helpful vision assistant."


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_no_answer_column(
    mock_load_data, mock_tokenizer, mock_processor, sample_dataset_example_no_answer
):
    """Test conversation transformation without answer column."""
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column=None,  # No answer column
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    conversation = dataset.transform_conversation(sample_dataset_example_no_answer)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 1  # Only user message

    # Check user message
    user_message = conversation.messages[0]
    assert user_message.role == Role.USER
    assert isinstance(user_message.content, list)
    assert len(user_message.content) == 2


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_missing_answer_column_in_data(
    mock_load_data, mock_tokenizer, mock_processor, sample_dataset_example_no_answer
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column=None,  # Answer column specified but not in data
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    conversation = dataset.transform_conversation(sample_dataset_example_no_answer)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 1  # Only user message, no assistant message


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_empty_answer(
    mock_load_data, mock_tokenizer, mock_processor
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    example = {
        "image": "https://example.com/image.jpg",
        "question": "What is this?",
        "answer": "",  # Empty answer
    }

    conversation = dataset.transform_conversation(example)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 2

    # Check assistant message has empty content
    assistant_message = conversation.messages[1]
    assert assistant_message.role == Role.ASSISTANT
    assert assistant_message.content == ""


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_missing_column(
    mock_load_data, mock_tokenizer, mock_processor
):
    """Test conversation transformation with missing required column."""
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    example = {
        "image": "test.jpg",
        # Missing question column
        "answer": "Some answer",
    }

    with pytest.raises(
        ValueError,
        match=(
            r"The column 'question_column' \(specified as question\) is not present"
            r" in the example. Available columns: \['image', 'answer'\]"
        ),
    ):
        dataset.transform_conversation(example)


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_with_url(
    mock_load_data, mock_tokenizer, mock_processor, sample_dataset_example_with_url
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    conversation = dataset.transform_conversation(sample_dataset_example_with_url)

    # Check image content item
    image_item = conversation.messages[0].content[0]
    assert isinstance(image_item, ContentItem)
    assert image_item.type == Type.IMAGE_URL
    assert image_item.content == "https://example.com/image.jpg"


@patch.object(HuggingFaceVisionDataset, "_load_data")
def test_transform_conversation_empty_system_prompt_column(
    mock_load_data, mock_tokenizer, mock_processor
):
    mock_load_data.return_value = pd.DataFrame()
    dataset = HuggingFaceVisionDataset(
        hf_dataset_path="test/dataset",
        image_column="image",
        question_column="question",
        answer_column="answer",
        system_prompt_column="system_prompt",
        tokenizer=mock_tokenizer,
        processor=mock_processor,
    )

    example = {
        "image": "https://example.com/image.jpg",
        "question": "What is this?",
        "answer": "A cat",
        "system_prompt": "",  # Empty system prompt
    }

    conversation = dataset.transform_conversation(example)

    # Should not add system message for empty prompt
    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 2  # No system message
    assert conversation.messages[0].role == Role.USER
    assert conversation.messages[1].role == Role.ASSISTANT
