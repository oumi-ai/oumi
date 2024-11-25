import tempfile
from pathlib import Path

import jsonlines
import pytest
from pandas.core.api import DataFrame as DataFrame

from oumi.core.types.conversation import Conversation, Type
from oumi.datasets.vision_language.vision_jsonlines import (
    VLJsonlinesDataset,
)


@pytest.fixture
def sample_jsonlines_data():
    return [
        {
            "messages": [
                {"role": "user", "content": "Describe this image:", "type": "text"},
                {"role": "user", "content": "path/to/image.jpg", "type": "image_path"},
                {
                    "role": "assistant",
                    "content": "A scenic view of the puget sound.",
                    "type": "text",
                },
            ]
        }
    ]


def test_jsonlines_init_with_data(sample_jsonlines_data, mock_tokenizer):
    dataset = VLJsonlinesDataset(data=sample_jsonlines_data, tokenizer=mock_tokenizer)
    assert len(dataset._data) == 1
    assert ["_messages_column"] == dataset._data.columns

    conversation = dataset.conversation(0)
    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 3
    assert conversation.messages[0].content == "Describe this image:"
    assert conversation.messages[1].content == "path/to/image.jpg"
    assert conversation.messages[1].type == Type.IMAGE_PATH
    assert conversation.messages[2].content == "A scenic view of the puget sound."


def test_jsonlines_init_with_dataset_path(sample_jsonlines_data, mock_tokenizer):
    with tempfile.TemporaryDirectory() as folder:
        vaild_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(vaild_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        dataset = VLJsonlinesDataset(
            dataset_path=vaild_jsonlines_filename, tokenizer=mock_tokenizer
        )
        assert len(dataset._data) == 1
        assert ["_messages_column"] == dataset._data.columns

        conversation = dataset.conversation(0)

        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 3
        assert conversation.messages[0].content == "Describe this image:"
        assert conversation.messages[1].content == "path/to/image.jpg"
        assert conversation.messages[1].type == Type.IMAGE_PATH
        assert conversation.messages[2].content == "A scenic view of the puget sound."


def test_jsonlines_init_with_invalid_input(sample_jsonlines_data, mock_tokenizer):
    with tempfile.TemporaryDirectory() as folder:
        valid_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(valid_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        with pytest.raises(ValueError, match="Dataset path or data must be provided"):
            VLJsonlinesDataset(tokenizer=mock_tokenizer)

        with pytest.raises(
            FileNotFoundError,
            match="Provided path does not exist: 'invalid_path.jsonl'.",
        ):
            VLJsonlinesDataset(
                dataset_path="invalid_path.jsonl", tokenizer=mock_tokenizer
            )

        with pytest.raises(
            ValueError,
            match="Either dataset_path or data must be provided, but not both",
        ):
            VLJsonlinesDataset(
                dataset_path=valid_jsonlines_filename, data=[], tokenizer=mock_tokenizer
            )

        # Directory ending with .jsonl
        temp_dir_name = Path(folder) / "subdir.jsonl"
        temp_dir_name.mkdir()
        with pytest.raises(
            ValueError,
            match="Provided path is a directory, expected a file",
        ):
            VLJsonlinesDataset(dataset_path=temp_dir_name, tokenizer=mock_tokenizer)

        with pytest.raises(ValueError, match="Tokenizer must be provided"):
            VLJsonlinesDataset(
                dataset_path=valid_jsonlines_filename,
            )
