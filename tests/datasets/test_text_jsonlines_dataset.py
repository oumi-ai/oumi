import tempfile
from pathlib import Path

import jsonlines
import pandas as pd
import pytest

from oumi.core.types.turn import Conversation
from oumi.datasets.sft_jsonlines import TextSftJsonLinesDataset


@pytest.fixture
def sample_jsonlines_data():
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you! How can I assist you today?",
                },
                {
                    "role": "user",
                    "content": "Can you explain what machine learning is?",
                },
                {
                    "role": "assistant",
                    "content": "Certainly! Machine learning is a"
                    " branch of artificial intelligence...",
                },
            ]
        }
    ]


def test_text_jsonlines_init_with_data(sample_jsonlines_data):
    dataset = TextSftJsonLinesDataset(data=sample_jsonlines_data)
    assert len(dataset._data) == 1
    assert ["messages"] == dataset._data.columns


def test_text_jsonlines_init_with_data_custom_data_column(sample_jsonlines_data):
    dataset = TextSftJsonLinesDataset(data=sample_jsonlines_data, data_column="foo")
    assert len(dataset._data) == 1
    assert ["foo"] == dataset._data.columns


def test_text_jsonlines_init_with_dataset_path(sample_jsonlines_data):
    with tempfile.TemporaryDirectory() as folder:
        valid_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(valid_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        dataset = TextSftJsonLinesDataset(dataset_path=valid_jsonlines_filename)
        assert len(dataset._data) == 1
        assert ["messages"] == dataset._data.columns


def test_text_jsonlines_transform_conversation(sample_jsonlines_data):
    dataset = TextSftJsonLinesDataset(data=sample_jsonlines_data)
    conversation = dataset.conversation(0)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 4
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Hello, how are you?"
    assert conversation.messages[1].role == "assistant"
    assert (
        conversation.messages[1].content
        == "I'm doing well, thank you! How can I assist you today?"
    )


def test_text_jsonlines_init_with_invalid_input(sample_jsonlines_data):
    with tempfile.TemporaryDirectory() as folder:
        valid_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(valid_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        with pytest.raises(ValueError, match="Dataset path or data must be provided"):
            TextSftJsonLinesDataset()

        with pytest.raises(ValueError, match="Dataset path or data must be provided"):
            TextSftJsonLinesDataset(data_column="some_column_name")

        with pytest.raises(
            ValueError, match="Dataset path does not exist: invalid_path.jsonl"
        ):
            TextSftJsonLinesDataset(dataset_path="invalid_path.jsonl")

        with pytest.raises(
            ValueError,
            match="Either dataset_path or data must be provided, but not both",
        ):
            TextSftJsonLinesDataset(dataset_path=valid_jsonlines_filename, data=[])

        with pytest.raises(
            ValueError,
            match="Dataset path must end with .jsonl",
        ):
            TextSftJsonLinesDataset(dataset_path="invalid_extension.json")

        # Directory ending with .jsonl
        temp_dir_name = Path(folder) / "subdir.jsonl"
        temp_dir_name.mkdir()
        with pytest.raises(
            ValueError,
            match="Dataset path is not a file",
        ):
            TextSftJsonLinesDataset(dataset_path=temp_dir_name)

        with pytest.raises(ValueError, match="Data column not found in dataset"):
            TextSftJsonLinesDataset(
                dataset_path=valid_jsonlines_filename, data_column="foo"
            )


def test_text_jsonlines_load_data():
    dataset = TextSftJsonLinesDataset(
        data=[{"messages": [{"role": "user", "content": "Test"}]}]
    )
    loaded_data = dataset._load_data()
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == 1
    assert ["messages"] == loaded_data.columns


def test_text_jsonlines_getitem():
    data = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm good, thanks!"},
            ]
        },
    ]
    dataset = TextSftJsonLinesDataset(data=data)

    item = dataset.conversation(0)
    assert len(item.messages) == 2

    with pytest.raises(IndexError):
        _ = dataset.conversation(2)


def test_text_jsonlines_len():
    data = [
        {"messages": [{"role": "user", "content": "Hello"}]},
        {"messages": [{"role": "user", "content": "How are you?"}]},
    ]
    dataset = TextSftJsonLinesDataset(data=data)
    assert len(dataset) == 2
