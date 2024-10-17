import json
import tempfile
from pathlib import Path

import jsonlines
import pandas as pd
import pytest

from oumi.core.types.conversation import Conversation
from oumi.datasets import TextSftJsonLinesDataset


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


@pytest.fixture
def sample_oumi_data():
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Translate the following English text to French\n\n"
                    "Hello, how are you?",
                },
                {
                    "role": "assistant",
                    "content": "Bonjour, comment allez-vous ?",
                },
            ]
        }
    ]


@pytest.fixture
def sample_alpaca_data():
    return [
        {
            "instruction": "Translate the following English text to French",
            "input": "Hello, how are you?",
            "output": "Bonjour, comment allez-vous ?",
        }
    ]


def test_text_jsonlines_init_with_data(sample_jsonlines_data):
    dataset = TextSftJsonLinesDataset(data=sample_jsonlines_data)
    assert len(dataset._data) == 1
    assert ["_messages_column"] == dataset._data.columns

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


def test_text_jsonlines_init_with_dataset_path(sample_jsonlines_data):
    with tempfile.TemporaryDirectory() as folder:
        valid_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(valid_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        dataset = TextSftJsonLinesDataset(dataset_path=valid_jsonlines_filename)
        assert len(dataset._data) == 1
        assert ["_messages_column"] == dataset._data.columns

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

        with pytest.raises(
            FileNotFoundError,
            match="Provided path does not exist: 'invalid_path.jsonl'.",
        ):
            TextSftJsonLinesDataset(dataset_path="invalid_path.jsonl")

        with pytest.raises(
            ValueError,
            match="Either dataset_path or data must be provided, but not both",
        ):
            TextSftJsonLinesDataset(dataset_path=valid_jsonlines_filename, data=[])

        # Directory ending with .jsonl
        temp_dir_name = Path(folder) / "subdir.jsonl"
        temp_dir_name.mkdir()
        with pytest.raises(
            ValueError,
            match="Provided path is a directory, expected a file",
        ):
            TextSftJsonLinesDataset(dataset_path=temp_dir_name)


def test_text_jsonlines_load_data():
    dataset = TextSftJsonLinesDataset(
        data=[{"messages": [{"role": "user", "content": "Test"}]}]
    )
    loaded_data = dataset._load_data()
    assert isinstance(loaded_data, pd.DataFrame)
    assert len(loaded_data) == 1
    assert ["_messages_column"] == loaded_data.columns


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


def test_load_oumi_format_jsonl(sample_oumi_data):
    with tempfile.TemporaryDirectory() as folder:
        file_path = Path(folder) / "oumi_data.jsonl"
        with jsonlines.open(file_path, mode="w") as writer:
            writer.write_all(sample_oumi_data)

        dataset = TextSftJsonLinesDataset(dataset_path=file_path)
        assert dataset._format == "oumi"
        assert len(dataset) == 1

        conversation = dataset.conversation(0)
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert (
            conversation.messages[0].content
            == "Translate the following English text to French\n\nHello, how are you?"
        )
        assert conversation.messages[1].role == "assistant"
        assert conversation.messages[1].content == "Bonjour, comment allez-vous ?"


def test_load_alpaca_format_json(sample_alpaca_data):
    with tempfile.TemporaryDirectory() as folder:
        file_path = Path(folder) / "alpaca_data.json"
        with open(file_path, "w") as f:
            json.dump(sample_alpaca_data, f)

        dataset = TextSftJsonLinesDataset(dataset_path=file_path, format="alpaca")
        assert dataset._format == "alpaca"
        assert len(dataset) == 1

        conversation = dataset.conversation(0)
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert (
            conversation.messages[0].content
            == "Translate the following English text to French\n\nHello, how are you?"
        )
        assert conversation.messages[1].role == "assistant"
        assert conversation.messages[1].content == "Bonjour, comment allez-vous ?"


def test_auto_detect_format(sample_oumi_data, sample_alpaca_data):
    oumi_dataset = TextSftJsonLinesDataset(data=sample_oumi_data)
    assert oumi_dataset._format == "oumi"

    alpaca_dataset = TextSftJsonLinesDataset(data=sample_alpaca_data)
    assert alpaca_dataset._format == "alpaca"


def test_invalid_format():
    with pytest.raises(ValueError, match="Invalid format:"):
        TextSftJsonLinesDataset(data=[{"messages": []}], format="invalid_format")


def test_unsupported_file_extension():
    with tempfile.TemporaryDirectory() as folder:
        file_path = Path(folder) / "data.txt"
        file_path.touch()

        with pytest.raises(ValueError, match="Unsupported file format:"):
            TextSftJsonLinesDataset(dataset_path=file_path)


def test_invalid_data_structure():
    with pytest.raises(ValueError, match="Invalid data format"):
        TextSftJsonLinesDataset(data=[["not a dict"]])  # type: ignore[arg-type]


def test_undetectable_format():
    with pytest.raises(ValueError, match="Unable to auto-detect format"):
        TextSftJsonLinesDataset(data=[{"unknown_key": "value"}])


def test_alpaca_to_conversation():
    alpaca_turn = {
        "instruction": "Translate to French",
        "input": "Hello",
        "output": "Bonjour",
    }
    dataset = TextSftJsonLinesDataset(data=[alpaca_turn], format="alpaca")
    conversation = dataset.conversation(0)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == "user"
    assert conversation.messages[0].content == "Translate to French\n\nHello"
    assert conversation.messages[1].role == "assistant"
    assert conversation.messages[1].content == "Bonjour"


def test_alpaca_missing_keys():
    invalid_alpaca_turn = {
        "instruction": "Translate to French",
        "input": "Hello",
        # Missing "output" key
    }
    dataset = TextSftJsonLinesDataset(data=[invalid_alpaca_turn], format="alpaca")

    with pytest.raises(ValueError, match="Invalid Alpaca format"):
        dataset.conversation(0)


def test_oumi_and_alpaca_format_prompt_equality(sample_oumi_data, sample_alpaca_data):
    oumi_dataset = TextSftJsonLinesDataset(data=sample_oumi_data)
    alpaca_dataset = TextSftJsonLinesDataset(data=sample_alpaca_data)

    oumi_conversation = oumi_dataset.conversation(0)
    alpaca_conversation = alpaca_dataset.conversation(0)

    assert len(oumi_conversation.messages) == len(alpaca_conversation.messages)
    assert oumi_conversation.messages[0].role == alpaca_conversation.messages[0].role
    assert (
        oumi_conversation.messages[0].content == alpaca_conversation.messages[0].content
    )
    assert oumi_conversation.messages[1].role == alpaca_conversation.messages[1].role
    assert (
        oumi_conversation.messages[1].content == alpaca_conversation.messages[1].content
    )
