import tempfile
from pathlib import Path

import jsonlines
import pytest
from pandas.core.api import DataFrame as DataFrame

from oumi.core.types.turn import Conversation, Type
from oumi.datasets.vision_language.vision_jsonlines import JsonlinesDataset


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


def test_jsonlines_init_with_data(sample_jsonlines_data):
    dataset = JsonlinesDataset(data=sample_jsonlines_data)
    assert len(dataset._data) == 1
    assert ["messages"] == dataset._data.columns


def test_jsonlines_init_with_data_custom_data_column(sample_jsonlines_data):
    dataset = JsonlinesDataset(data=sample_jsonlines_data, data_column="foo")
    assert len(dataset._data) == 1
    assert ["foo"] == dataset._data.columns


def test_jsonlines_init_with_dataset_path(sample_jsonlines_data):
    with tempfile.TemporaryDirectory() as folder:
        vaild_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(vaild_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        dataset = JsonlinesDataset(dataset_path=vaild_jsonlines_filename)
        assert len(dataset._data) == 1
        assert ["messages"] == dataset._data.columns


def test_jsonlines_transform_conversation(sample_jsonlines_data):
    dataset = JsonlinesDataset(data=sample_jsonlines_data)
    conversation = dataset.conversation(0)

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 3
    assert conversation.messages[0].content == "Describe this image:"
    assert conversation.messages[1].content == "path/to/image.jpg"
    assert conversation.messages[1].type == Type.IMAGE_PATH
    assert conversation.messages[2].content == "A scenic view of the puget sound."


def test_jsonlines_init_with_invalid_input(sample_jsonlines_data):
    with tempfile.TemporaryDirectory() as folder:
        vaild_jsonlines_filename = Path(folder) / "valid_path.jsonl"
        with jsonlines.open(vaild_jsonlines_filename, mode="w") as writer:
            writer.write_all(sample_jsonlines_data)

        with pytest.raises(ValueError, match="Dataset path or data must be provided"):
            JsonlinesDataset()

        with pytest.raises(ValueError, match="Dataset path or data must be provided"):
            JsonlinesDataset(data_column="some_column_name")

        with pytest.raises(
            ValueError, match="Dataset path does not exist: invalid_path.jsonl"
        ):
            JsonlinesDataset(dataset_path="invalid_path.jsonl")

        with pytest.raises(
            ValueError,
            match="Either dataset_path or data must be provided, but not both",
        ):
            JsonlinesDataset(dataset_path=vaild_jsonlines_filename, data=[])

        with pytest.raises(
            ValueError,
            match="Dataset path must end with .jsonl",
        ):
            JsonlinesDataset(dataset_path="invalid_extension.json")

        # Directory ending with .jsonl
        temp_dir_name = Path(folder) / "subdir.jsonl"
        temp_dir_name.mkdir()
        with pytest.raises(
            ValueError,
            match="Dataset path is not a file",
        ):
            JsonlinesDataset(dataset_path=temp_dir_name)

        with pytest.raises(ValueError, match="Data column not found in dataset"):
            JsonlinesDataset(dataset_path=vaild_jsonlines_filename, data_column="foo")
