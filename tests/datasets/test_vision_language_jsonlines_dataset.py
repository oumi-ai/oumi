import pytest
from pandas.core.api import DataFrame as DataFrame

from lema.core.types.turn import Conversation, Type
from lema.datasets.vision_language.vision_jsonlines import JsonlinesDataset


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
    assert "messages" in dataset._data.columns


def test_jsonlines_transform_conversation(sample_jsonlines_data):
    dataset = JsonlinesDataset(data=sample_jsonlines_data)
    conversation = dataset.transform_conversation(sample_jsonlines_data[0])

    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 3
    assert conversation.messages[0].content == "Describe this image:"
    assert conversation.messages[1].content == "path/to/image.jpg"
    assert conversation.messages[1].type == Type.IMAGE_PATH
    assert conversation.messages[2].content == "A scenic view of the puget sound."


def test_jsonlines_init_with_invalid_input():
    with pytest.raises(ValueError, match="Dataset path or data must be provided"):
        JsonlinesDataset()

    with pytest.raises(
        ValueError, match="Dataset path does not exist: invalid_path.txt"
    ):
        JsonlinesDataset(dataset_path="invalid_path.txt")

    with pytest.raises(
        ValueError, match="Either dataset_path or data must be provided, but not both"
    ):
        JsonlinesDataset(dataset_path="valid_path.jsonl", data=[])
