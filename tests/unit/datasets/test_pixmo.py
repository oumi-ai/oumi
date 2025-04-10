from unittest import mock

import pytest

from oumi.core.types.conversation import Conversation, Role
from oumi.datasets.vision_language import (
    PixmoAskModelAnythingDataset,
    PixmoCapDataset,
    PixmoCapQADataset,
)


@pytest.fixture
def sample_pixmo_ask_model_anything_entry():
    return {
        "image_url": "http://oumi.ai/test.png",
        "image_sha256": "1234567890",
        "question": "What type of machine is this?",
        "answer": "This is a vintage-style popcorn cart.",
    }


@pytest.fixture
def sample_pixmo_cap_entry():
    return {
        "image_url": "http://oumi.ai/test.png",
        "caption": "This photograph depicts a striking black bird",
        "transcripts": ["a", "b", "c"],
    }


@pytest.fixture
def sample_pixmo_cap_qa_entry():
    return {
        "image_url": "http://oumi.ai/test.png",
        "question": "[USER] Color? [ASSISTANT] Blue [USER] Time?[ASSISTANT]",
        "answer": "Noon",
    }


def test_pixmo_ask_model_anything_dataset(sample_pixmo_ask_model_anything_entry):
    with mock.patch.object(
        PixmoAskModelAnythingDataset, "__init__", return_value=None
    ) as _:
        dataset = PixmoAskModelAnythingDataset()
    conversation = dataset.transform_conversation(sample_pixmo_ask_model_anything_entry)
    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == Role.USER
    assert conversation.messages[1].role == Role.ASSISTANT


def test_pixmo_cap_dataset(sample_pixmo_cap_entry):
    with mock.patch.object(PixmoCapDataset, "__init__", return_value=None) as _:
        dataset = PixmoCapDataset()
    conversation = dataset.transform_conversation(sample_pixmo_cap_entry)
    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) >= 2
    assert conversation.messages[0].role == Role.USER
    assert conversation.messages[1].role == Role.ASSISTANT


def test_pixmo_cap_qa_dataset(sample_pixmo_cap_qa_entry):
    with mock.patch.object(PixmoCapQADataset, "__init__", return_value=None) as _:
        dataset = PixmoCapQADataset()
    conversation = dataset.transform_conversation(sample_pixmo_cap_qa_entry)
    assert isinstance(conversation, Conversation)
    assert len(conversation.messages) >= 3
    assert conversation.messages[0].role == Role.USER
    assert conversation.messages[1].role == Role.USER
    assert conversation.messages[2].role == Role.ASSISTANT
