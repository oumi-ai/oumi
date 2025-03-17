from unittest import mock

import pytest

from oumi.core.types.conversation import Conversation, Role
from oumi.datasets.sft.tulu3_sft_mixture import Tulu3MixtureDataset


@pytest.fixture
def sample_tulu_entry():
    return {
        "id": "an id",
        "messages": [
            {"role": "system", "content": "A system message"},
            {"role": "user", "content": "A user message"},
            {"role": "assistant", "content": "An assistant message"},
        ],
        "source": "A source",
    }


@pytest.fixture
def invalid_role_tulu_entry():
    return {
        "id": "an id",
        "messages": [
            {"role": "invalid", "content": "Not used in test"},
        ],
        "source": "A source",
    }


def test_tulu3_mixture_dataset(sample_tulu_entry):
    with mock.patch.object(Tulu3MixtureDataset, "__init__", return_value=None) as _:
        dataset = Tulu3MixtureDataset()
        conversation = dataset.transform_conversation(sample_tulu_entry)
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 3
        assert conversation.messages[0].role == Role.SYSTEM
        assert conversation.messages[1].role == Role.USER
        assert conversation.messages[2].role == Role.ASSISTANT
        for converted, original in zip(
            conversation.messages, sample_tulu_entry["messages"]
        ):
            assert converted.content == original["content"]


def test_tulu3_mixture_dataset_throws_valueerror_on_bad_role(invalid_role_tulu_entry):
    with mock.patch.object(Tulu3MixtureDataset, "__init__", return_value=None) as _:
        dataset = Tulu3MixtureDataset()
        with pytest.raises(ValueError):
            dataset.transform_conversation(invalid_role_tulu_entry)
