from collections.abc import Sequence

import pytest
from transformers import AutoTokenizer

from oumi.core.types.turn import Conversation, Message
from oumi.datasets import ArgillaDollyDataset


@pytest.fixture
def dolly_dataset():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return ArgillaDollyDataset(split="train", tokenizer=tokenizer)


@pytest.skip(
    "This test is very time consuming, and should be run manually.",
    allow_module_level=True,
)
def test_dolly_dataset_model_inputs(dolly_dataset):
    # Check that the dataset is not empty
    assert len(dolly_dataset) > 0

    # Iterate through all items in the dataset
    for idx in range(len(dolly_dataset)):
        item = dolly_dataset[idx]

        # Check that each item has the expected keys
        assert "input_ids" in item
        assert isinstance(item["input_ids"], Sequence)


@pytest.skip(
    "This test is very time consuming, and should be run manually.",
    allow_module_level=True,
)
def test_dolly_dataset_conversation(dolly_dataset):
    # Check that the dataset is not empty
    assert len(dolly_dataset) > 0

    # Iterate through all items in the dataset
    for idx in range(len(dolly_dataset)):
        # Check the conversation structure
        conversation = dolly_dataset.conversation(idx)
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) > 0

        # Check that each message in the conversation has the expected structure
        for message in conversation.messages:
            assert isinstance(message, Message)
            assert message.role in ["user", "assistant"]
            assert isinstance(message.content, str)
            assert len(message.content) > 0
            assert message.type == "text"

        # Check that the first message is from the user
        assert conversation.messages[0].role == "user"
        assert conversation.messages[-1].role == "assistant"


@pytest.skip(
    "This test is very time consuming, and should be run manually.",
    allow_module_level=True,
)
def test_dolly_dataset_prompt_generation(dolly_dataset):
    for idx in range(len(dolly_dataset)):
        prompt = dolly_dataset.prompt(idx)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
