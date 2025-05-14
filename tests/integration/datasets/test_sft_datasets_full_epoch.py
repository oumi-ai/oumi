from collections.abc import Sequence

import pytest
from transformers import AutoTokenizer

from oumi.core.registry import REGISTRY
from oumi.core.types.conversation import Conversation, Message


def is_content_empty_expected(dataset_name, conversation_idx, message_idx):
    """Determine if the content of a message is expected to be empty.

    In 99.999% of cases, no message should have empty content. However there are
    some known cases where the content is expected to be empty. This function
    contains a hard-coded list of such known cases.
    """
    known_empty_messages = {
        # This is the answer to the prompt: "write 5 empty sentences"
        ("Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1", 318135, 1),
    }
    return (dataset_name, conversation_idx, message_idx) in known_empty_messages


@pytest.fixture(
    params=[
        "Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1",
        "Magpie-Align/Magpie-Pro-300K-Filtered",
        "argilla/magpie-ultra-v0.1",
        "argilla/databricks-dolly-15k-curated-en",
    ]
)
def dataset_fixture(request):
    dataset_name = request.param
    dataset_class = REGISTRY.get_dataset(dataset_name)
    if dataset_class is None:
        pytest.fail(f"Dataset {dataset_name} not found in registry")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return dataset_name, dataset_class(
        dataset_name=dataset_name, split="train", tokenizer=tokenizer
    )


@pytest.mark.e2e
def test_dataset_conversation(dataset_fixture):
    dataset_name, dataset = dataset_fixture
    assert len(dataset) > 0, f"Dataset {dataset_name} is empty"

    # Iterate through all items in the dataset
    for idx in range(len(dataset)):
        conversation = dataset.conversation(idx)

        # Check the conversation structure
        assert isinstance(conversation, Conversation), (
            f"Conversation at index {idx} is not a Conversation object. "
            f"Type: {type(conversation)}"
        )
        assert len(conversation.messages) > 0, (
            f"Conversation at index {idx} has no messages"
        )

        # Check that each message in the conversation has the expected structure
        for msg_idx, message in enumerate(conversation.messages):
            assert isinstance(message, Message)
            assert message.role in ["user", "assistant"], (
                f"Invalid role for message {msg_idx} in conversation at index {idx}. "
                f"Role: {message.role}"
            )
            assert isinstance(message.content, str)

            if is_content_empty_expected(dataset_name, idx, msg_idx):
                assert message.content == "", (
                    f"Content of message {msg_idx} in conversation at index {idx} "
                    f"is not empty. Content: {message.content}"
                )
            else:
                assert len(message.content) > 0, (
                    f"Content of message {msg_idx} in conversation "
                    f"at index {idx} is empty"
                )
            assert isinstance(message.content, str), (
                f"Type of message {msg_idx} in conversation at index {idx} "
                f"is not 'text'. Type: {type(message.content)}"
            )

        assert conversation.messages[0].role == "user", (
            f"First message in conversation at index {idx} is not from user. "
            f"Role: {conversation.messages[0].role}"
        )
        assert conversation.messages[-1].role == "assistant", (
            f"Last message in conversation at index {idx} is not from assistant. "
            f"Role: {conversation.messages[-1].role}"
        )


@pytest.mark.skip(
    reason="This test is very time consuming, and should be run manually."
)
def test_dataset_prompt_generation(dataset_fixture):
    dataset_name, dataset = dataset_fixture
    assert len(dataset) > 0, f"Dataset {dataset_name} is empty"

    for idx in range(len(dataset)):
        prompt = dataset.prompt(idx)
        assert isinstance(prompt, str), (
            f"Prompt at index {idx} is not a string. Type: {type(prompt)}"
        )
        assert len(prompt) > 0, f"Prompt at index {idx} is empty"


@pytest.mark.skip(
    reason="This test is very time consuming, and should be run manually."
)
def test_dataset_model_inputs(dataset_fixture):
    dataset_name, dataset = dataset_fixture
    assert len(dataset) > 0, f"Dataset {dataset_name} is empty"

    # Iterate through all items in the dataset
    for idx in range(len(dataset)):
        item = dataset[idx]

        # Check that each item has the expected keys
        assert "input_ids" in item, f"'input_ids' not found in item at index {idx}"
        assert isinstance(item["input_ids"], Sequence), (
            f"'input_ids' is not a Sequence at index {idx}. "
            f"Type: {type(item['input_ids'])}"
        )
