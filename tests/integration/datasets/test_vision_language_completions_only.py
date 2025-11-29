import pytest
import torch

from oumi.builders import build_data_collator, build_tokenizer
from oumi.core.configs import ModelParams
from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.types import ContentItem, Conversation, Message, Role
from oumi.core.types.conversation import Type


@pytest.fixture
def phi3_tokenizer():
    """Create Phi-3 tokenizer for testing."""
    model_params = ModelParams(
        model_name="microsoft/Phi-3-vision-128k-instruct",
        device_map="cpu",
        trust_remote_code=True,
        chat_template="phi3-instruct",
    )
    return build_tokenizer(model_params)


@pytest.fixture
def sample_conversation(root_testdata_dir):
    """Create a sample vision-language conversation."""
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(
                        type=Type.TEXT, content="What objects are in this image?"
                    ),
                    ContentItem(
                        type=Type.IMAGE_PATH,
                        content=str(root_testdata_dir / "images/oumi_logo_dark.png"),
                    ),
                ],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ContentItem(
                        type=Type.TEXT,
                        content="This image shows the Oumi logo with stylized text.",
                    )
                ],
            ),
        ]
    )


def test_phi3_tokenization(phi3_tokenizer):
    """Test that we understand Phi-3's tokenization behavior correctly."""
    # Known tokenization from our analysis
    response_template = "<|assistant|>"
    instruction_template = "<|user|>"

    response_tokens = phi3_tokenizer.encode(response_template, add_special_tokens=False)
    instruction_tokens = phi3_tokenizer.encode(
        instruction_template, add_special_tokens=False
    )

    # Verify expected token values
    assert response_tokens == [32001], f"Expected [32001], got {response_tokens}"
    assert instruction_tokens == [32010], f"Expected [32010], got {instruction_tokens}"

    # Test full conversation tokenization
    messages = [
        {"role": "user", "content": "What is this?"},
        {"role": "assistant", "content": "This is a test."},
    ]

    prompt = phi3_tokenizer.apply_chat_template(messages, tokenize=False)
    expected_prompt = (
        "<|user|>\nWhat is this?<|end|>\n<|assistant|>\nThis is a test.<|end|>\n"
    )
    assert prompt == expected_prompt, (
        f"Expected {repr(expected_prompt)}, got {repr(prompt)}"
    )

    tokens = phi3_tokenizer.encode(prompt, add_special_tokens=False)
    expected_tokens = [
        32010,
        29871,
        13,
        5618,
        338,
        445,
        29973,
        32007,
        29871,
        13,
        32001,
        910,
        338,
        263,
        1243,
        29889,
        32007,
        29871,
        13,
    ]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"


def test_vision_language_completions_only(phi3_tokenizer, sample_conversation):
    """Test vision language collator with exact token-level validation."""
    # Create collator with completions-only training
    collator = build_data_collator(
        collator_name="vision_language_sft",
        tokenizer=phi3_tokenizer,
        processor_name="microsoft/Phi-3-vision-128k-instruct",
        train_on_completions_only=True,
        response_template="<|assistant|>",
        instruction_template="<|user|>",
        trust_remote_code=True,
        max_length=512,
    )

    # Process batch
    batch = [{"conversation_json": sample_conversation.to_json()}]
    result = collator(batch)

    # Verify result structure
    required_keys = ["input_ids", "labels", "attention_mask", "pixel_values"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], torch.Tensor), f"{key} is not a tensor"

    # Extract tensors
    input_ids = result["input_ids"][0]
    labels = result["labels"][0]
    attention_mask = result["attention_mask"][0]

    # Verify shapes match
    assert input_ids.shape == labels.shape == attention_mask.shape

    # Find template positions in the tokenized sequence
    input_ids_list = input_ids.tolist()
    labels_list = labels.tolist()

    # Find assistant template position (token 32001)
    assistant_positions = [
        i for i, token in enumerate(input_ids_list) if token == 32001
    ]
    assert len(assistant_positions) == 1, (
        f"Expected 1 assistant template, found {len(assistant_positions)}"
    )

    assistant_pos = assistant_positions[0]

    # Validate masking pattern
    # Everything up to and including the assistant template should be masked
    for i in range(assistant_pos + 1):
        assert labels_list[i] == LABEL_IGNORE_INDEX, (
            f"Token at position {i} (value: {input_ids_list[i]}) should be masked, "
            f"but label is {labels_list[i]}"
        )

    # At least some tokens after the assistant template should be unmasked
    unmasked_count = sum(
        1
        for i in range(assistant_pos + 1, len(labels_list))
        if labels_list[i] != LABEL_IGNORE_INDEX
    )
    assert unmasked_count > 0, "No assistant response tokens are unmasked"

    # Verify that unmasked labels match input_ids
    for i in range(len(labels_list)):
        if labels_list[i] != LABEL_IGNORE_INDEX:
            assert labels_list[i] == input_ids_list[i], (
                f"Unmasked label at position {i} ({labels_list[i]}) "
                f"doesn't match input_id ({input_ids_list[i]})"
            )


def test_vision_language_without_completions_only(phi3_tokenizer, sample_conversation):
    # Create collator without completions-only training
    collator = build_data_collator(
        collator_name="vision_language_sft",
        tokenizer=phi3_tokenizer,
        processor_name="microsoft/Phi-3-vision-128k-instruct",
        train_on_completions_only=False,
        trust_remote_code=True,
        max_length=512,
    )

    # Process batch
    batch = [{"conversation_json": sample_conversation.to_json()}]
    result = collator(batch)

    # Verify result structure
    assert "input_ids" in result
    assert "labels" in result

    input_ids = result["input_ids"][0].tolist()
    labels = result["labels"][0].tolist()

    # Verify that input_ids and labels have the same shape
    assert len(input_ids) == len(labels)

    # Verify that unmasked labels match input_ids
    # OR image tokens are masked
    for i, (input_id, label) in enumerate(zip(input_ids, labels)):
        if label != LABEL_IGNORE_INDEX:
            assert label == input_id or label == 32000 and input_id == -1, (
                f"Position {i}: unmasked label {label} doesn't match "
                f"input_id {input_id}"
            )


def test_vision_language_completions_only_wrong_template(
    phi3_tokenizer, sample_conversation
):
    """Test exact behavior when response template is not found."""
    # Create collator with a non-existent response template
    collator = build_data_collator(
        collator_name="vision_language_sft",
        tokenizer=phi3_tokenizer,
        processor_name="microsoft/Phi-3-vision-128k-instruct",
        train_on_completions_only=True,
        response_template="<|nonexistent|>",  # This template won't be found
        trust_remote_code=True,
        max_length=512,
    )

    # Process batch
    batch = [{"conversation_json": sample_conversation.to_json()}]
    result = collator(batch)

    # When template is not found, ALL tokens should be masked
    labels = result["labels"][0].tolist()
    input_ids = result["input_ids"][0].tolist()

    # Verify every single token is masked
    for i, label in enumerate(labels):
        assert label == LABEL_IGNORE_INDEX, (
            f"Position {i} (token {input_ids[i]}) should be masked when "
            f"template not found, but label is {label}"
        )

    # Verify the count
    masked_count = sum(1 for label in labels if label == LABEL_IGNORE_INDEX)
    assert masked_count == len(labels), (
        f"Expected all {len(labels)} tokens to be masked, but only "
        f"{masked_count} are masked"
    )
