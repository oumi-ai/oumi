import pytest

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
def sample_conversation():
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
                        content="tests/testdata/images/oumi_logo_dark.png",
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


def test_vision_language_completions_only_with_phi3(
    phi3_tokenizer, sample_conversation
):
    """Test vision language collator with completions-only training enabled."""
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
    assert "input_ids" in result
    assert "labels" in result
    assert "attention_mask" in result
    assert "pixel_values" in result

    # Check that labels are properly masked
    labels = result["labels"][0]

    # Count non-ignored tokens
    non_ignored_mask = labels != LABEL_IGNORE_INDEX
    non_ignored_count = non_ignored_mask.sum().item()
    total_count = len(labels)

    # Should have some tokens masked and some not masked
    assert non_ignored_count > 0, (
        "Should have some non-ignored tokens (assistant response)"
    )
    assert non_ignored_count < total_count, (
        "Should have some ignored tokens (user prompt)"
    )

    # Check that tokens before assistant response are masked
    first_non_ignored_idx = -1
    for i in range(len(labels)):
        if labels[i].item() != LABEL_IGNORE_INDEX:
            first_non_ignored_idx = i
            break

    assert first_non_ignored_idx > 0, "Should find non-ignored tokens"

    # Verify that all tokens before the first non-ignored are masked
    assert all(labels[:first_non_ignored_idx] == LABEL_IGNORE_INDEX), (
        "All tokens before assistant response should be masked"
    )

    # Verify that all tokens after the first non-ignored not masked
    assert all(labels[first_non_ignored_idx + 1 :] != LABEL_IGNORE_INDEX), (
        "All tokens after assistant response should be masked"
    )


def test_vision_language_without_completions_only(phi3_tokenizer, sample_conversation):
    """Test vision language collator without completions-only training."""
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

    labels = result["labels"][0]

    # Without completions-only, most tokens should not be masked
    # (except for special tokens like image placeholders)
    non_ignored_mask = labels != LABEL_IGNORE_INDEX
    non_ignored_count = non_ignored_mask.sum().item()
    total_count = len(labels)

    # Most tokens should be unmasked
    assert non_ignored_count == total_count


def test_vision_language_completions_only_missing_template(
    phi3_tokenizer, sample_conversation
):
    """Test that response template not found is handled gracefully."""
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

    # When template is not found, all tokens should be masked
    labels = result["labels"][0]
    assert all(labels == LABEL_IGNORE_INDEX), (
        "When response template is not found, all tokens should be masked"
    )


def test_vision_language_completions_only_multiple_turns(phi3_tokenizer):
    """Test completions-only with invalid multi-turn conversation."""
    # Create a multi-turn conversation (not supported for completions-only)
    multi_turn_conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[ContentItem(type=Type.TEXT, content="First question")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=[ContentItem(type=Type.TEXT, content="First answer")],
            ),
            Message(
                role=Role.USER,
                content=[ContentItem(type=Type.TEXT, content="Second question")],
            ),
        ]
    )

    # Create collator with completions-only training
    collator = build_data_collator(
        collator_name="vision_language_sft",
        tokenizer=phi3_tokenizer,
        processor_name="microsoft/Phi-3-vision-128k-instruct",
        train_on_completions_only=True,
        response_template="<|assistant|>",
        trust_remote_code=True,
        max_length=512,
    )

    # Process batch should raise an error
    batch = [{"conversation_json": multi_turn_conversation.to_json()}]

    with pytest.raises(ValueError, match="exactly 2 messages"):
        collator(batch)


def test_vision_language_completions_only_with_instruction_template(
    phi3_tokenizer, sample_conversation
):
    """Test vision language collator with both response and instruction templates."""
    # Create collator with both templates
    collator = build_data_collator(
        collator_name="vision_language_sft",
        tokenizer=phi3_tokenizer,
        processor_name="microsoft/Phi-3-vision-128k-instruct",
        train_on_completions_only=True,
        response_template="<|assistant|>",
        instruction_template="<|user|>",  # Optional instruction template
        trust_remote_code=True,
        max_length=512,
    )

    # Process batch
    batch = [{"conversation_json": sample_conversation.to_json()}]
    result = collator(batch)

    # Verify that masking still works correctly
    labels = result["labels"][0]
    non_ignored_mask = labels != LABEL_IGNORE_INDEX
    non_ignored_count = non_ignored_mask.sum().item()

    assert non_ignored_count > 0, "Should have some non-ignored tokens"
    assert non_ignored_count < len(labels), "Should have some ignored tokens"
