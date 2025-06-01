from unittest.mock import Mock

import numpy as np
import pytest
import torch

from oumi.core.tokenizers.utils import (
    find_token_sequence,
    mask_labels_for_completions_only,
)


# Token Sequence Finding Tests
@pytest.mark.parametrize(
    "sequence,target,expected",
    [
        ([1, 2, 3, 4, 5], [3, 4], 2),
        ([1, 2, 3, 4, 5], [1, 2], 0),
        ([1, 2, 3, 4, 5], [4, 5], 3),
        ([1, 2, 3, 4, 5], [6, 7], None),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0),
        ([], [1], None),
    ],
)
def test_find_token_sequence(sequence, target, expected):
    assert find_token_sequence(sequence, target) == expected


def test_find_token_sequence_with_tensor():
    tensor = torch.tensor([1, 2, 3, 4, 5])
    assert find_token_sequence(tensor, [3, 4]) == 2


def test_find_token_sequence_with_numpy():
    arr = np.array([1, 2, 3, 4, 5])
    assert find_token_sequence(arr, [3, 4]) == 2


# Mask Labels Tests
def test_basic_masking():
    labels = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    response_tokens = [3, 4]

    mask_labels_for_completions_only(labels, response_tokens)

    # Should mask [1, 2, 3, 4] and keep [5, 6, 7, 8]
    expected = np.array([-100, -100, -100, -100, 5, 6, 7, 8])
    np.testing.assert_array_equal(labels, expected)


def test_no_response_template_found():
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [9, 10]

    mask_labels_for_completions_only(labels, response_tokens)

    # Should mask everything
    expected = np.array([-100, -100, -100, -100, -100])
    np.testing.assert_array_equal(labels, expected)


def test_custom_ignore_index():
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [2, 3]

    mask_labels_for_completions_only(labels, response_tokens, ignore_index=-999)

    expected = np.array([-999, -999, -999, 4, 5])
    np.testing.assert_array_equal(labels, expected)


def test_response_at_start():
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [1, 2]

    mask_labels_for_completions_only(labels, response_tokens)

    # Should only mask the response template itself
    expected = np.array([-100, -100, 3, 4, 5])
    np.testing.assert_array_equal(labels, expected)


# Feature Generator Tests
@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.encode = Mock(
        side_effect=lambda text, **kwargs: {
            "Assistant:": [100, 101],
            "User:": [200, 201],
        }.get(text, [])
    )
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_processor(mock_tokenizer):
    processor = Mock()
    processor.tokenizer = mock_tokenizer
    processor.chat_template = None
    return processor


@pytest.fixture
def feature_generator(mock_processor, mock_tokenizer):
    from oumi.core.feature_generators.vision_language_conversation_feature_generator import (
        VisionLanguageConversationFeatureGenerator,
    )

    fg = VisionLanguageConversationFeatureGenerator.__new__(
        VisionLanguageConversationFeatureGenerator
    )
    fg._processor = mock_processor
    fg._tokenizer = mock_tokenizer
    fg._train_on_completions_only = True
    fg._response_template = "Assistant:"
    fg._response_token_ids = [100, 101]
    fg._special_tokens = Mock(label_ignore_index=-100)

    return fg


def test_apply_completion_only_masking_list(feature_generator):
    inputs = {
        "labels": [[1, 2, 100, 101, 3, 4, 5], [10, 100, 101, 20, 30]],
        "input_ids": [[1, 2, 100, 101, 3, 4, 5], [10, 100, 101, 20, 30]],
    }

    feature_generator._apply_completion_only_masking(inputs)

    expected = [[-100, -100, -100, -100, 3, 4, 5], [-100, -100, -100, 20, 30]]
    assert inputs["labels"] == expected


def test_apply_completion_only_masking_tensor(feature_generator):
    labels = torch.tensor([[1, 2, 100, 101, 3, 4, 5]])
    inputs = {"labels": labels}

    feature_generator._apply_completion_only_masking(inputs)

    expected = torch.tensor([[-100, -100, -100, -100, 3, 4, 5]])
    torch.testing.assert_close(inputs["labels"], expected)


def test_apply_completion_only_masking_numpy(feature_generator):
    inputs = {
        "labels": np.array([[1, 2, 100, 101, 3, 4, 5]]),
    }

    feature_generator._apply_completion_only_masking(inputs)

    expected = np.array([[-100, -100, -100, -100, 3, 4, 5]])
    np.testing.assert_array_equal(inputs["labels"], expected)


# Multi-turn Conversation Tests
@pytest.fixture
def feature_generator_multiturn():
    from oumi.core.feature_generators.vision_language_conversation_feature_generator import (
        VisionLanguageConversationFeatureGenerator,
    )

    fg = VisionLanguageConversationFeatureGenerator.__new__(
        VisionLanguageConversationFeatureGenerator
    )
    fg._response_token_ids = [100, 101]  # "Assistant:"
    fg._user_token_ids = [200, 201]  # "User:"
    fg._special_tokens = Mock(label_ignore_index=-100)

    return fg


def test_find_all_template_positions(feature_generator_multiturn):
    input_ids = np.array([1, 100, 101, 2, 3, 100, 101, 4, 5])
    positions = feature_generator_multiturn._find_all_template_positions(
        input_ids, [100, 101]
    )
    assert positions == [3, 7]  # Positions after the template


def test_mask_single_conversation_multiple_turns(feature_generator_multiturn):
    # User: hi [200,201,10] Assistant: hello [100,101,20] User: bye [200,201,30] Assistant: farewell [100,101,40]
    input_ids = np.array([200, 201, 10, 100, 101, 20, 200, 201, 30, 100, 101, 40])
    labels = np.array([200, 201, 10, 100, 101, 20, 200, 201, 30, 100, 101, 40])

    feature_generator_multiturn._mask_single_conversation(labels, input_ids)

    # Should mask everything except assistant responses (20 and 40)
    expected = np.array(
        [-100, -100, -100, -100, -100, 20, -100, -100, -100, -100, -100, 40]
    )
    np.testing.assert_array_equal(labels, expected)


def test_mask_single_conversation_no_user_template(feature_generator_multiturn):
    # Remove user template info
    feature_generator_multiturn._user_token_ids = None

    input_ids = np.array([1, 2, 100, 101, 3, 4, 5])
    labels = np.array([1, 2, 100, 101, 3, 4, 5])

    feature_generator_multiturn._mask_single_conversation(labels, input_ids)

    # Should mask everything before first assistant response
    expected = np.array([-100, -100, -100, -100, 3, 4, 5])
    np.testing.assert_array_equal(labels, expected)


# Edge Case Tests
def test_empty_response_template():
    labels = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        mask_labels_for_completions_only(labels, [])


def test_empty_labels():
    labels = np.array([])
    mask_labels_for_completions_only(labels, [1, 2])
    assert len(labels) == 0


def test_response_template_longer_than_sequence():
    labels = np.array([1, 2])
    response_tokens = [1, 2, 3, 4, 5]

    mask_labels_for_completions_only(labels, response_tokens)

    # Should mask everything
    expected = np.array([-100, -100])
    np.testing.assert_array_equal(labels, expected)


# Integration Tests
def test_collator_initialization_validation():
    from oumi.core.collators.vision_language_sft_collator import (
        VisionLanguageSftCollator,
    )

    # Should raise error when train_on_completions_only=True without response_template
    with pytest.raises(ValueError, match="response_template must be provided"):
        VisionLanguageSftCollator(
            processor=Mock(),
            tokenizer=Mock(),
            train_on_completions_only=True,
            response_template=None,
        )

    # Should succeed with both parameters
    collator = VisionLanguageSftCollator(
        processor=Mock(),
        tokenizer=Mock(),
        train_on_completions_only=True,
        response_template="Assistant:",
    )
    assert collator is not None


def test_validate_conversations_for_completion_only_training(feature_generator):
    from oumi.core.types.conversation import Conversation, Message, Role

    # Valid conversation
    valid_conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there"),
        ]
    )

    # Should not raise
    feature_generator._validate_conversations_for_completion_only_training([valid_conv])

    # Invalid: wrong number of messages
    invalid_conv1 = Conversation(messages=[Message(role=Role.USER, content="Hello")])

    with pytest.raises(ValueError, match="exactly 2 messages"):
        feature_generator._validate_conversations_for_completion_only_training(
            [invalid_conv1]
        )

    # Invalid: wrong role order
    invalid_conv2 = Conversation(
        messages=[
            Message(role=Role.ASSISTANT, content="Hi"),
            Message(role=Role.USER, content="Hello"),
        ]
    )

    with pytest.raises(ValueError, match="First message must be from USER"):
        feature_generator._validate_conversations_for_completion_only_training(
            [invalid_conv2]
        )


def test_multiple_response_templates_in_sequence():
    # Test when response template appears multiple times
    labels = np.array([1, 2, 3, 4, 5, 3, 4, 6, 7])
    response_tokens = [3, 4]

    mask_labels_for_completions_only(labels, response_tokens)

    # Should only mask up to first occurrence
    expected = np.array([-100, -100, -100, -100, 5, 3, 4, 6, 7])
    np.testing.assert_array_equal(labels, expected)


def test_response_template_with_warning(caplog):
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [9, 10]

    mask_labels_for_completions_only(
        labels, response_tokens, response_template="Assistant:"
    )

    # Check warning was logged
    assert "Could not find response template" in caplog.text
