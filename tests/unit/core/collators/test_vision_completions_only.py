from unittest.mock import Mock

import numpy as np
import pytest

from oumi.core.tokenizers.utils import (
    find_all_sequences,
    mask_labels_for_completions_only,
    mask_labels_without_user_template,
)


# Find All Sequences Tests
@pytest.mark.parametrize(
    "sequence,target,expected",
    [
        ([1, 2, 3, 4, 5], [3, 4], [4]),
        ([1, 2, 3, 4, 5], [1, 2], [2]),
        ([1, 2, 3, 4, 5], [4, 5], [5]),
        ([1, 2, 3, 4, 5], [6, 7], []),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [5]),
        ([], [1], []),
        ([1, 2, 3, 4, 3, 4, 5], [3, 4], [4, 6]),  # Multiple occurrences
    ],
)
def test_find_all_sequences(sequence, target, expected):
    arr = np.array(sequence)
    result = find_all_sequences(arr, target)
    assert result == expected


# Mask Labels Tests - Basic scenarios
def test_basic_masking_no_user_template():
    """Test basic masking without user template (last assistant turn only strategy)."""
    labels = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    response_tokens = [3, 4]

    mask_labels_without_user_template(labels, response_tokens)

    # Should mask everything except the last assistant response content: [5, 6, 7, 8]
    # Template [3, 4] is masked, only content after it [5, 6, 7, 8] is kept
    expected = np.array([-100, -100, -100, -100, 5, 6, 7, 8])
    np.testing.assert_array_equal(labels, expected)


def test_masking_with_user_template():
    """Test masking with both user and assistant templates."""
    # Conversation: User: [200, 201, 10] Assistant: [100, 101, 20, 21]
    # User: [200, 201, 30] Assistant: [100, 101, 40, 41]
    labels = np.array([200, 201, 10, 100, 101, 20, 21, 200, 201, 30, 100, 101, 40, 41])
    response_tokens = [100, 101]  # Assistant template
    instruction_tokens = [200, 201]  # User template

    mask_labels_for_completions_only(labels, response_tokens, instruction_tokens)

    # Should mask everything except assistant responses: [20, 21] and [40, 41]
    # (templates are masked)
    expected = np.array(
        [-100, -100, -100, -100, -100, 20, 21, -100, -100, -100, -100, -100, 40, 41]
    )
    np.testing.assert_array_equal(labels, expected)


def test_no_response_template_found():
    """Test when response template is not found."""
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [9, 10]

    mask_labels_without_user_template(labels, response_tokens)

    # Should mask everything
    expected = np.array([-100, -100, -100, -100, -100])
    np.testing.assert_array_equal(labels, expected)


def test_custom_ignore_index():
    """Test using custom ignore index."""
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [2, 3]

    mask_labels_without_user_template(labels, response_tokens, ignore_index=-999)

    # Should mask [1, 2, 3] (including template) and keep [4, 5]
    expected = np.array([-999, -999, -999, 4, 5])
    np.testing.assert_array_equal(labels, expected)


def test_response_at_start():
    """Test when response template is at the beginning."""
    labels = np.array([1, 2, 3, 4, 5])
    response_tokens = [1, 2]

    mask_labels_without_user_template(labels, response_tokens)

    # Should mask the template [1, 2] and keep only the last response content [3, 4, 5]
    expected = np.array([-100, -100, 3, 4, 5])
    np.testing.assert_array_equal(labels, expected)


def test_multiple_responses_no_user_template():
    """Test multiple response templates without user template."""
    labels = np.array([1, 2, 3, 4, 5, 3, 4, 6, 7])
    response_tokens = [3, 4]

    mask_labels_without_user_template(labels, response_tokens)

    # Should mask everything except the last assistant response: [6, 7]
    # First response [5] should be masked since we only train on the last assistant turn
    expected = np.array([-100, -100, -100, -100, -100, -100, -100, 6, 7])
    np.testing.assert_array_equal(labels, expected)


def test_single_turn_conversation():
    """Test single-turn conversation with user and assistant templates."""
    # User: [200, 201, 10, 11] Assistant: [100, 101, 20, 21, 22]
    labels = np.array([200, 201, 10, 11, 100, 101, 20, 21, 22])
    response_tokens = [100, 101]
    instruction_tokens = [200, 201]

    mask_labels_for_completions_only(labels, response_tokens, instruction_tokens)

    # Should mask everything except assistant response content: [20, 21, 22]
    # (template [100, 101] is masked)
    expected = np.array([-100, -100, -100, -100, -100, -100, 20, 21, 22])
    np.testing.assert_array_equal(labels, expected)


# Simpler Feature Generator Fixture
@pytest.fixture
def simple_feature_generator():
    """Simple mock feature generator for testing completion-only masking."""
    fg = Mock()
    fg._response_token_ids = [100, 101]  # "Assistant:"
    fg._instruction_token_ids = [200, 201]  # "User:"

    # Mock the special tokens
    special_tokens = Mock()
    special_tokens.label_ignore_index = -100
    fg._special_tokens = special_tokens

    # Import the actual masking methods we need to test
    from oumi.core.feature_generators.vision_language_conversation_feature_generator import (  # noqa: E501
        VisionLanguageConversationFeatureGenerator,
    )

    # Bind the actual methods to our mock
    fg_class = VisionLanguageConversationFeatureGenerator
    fg._mask_single_conversation = fg_class._mask_single_conversation.__get__(fg)
    fg._apply_completion_only_masking = fg_class._apply_completion_only_masking.__get__(
        fg
    )

    return fg


def test_find_all_template_positions(simple_feature_generator):
    """Test finding all template positions in sequence."""
    from oumi.core.tokenizers.utils import find_all_sequences

    input_ids = np.array([1, 100, 101, 2, 3, 100, 101, 4, 5])
    positions = find_all_sequences(input_ids, [100, 101])
    assert positions == [3, 7]  # Positions after the template


def test_mask_single_conversation_with_user_template(simple_feature_generator):
    """Test masking single conversation with user template."""
    # User: [200, 201, 10] Assistant: [100, 101, 20]
    # User: [200, 201, 30] Assistant: [100, 101, 40]
    input_ids = np.array([200, 201, 10, 100, 101, 20, 200, 201, 30, 100, 101, 40])
    labels = np.array([200, 201, 10, 100, 101, 20, 200, 201, 30, 100, 101, 40])

    simple_feature_generator._mask_single_conversation(labels, input_ids)

    # Should mask everything except assistant response content: 20 and 40
    # (templates are masked)
    expected = np.array(
        [-100, -100, -100, -100, -100, 20, -100, -100, -100, -100, -100, 40]
    )
    np.testing.assert_array_equal(labels, expected)


def test_mask_single_conversation_no_user_template(simple_feature_generator):
    """Test masking single conversation without user template."""
    # Remove user template info
    simple_feature_generator._instruction_token_ids = None

    input_ids = np.array([1, 2, 100, 101, 3, 4, 5])
    labels = np.array([1, 2, 100, 101, 3, 4, 5])

    simple_feature_generator._mask_single_conversation(labels, input_ids)

    # Should mask everything including template and keep only response content
    expected = np.array([-100, -100, -100, -100, 3, 4, 5])
    np.testing.assert_array_equal(labels, expected)


def test_apply_completion_only_masking_list(simple_feature_generator):
    """Test applying completion-only masking to list inputs."""
    inputs = {
        "labels": [[1, 2, 100, 101, 3, 4, 5], [10, 11, 100, 101, 20, 30, 40]],
        "input_ids": [[1, 2, 100, 101, 3, 4, 5], [10, 11, 100, 101, 20, 30, 40]],
    }

    simple_feature_generator._apply_completion_only_masking(inputs)

    # Should mask everything including templates and keep only response content
    expected = [[-100, -100, -100, -100, 3, 4, 5], [-100, -100, -100, -100, 20, 30, 40]]
    assert inputs["labels"] == expected


def test_apply_completion_only_masking_numpy(simple_feature_generator):
    """Test applying completion-only masking to numpy inputs."""
    inputs = {
        "labels": np.array([[1, 2, 100, 101, 3, 4, 5]]),
        "input_ids": np.array([[1, 2, 100, 101, 3, 4, 5]]),
    }

    simple_feature_generator._apply_completion_only_masking(inputs)

    # Should mask everything including template and keep only response content
    expected = np.array([[-100, -100, -100, -100, 3, 4, 5]])
    np.testing.assert_array_equal(inputs["labels"], expected)


# Edge Case Tests
def test_empty_labels():
    """Test with empty labels array."""
    labels = np.array([])
    mask_labels_without_user_template(labels, [1, 2])
    assert len(labels) == 0


def test_response_template_longer_than_sequence():
    """Test when response template is longer than the entire sequence."""
    labels = np.array([1, 2])
    response_tokens = [1, 2, 3, 4, 5]

    mask_labels_without_user_template(labels, response_tokens)

    # Should mask everything since template not found
    expected = np.array([-100, -100])
    np.testing.assert_array_equal(labels, expected)


def test_no_assistant_response_with_user_template():
    """Test conversation with user template but no assistant response."""
    labels = np.array([200, 201, 10, 11, 12])  # Only user message
    response_tokens = [100, 101]  # Assistant template
    instruction_tokens = [200, 201]  # User template

    mask_labels_for_completions_only(labels, response_tokens, instruction_tokens)

    # Should mask everything since no assistant response found
    expected = np.array([-100, -100, -100, -100, -100])
    np.testing.assert_array_equal(labels, expected)


def test_assistant_response_at_end():
    """Test when assistant response is at the very end."""
    labels = np.array([200, 201, 10, 100, 101])
    response_tokens = [100, 101]
    instruction_tokens = [200, 201]

    mask_labels_for_completions_only(labels, response_tokens, instruction_tokens)

    # Should mask everything including the assistant template
    # (no content after template)
    expected = np.array([-100, -100, -100, -100, -100])
    np.testing.assert_array_equal(labels, expected)
