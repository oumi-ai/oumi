# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock, call, patch

import pytest

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    PermutableAttribute,
    PermutableAttributeValue,
)
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.core.types.conversation import Conversation, Message, Role


@pytest.fixture
def mock_permutable_attributes():
    """Create mock permutable attributes for testing."""
    return [
        PermutableAttribute(
            id="style",
            attribute="Writing Style",
            description="The style of writing to use",
            possible_values=[
                PermutableAttributeValue(
                    id="formal",
                    value="Formal",
                    description="A formal writing style",
                ),
                PermutableAttributeValue(
                    id="casual",
                    value="Casual",
                    description="A casual writing style",
                ),
            ],
        ),
        PermutableAttribute(
            id="topic",
            attribute="Topic",
            description="The topic to write about",
            possible_values=[
                PermutableAttributeValue(
                    id="tech",
                    value="Technology",
                    description="Technology topics",
                ),
                PermutableAttributeValue(
                    id="science",
                    value="Science",
                    description="Science topics",
                ),
            ],
        ),
    ]


@pytest.fixture
def mock_general_synthesis_params(mock_permutable_attributes):
    """Create mock GeneralSynthesisParams for testing."""
    return GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
    )


@pytest.fixture
def mock_generated_attribute():
    """Create mock GeneratedAttribute for testing."""
    return GeneratedAttribute(
        id="generated_content",
        instruction_messages=Conversation(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content="You are a helpful assistant.",
                ),
                Message(
                    role=Role.USER,
                    content="Write a {style.value} paragraph about {topic.value}.",
                ),
            ]
        ),
    )


@pytest.fixture
def mock_samples():
    """Create mock samples for testing."""
    return [
        {"style": "formal", "topic": "tech"},
        {"style": "casual", "topic": "science"},
        {"non_permutable": "some_value"},
    ]


def test_init_with_permutable_attributes(mock_general_synthesis_params):
    """Test initialization with permutable attributes."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    assert synthesizer._params == mock_general_synthesis_params
    assert synthesizer._formatter is not None


def test_init_without_permutable_attributes():
    """Test initialization without permutable attributes."""
    params = GeneralSynthesisParams()
    synthesizer = AttributeSynthesizer(params)
    assert synthesizer._params == params
    assert synthesizer._formatter is not None


def test_synthesize_returns_conversations(
    mock_general_synthesis_params, mock_generated_attribute
):
    """Test that synthesize returns list of Conversation objects."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    # Use samples that have all required fields
    samples = [
        {"style": "formal", "topic": "tech"},
        {"style": "casual", "topic": "science"},
    ]
    result = synthesizer.synthesize(samples, mock_generated_attribute)

    assert isinstance(result, list)
    assert len(result) == len(samples)
    for conversation in result:
        assert isinstance(conversation, Conversation)


@patch("oumi.core.synthesis.attribute_synthesizer.AttributeFormatter")
def test_format_instructions_with_permutable_attributes(
    mock_formatter_class, mock_general_synthesis_params, mock_generated_attribute
):
    """Test formatting instructions with permutable attributes."""
    # Mock the formatter instance
    mock_formatter = Mock()
    mock_formatter.format.side_effect = [
        "You are a helpful assistant.",
        "Write a Formal paragraph about Technology.",
    ]
    mock_formatter_class.return_value = mock_formatter

    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {"style": "formal", "topic": "tech"}

    result = synthesizer._format_instructions(
        sample,
        mock_generated_attribute.instruction_messages,
    )

    assert isinstance(result, Conversation)
    assert len(result.messages) == 2

    # Check that the formatting worked correctly
    user_message = result.messages[1]
    assert user_message.role == Role.USER
    assert user_message.content == "Write a Formal paragraph about Technology."

    # Verify formatter was called correctly for both messages
    expected_calls = [
        call(sample, "You are a helpful assistant.", missing_values_allowed=False),
        call(
            sample,
            "Write a {style.value} paragraph about {topic.value}.",
            missing_values_allowed=False,
        ),
    ]
    mock_formatter.format.assert_has_calls(expected_calls)


@patch("oumi.core.synthesis.attribute_synthesizer.AttributeFormatter")
def test_format_instructions_with_non_permutable_attributes(
    mock_formatter_class, mock_general_synthesis_params
):
    """Test formatting instructions with non-permutable attributes."""
    # Mock the formatter instance
    mock_formatter = Mock()
    mock_formatter.format.return_value = "Use this value: some_value"
    mock_formatter_class.return_value = mock_formatter

    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {"non_permutable": "some_value"}

    instruction_messages = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="Use this value: {non_permutable}",
            ),
        ]
    )

    result = synthesizer._format_instructions(sample, instruction_messages)

    assert isinstance(result, Conversation)
    assert len(result.messages) == 1

    user_message = result.messages[0]
    assert user_message.role == Role.USER
    assert user_message.content == "Use this value: some_value"


@patch("oumi.core.synthesis.attribute_synthesizer.AttributeFormatter")
def test_format_instructions_with_non_string_content(
    mock_formatter_class, mock_general_synthesis_params
):
    """Test formatting instructions with non-string content (should be skipped)."""
    # Mock the formatter instance
    mock_formatter = Mock()
    mock_formatter_class.return_value = mock_formatter

    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {"style": "formal"}

    # Create a message with non-string content (list of ContentItem)
    from oumi.core.types.conversation import ContentItem, Type

    content_items = [
        ContentItem(type=Type.TEXT, content="Write in {style} style"),
    ]

    instruction_messages = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=content_items,
            ),
        ]
    )

    result = synthesizer._format_instructions(sample, instruction_messages)

    assert isinstance(result, Conversation)
    assert len(result.messages) == 1

    # Content should remain unchanged since it's not a string
    assert result.messages[0].content == content_items

    # Formatter should not have been called
    mock_formatter.format.assert_not_called()


@patch("oumi.core.synthesis.attribute_synthesizer.AttributeFormatter")
def test_format_instructions_preserves_original_message(
    mock_formatter_class, mock_general_synthesis_params
):
    """Test that formatting preserves the original message structure."""
    # Mock the formatter instance
    mock_formatter = Mock()
    mock_formatter.format.return_value = "Formatted content"
    mock_formatter_class.return_value = mock_formatter

    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {"style": "formal"}

    original_conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="Original {style} content",
            ),
        ],
        conversation_id="test_id",
        metadata={"test": "metadata"},
    )

    result = synthesizer._format_instructions(sample, original_conversation)

    # Original should be unchanged
    assert original_conversation.messages[0].content == "Original {style} content"

    # New conversation should have formatted content
    assert result.messages[0].content == "Formatted content"
    assert result.conversation_id == "test_id"
    assert result.metadata == {"test": "metadata"}


def test_synthesize_with_multiple_samples(
    mock_general_synthesis_params, mock_generated_attribute
):
    """Test synthesize with multiple samples."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    samples = [
        {"style": "formal", "topic": "tech"},
        {"style": "casual", "topic": "science"},
    ]

    result = synthesizer.synthesize(samples, mock_generated_attribute)

    assert len(result) == 2
    for conversation in result:
        assert isinstance(conversation, Conversation)
        assert len(conversation.messages) == 2


def test_synthesize_with_empty_samples(
    mock_general_synthesis_params, mock_generated_attribute
):
    """Test synthesize with empty samples list."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    samples = []

    result = synthesizer.synthesize(samples, mock_generated_attribute)

    assert result == []
