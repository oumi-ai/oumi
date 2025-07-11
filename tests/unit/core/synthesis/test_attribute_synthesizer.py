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

import pytest

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    PermutableAttribute,
    PermutableAttributeValue,
)
from oumi.core.synthesis.attribute_synthesizer import (
    AttributeSynthesizer,
    _AttributeInfo,
    _AttributeValueInfo,
)
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
                    sample_rate=0.6,
                ),
                PermutableAttributeValue(
                    id="casual",
                    value="Casual",
                    description="A casual writing style",
                    sample_rate=0.4,
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
                    sample_rate=0.5,
                ),
                PermutableAttributeValue(
                    id="science",
                    value="Science",
                    description="Science topics",
                    sample_rate=0.5,
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


def test_attribute_value_info_init():
    """Test initialization of _AttributeValueInfo."""
    info = _AttributeValueInfo("test_value", "test description")
    assert info._value_name == "test_value"
    assert info.description == "test description"


def test_attribute_value_info_str_representation():
    """Test string representation of _AttributeValueInfo."""
    info = _AttributeValueInfo("test_value", "test description")
    assert str(info) == "test_value"


def test_attribute_info_init():
    """Test initialization of _AttributeInfo."""
    info = _AttributeInfo(
        attribute_id="test_id",
        attribute_name="test_name",
        attribute_description="test attribute description",
        value_name="test_value",
        value_description="test value description",
    )
    assert info.attribute_id == "test_id"
    assert info._attribute_name == "test_name"
    assert info.description == "test attribute description"
    assert isinstance(info.value, _AttributeValueInfo)
    assert str(info.value) == "test_value"
    assert info.value.description == "test value description"


def test_attribute_info_str_representation():
    """Test string representation of _AttributeInfo."""
    info = _AttributeInfo(
        attribute_id="test_id",
        attribute_name="test_name",
        attribute_description="test attribute description",
        value_name="test_value",
        value_description="test value description",
    )
    assert str(info) == "test_name"


def test_init_with_permutable_attributes(mock_general_synthesis_params):
    """Test initialization with permutable attributes."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    assert synthesizer._params == mock_general_synthesis_params
    assert len(synthesizer._permutable_attribute_map) == 2
    assert "style" in synthesizer._permutable_attribute_map
    assert "topic" in synthesizer._permutable_attribute_map


def test_init_without_permutable_attributes():
    """Test initialization without permutable attributes."""
    params = GeneralSynthesisParams()
    synthesizer = AttributeSynthesizer(params)
    assert synthesizer._params == params
    assert synthesizer._permutable_attribute_map == {}


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


def test_get_permutable_attribute_value_info_invalid_value(
    mock_general_synthesis_params,
):
    """Test error when getting info for invalid attribute value."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    with pytest.raises(
        ValueError, match="Attribute value invalid_value not found for attribute style"
    ):
        synthesizer._get_permutable_attribute_value_info("style", "invalid_value")


def test_format_instructions_with_permutable_attributes(
    mock_general_synthesis_params, mock_generated_attribute
):
    """Test formatting instructions with permutable attributes."""
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
    # The content should have the attribute objects substituted
    assert "Write a Formal paragraph about Technology." in user_message.content


def test_format_instructions_with_non_permutable_attributes(
    mock_general_synthesis_params,
):
    """Test formatting instructions with non-permutable attributes."""
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


def test_format_instructions_with_mixed_attributes(mock_general_synthesis_params):
    """Test formatting instructions with permutable and non-permutable attributes."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {"style": "casual", "non_permutable": "mixed_value"}

    instruction_messages = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="{style} - ({style.description}) "
                "{style.value} - ({style.value.description}) "
                "style about {non_permutable}",
            ),
        ]
    )

    original_content = instruction_messages.messages[0].content

    result = synthesizer._format_instructions(sample, instruction_messages)

    # Original should be unchanged
    assert instruction_messages.messages[0].content == original_content

    assert isinstance(result, Conversation)
    assert len(result.messages) == 1

    user_message = result.messages[0]
    assert user_message.role == Role.USER
    assert (
        "Writing Style - (The style of writing to use) Casual - (A casual"
        " writing style) style about mixed_value" in user_message.content
    )


def test_format_instructions_with_unresolved_fields(mock_general_synthesis_params):
    """Test error when format string contains unresolved fields."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {"style": "formal"}

    instruction_messages = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="Write in {style} style about {missing_field}",
            ),
        ]
    )

    with pytest.raises(KeyError, match="missing_field"):
        synthesizer._format_instructions(sample, instruction_messages)


def test_format_instructions_with_non_string_content(mock_general_synthesis_params):
    """Test formatting instructions with non-string content (should be skipped)."""
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


def test_format_instructions_with_empty_sample(mock_general_synthesis_params):
    """Test formatting instructions with empty sample."""
    synthesizer = AttributeSynthesizer(mock_general_synthesis_params)
    sample = {}

    instruction_messages = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="No formatting needed",
            ),
        ]
    )

    result = synthesizer._format_instructions(sample, instruction_messages)

    assert isinstance(result, Conversation)
    assert len(result.messages) == 1
    assert result.messages[0].content == "No formatting needed"


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
