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

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    GeneratedAttributePostprocessingParams,
    PermutableAttribute,
    PermutableAttributeValue,
)
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.core.types.conversation import Conversation, Message, Role


@pytest.fixture
def mock_inference_config():
    """Create a mock inference config."""
    return Mock(spec=InferenceConfig)


@pytest.fixture
def mock_inference_engine():
    """Create a mock inference engine."""
    return Mock()


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
def mock_generated_attribute_with_postprocessing():
    """Create mock GeneratedAttribute with postprocessing for testing."""
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
        postprocessing_params=GeneratedAttributePostprocessingParams(
            id="processed_content",
            keep_original_text_attribute=True,
            cut_prefix="Response: ",
            cut_suffix=" [END]",
            strip_whitespace=True,
            added_prefix="New: ",
            added_suffix=" (done)",
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
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
    assert synthesizer._params == mock_general_synthesis_params
    assert synthesizer._formatter is not None


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_init_without_permutable_attributes(mock_get_engine, mock_inference_config):
    """Test initialization without permutable attributes."""
    mock_get_engine.return_value = Mock()

    params = GeneralSynthesisParams()
    synthesizer = AttributeSynthesizer(params, mock_inference_config)
    assert synthesizer._params == params
    assert synthesizer._formatter is not None


def test_synthesize_returns_conversations(
    mock_general_synthesis_params, mock_generated_attribute
):
    """Test that synthesize returns list of dictionaries."""
    mock_inference_engine = Mock()
    mock_get_engine.return_value = mock_inference_engine

    # Mock the inference engine's infer method to return conversations with responses
    mock_inference_engine.infer.return_value = [
        Conversation(
            messages=[
                Message(role=Role.USER, content="Test query"),
                Message(role=Role.ASSISTANT, content="Test response 1"),
            ]
        ),
        Conversation(
            messages=[
                Message(role=Role.USER, content="Test query"),
                Message(role=Role.ASSISTANT, content="Test response 2"),
            ]
        ),
    ]

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )
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
    mock_get_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test synthesize with multiple samples."""
    mock_inference_engine = Mock()
    mock_get_engine.return_value = mock_inference_engine

    # Mock the inference engine's infer method to return conversations with responses
    mock_inference_engine.infer.return_value = [
        Conversation(
            messages=[
                Message(role=Role.USER, content="Test query"),
                Message(role=Role.ASSISTANT, content="Test response 1"),
            ]
        ),
        Conversation(
            messages=[
                Message(role=Role.USER, content="Test query"),
                Message(role=Role.ASSISTANT, content="Test response 2"),
            ]
        ),
    ]

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
    samples = [
        {"style": "formal", "topic": "tech"},
        {"style": "casual", "topic": "science"},
    ]

    result = synthesizer.synthesize(samples, mock_generated_attribute)

    assert len(result) == 2
    for item in result:
        assert isinstance(item, dict)
        assert "generated_content" in item


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_synthesize_with_empty_samples(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test synthesize with empty samples list."""
    mock_inference_engine = Mock()
    mock_get_engine.return_value = mock_inference_engine

    # Mock the inference engine's infer method to return empty list
    mock_inference_engine.infer.return_value = []

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
    samples = []

    result = synthesizer.synthesize(samples, mock_generated_attribute)

    assert result == []


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_postprocess_sample(mock_get_engine):
    """Test postprocessing a sample."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(GeneralSynthesisParams(), Mock())

    response = "Response: Here is the formal text [END]"
    postprocessing_params = GeneratedAttributePostprocessingParams(
        id="processed_content",
        cut_prefix="Response: ",
        cut_suffix=" [END]",
        strip_whitespace=True,
        added_prefix="New: ",
        added_suffix=" (done)",
    )

    result = synthesizer._postprocess_sample(response, postprocessing_params)

    assert result == "New: Here is the formal text (done)"


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_postprocess_sample_with_regex(mock_get_engine):
    """Test postprocessing a sample with regex."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(GeneralSynthesisParams(), Mock())

    response = "The answer is 42 and that's final."
    postprocessing_params = GeneratedAttributePostprocessingParams(
        id="processed_content",
        regex=r"\d+",
        added_prefix="Number: ",
    )

    result = synthesizer._postprocess_sample(response, postprocessing_params)

    assert result == "Number: 42"


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_postprocess_sample_with_no_regex_match(mock_get_engine):
    """Test postprocessing a sample when regex doesn't match."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(GeneralSynthesisParams(), Mock())

    response = "No numbers here!"
    postprocessing_params = GeneratedAttributePostprocessingParams(
        id="processed_content",
        regex=r"\d+",
        added_prefix="Number: ",
    )

    result = synthesizer._postprocess_sample(response, postprocessing_params)

    assert result == "Number: No numbers here!"


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_postprocess_sample_minimal(mock_get_engine):
    """Test postprocessing a sample with minimal parameters."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(GeneralSynthesisParams(), Mock())

    response = "  Test response  "
    postprocessing_params = GeneratedAttributePostprocessingParams(
        id="processed_content",
        strip_whitespace=True,
    )

    result = synthesizer._postprocess_sample(response, postprocessing_params)

    assert result == "Test response"
