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

from unittest.mock import Mock, patch

import pytest

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    GeneratedAttributePostprocessingParams,
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


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_init_with_permutable_attributes(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_inference_config,
):
    """Test initialization with permutable attributes."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
    assert synthesizer._params == mock_general_synthesis_params
    assert len(synthesizer._permutable_attribute_map) == 2
    assert "style" in synthesizer._permutable_attribute_map
    assert "topic" in synthesizer._permutable_attribute_map
    mock_get_engine.assert_called_once_with(mock_inference_config)


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_init_without_permutable_attributes(mock_get_engine, mock_inference_config):
    """Test initialization without permutable attributes."""
    mock_get_engine.return_value = Mock()

    params = GeneralSynthesisParams()
    synthesizer = AttributeSynthesizer(params, mock_inference_config)
    assert synthesizer._params == params
    assert synthesizer._permutable_attribute_map == {}
    mock_get_engine.assert_called_once_with(mock_inference_config)


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_synthesize_returns_dict_list(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
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
    for item in result:
        assert isinstance(item, dict)
        assert "generated_content" in item
        assert isinstance(item["generated_content"], str)

    # Verify the inference engine was called
    mock_inference_engine.infer.assert_called_once()


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_synthesize_with_postprocessing(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_generated_attribute_with_postprocessing,
    mock_inference_config,
):
    """Test synthesize with postprocessing parameters."""
    mock_inference_engine = Mock()
    mock_get_engine.return_value = mock_inference_engine

    # Mock the inference engine's infer method to return conversations with responses
    mock_inference_engine.infer.return_value = [
        Conversation(
            messages=[
                Message(role=Role.USER, content="Test query"),
                Message(
                    role=Role.ASSISTANT,
                    content="Response: Here is the formal text [END]",
                ),
            ]
        ),
    ]

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )
    samples = [{"style": "formal", "topic": "tech"}]
    result = synthesizer.synthesize(
        samples, mock_generated_attribute_with_postprocessing
    )

    assert len(result) == 1
    assert "generated_content" in result[0]
    assert "processed_content" in result[0]

    # Check that postprocessing was applied
    processed_content = result[0]["processed_content"]
    assert processed_content == "New: Here is the formal text (done)"

    # Original content should also be preserved
    original_content = result[0]["generated_content"]
    assert original_content == "Response: Here is the formal text [END]"


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_get_permutable_attribute_value_info_invalid_value(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_inference_config,
):
    """Test error when getting info for invalid attribute value."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )
    with pytest.raises(
        ValueError,
        match="Attribute value invalid_value not found for attribute style",
    ):
        synthesizer._get_permutable_attribute_value_info("style", "invalid_value")


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_format_instructions_with_permutable_attributes(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test formatting instructions with permutable attributes."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )
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


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_format_instructions_with_non_permutable_attributes(
    mock_get_engine,
    mock_general_synthesis_params,
    mock_inference_config,
):
    """Test formatting instructions with non-permutable attributes."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
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


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_format_instructions_with_mixed_attributes(
    mock_get_engine, mock_general_synthesis_params, mock_inference_config
):
    """Test formatting instructions with permutable and non-permutable attributes."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
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


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_format_instructions_with_unresolved_fields(
    mock_get_engine, mock_general_synthesis_params, mock_inference_config
):
    """Test error when format string contains unresolved fields."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
    sample = {"style": "formal"}

    instruction_messages = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="Write in {style} style about {missing_field}",
            ),
        ]
    )

    with pytest.raises(
        ValueError,
        match="Missing value for placeholder: missing_field",
    ):
        synthesizer._format_instructions(sample, instruction_messages)


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_format_instructions_with_non_string_content(
    mock_get_engine, mock_general_synthesis_params, mock_inference_config
):
    """Test formatting instructions with non-string content (should be skipped)."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
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


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
def test_format_instructions_with_empty_sample(
    mock_get_engine, mock_general_synthesis_params, mock_inference_config
):
    """Test formatting instructions with empty sample."""
    mock_get_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
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


@patch("oumi.core.synthesis.attribute_synthesizer.get_engine")
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
