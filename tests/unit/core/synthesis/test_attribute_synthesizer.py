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
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
    GeneratedAttributePostprocessingParams,
    PermutableAttribute,
    PermutableAttributeValue,
    TextMessage,
)
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.core.types.conversation import Conversation, Message, Role


@pytest.fixture
def mock_inference_config():
    """Create a mock inference config."""
    mock = Mock(spec=InferenceConfig)
    mock.engine = InferenceEngineType.NATIVE
    mock.model = Mock(spec=ModelParams)
    mock.remote_params = Mock(spec=RemoteParams)
    return mock


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
        instruction_messages=[
            TextMessage(
                role=Role.SYSTEM,
                content="You are a helpful assistant.",
            ),
            TextMessage(
                role=Role.USER,
                content="Write a {style.value} paragraph about {topic.value}.",
            ),
        ],
    )


@pytest.fixture
def mock_samples():
    """Create mock samples for testing."""
    return [
        {"style": "formal", "topic": "tech"},
        {"style": "casual", "topic": "science"},
        {"non_permutable": "some_value"},
    ]


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_synthesize_returns_dict_list(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test that synthesize returns list of dictionaries."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine

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


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_format_instructions_with_permutable_attributes(
    mock_formatter_class,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test formatting instructions with permutable attributes."""
    # Mock the formatter instance
    mock_formatter = Mock()
    mock_formatter.format.side_effect = [
        "You are a helpful assistant.",
        "Write a Formal paragraph about Technology.",
    ]
    mock_formatter_class.return_value = mock_formatter

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
    assert result.messages[0].role == Role.SYSTEM
    assert result.messages[0].content == "You are a helpful assistant."
    assert result.messages[1].role == Role.USER
    assert result.messages[1].content == "Write a Formal paragraph about Technology."


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_synthesize_with_multiple_samples(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test synthesize with multiple samples."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine

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


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_synthesize_with_empty_samples(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_generated_attribute,
    mock_inference_config,
):
    """Test synthesize with empty samples list."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine

    # Mock the inference engine's infer method to return empty list
    mock_inference_engine.infer.return_value = []

    synthesizer = AttributeSynthesizer(
        mock_general_synthesis_params, mock_inference_config
    )
    samples = []

    result = synthesizer.synthesize(samples, mock_generated_attribute)

    assert result == []


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_postprocess_sample(mock_build_inference_engine):
    """Test postprocessing a sample."""
    mock_build_inference_engine.return_value = Mock()

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


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_postprocess_sample_with_regex(mock_build_inference_engine):
    """Test postprocessing a sample with regex."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = AttributeSynthesizer(GeneralSynthesisParams(), Mock())

    response = "The answer is 42 and that's final."
    postprocessing_params = GeneratedAttributePostprocessingParams(
        id="processed_content",
        regex=r"\d+",
        added_prefix="Number: ",
    )

    result = synthesizer._postprocess_sample(response, postprocessing_params)

    assert result == "Number: 42"


@patch("oumi.core.synthesis.attribute_synthesizer.build_inference_engine")
def test_postprocess_sample_with_no_regex_match(mock_build_inference_engine):
    """Test postprocessing a sample when regex doesn't match."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = AttributeSynthesizer(GeneralSynthesisParams(), Mock())

    response = "No numbers here!"
    postprocessing_params = GeneratedAttributePostprocessingParams(
        id="processed_content",
        regex=r"\d+",
        added_prefix="Number: ",
    )

    result = synthesizer._postprocess_sample(response, postprocessing_params)

    assert result == "Number: No numbers here!"
