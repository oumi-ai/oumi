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

from unittest.mock import Mock

import pytest

from oumi.core.configs.params.synthesis_params import (
    GeneratedAttribute,
    PermutableAttribute,
    PermutableAttributeValue,
    TextMessage,
)
from oumi.core.synthesis.attribute_synthesizer import AttributeSynthesizer
from oumi.core.synthesis.data_synthesizer import DataSynthesizer
from oumi.core.types.conversation import Role


@pytest.fixture
def mock_attribute_synthesizer():
    """Create a mock attribute synthesizer."""
    return Mock(spec=AttributeSynthesizer)


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
    ]


@pytest.fixture
def mock_generated_attributes():
    """Create mock generated attributes for testing."""
    return [
        GeneratedAttribute(
            id="content",
            instruction_messages=[
                TextMessage(
                    role=Role.SYSTEM,
                    content="You are a helpful assistant.",
                ),
                TextMessage(
                    role=Role.USER,
                    content="Write a {style.value} paragraph.",
                ),
            ],
        ),
        GeneratedAttribute(
            id="summary",
            instruction_messages=[
                TextMessage(
                    role=Role.SYSTEM,
                    content="You are a helpful assistant.",
                ),
                TextMessage(
                    role=Role.USER,
                    content="Summarize the content: {content}",
                ),
            ],
        ),
    ]


@pytest.fixture
def mock_dataset_plan_samples():
    """Create mock dataset plan samples for testing."""
    return [
        {"style": "formal", "topic": "technology"},
        {"style": "casual", "topic": "science"},
        {"style": "formal", "topic": "history"},
    ]


def test_init(
    mock_generated_attributes,
    mock_attribute_synthesizer,
):
    """Test initialization of DataSynthesizer."""
    synthesizer = DataSynthesizer(mock_generated_attributes, mock_attribute_synthesizer)

    assert synthesizer._generated_attributes == mock_generated_attributes
    assert synthesizer._attribute_synthesizer == mock_attribute_synthesizer


def test_synthesize_with_no_generated_attributes(
    mock_attribute_synthesizer,
    mock_dataset_plan_samples,
):
    """Test synthesize method with no generated attributes."""
    synthesizer = DataSynthesizer([], mock_attribute_synthesizer)

    result = synthesizer.synthesize(mock_dataset_plan_samples)

    assert result == mock_dataset_plan_samples
    mock_attribute_synthesizer.synthesize.assert_not_called()


def test_synthesize_with_single_generated_attribute(
    mock_generated_attributes,
    mock_attribute_synthesizer,
    mock_dataset_plan_samples,
):
    """Test synthesize method with a single generated attribute."""
    # Mock the attribute synthesizer to return synthesized content
    mock_attribute_synthesizer.synthesize.return_value = [
        {"content": "This is formal tech content."},
        {"content": "This is casual science content."},
        {"content": "This is formal history content."},
    ]

    # Use only one generated attribute
    single_generated_attribute = [mock_generated_attributes[0]]

    synthesizer = DataSynthesizer(
        single_generated_attribute, mock_attribute_synthesizer
    )

    result = synthesizer.synthesize(mock_dataset_plan_samples)

    expected_result = [
        {
            "style": "formal",
            "topic": "technology",
            "content": "This is formal tech content.",
        },
        {
            "style": "casual",
            "topic": "science",
            "content": "This is casual science content.",
        },
        {
            "style": "formal",
            "topic": "history",
            "content": "This is formal history content.",
        },
    ]

    assert result == expected_result
    mock_attribute_synthesizer.synthesize.assert_called_once_with(
        mock_dataset_plan_samples, single_generated_attribute[0]
    )


def test_synthesize_with_multiple_generated_attributes(
    mock_generated_attributes,
    mock_attribute_synthesizer,
    mock_dataset_plan_samples,
):
    """Test synthesize method with multiple generated attributes."""
    # Mock the attribute synthesizer to return different content for each attribute
    mock_attribute_synthesizer.synthesize.side_effect = [
        # First call for "content" attribute
        [
            {"content": "This is formal tech content."},
            {"content": "This is casual science content."},
            {"content": "This is formal history content."},
        ],
        # Second call for "summary" attribute
        [
            {"summary": "Summary of tech content."},
            {"summary": "Summary of science content."},
            {"summary": "Summary of history content."},
        ],
    ]

    synthesizer = DataSynthesizer(mock_generated_attributes, mock_attribute_synthesizer)

    result = synthesizer.synthesize(mock_dataset_plan_samples)

    expected_result = [
        {
            "style": "formal",
            "topic": "technology",
            "content": "This is formal tech content.",
            "summary": "Summary of tech content.",
        },
        {
            "style": "casual",
            "topic": "science",
            "content": "This is casual science content.",
            "summary": "Summary of science content.",
        },
        {
            "style": "formal",
            "topic": "history",
            "content": "This is formal history content.",
            "summary": "Summary of history content.",
        },
    ]

    assert result == expected_result
    assert mock_attribute_synthesizer.synthesize.call_count == 2

    # Verify the calls were made with the correct arguments
    calls = mock_attribute_synthesizer.synthesize.call_args_list
    assert calls[0][0][1] == mock_generated_attributes[0]
    assert calls[1][0][1] == mock_generated_attributes[1]


def test_synthesize_with_empty_dataset_plan_samples(
    mock_generated_attributes,
    mock_attribute_synthesizer,
):
    """Test synthesize method with empty dataset plan samples."""
    # Mock the attribute synthesizer to return empty list
    mock_attribute_synthesizer.synthesize.return_value = []

    synthesizer = DataSynthesizer(mock_generated_attributes, mock_attribute_synthesizer)

    result = synthesizer.synthesize([])

    assert result == []
    assert mock_attribute_synthesizer.synthesize.call_count == 0


def test_synthesize_sequential_processing(
    mock_generated_attributes,
    mock_attribute_synthesizer,
    mock_dataset_plan_samples,
):
    """Test that attributes are processed sequentially and build on each other."""

    # Mock the attribute synthesizer to return different content for each attribute
    def mock_synthesize(samples, generated_attribute):
        if generated_attribute.id == "content":
            return [{"content": f"Content for {sample['style']}"} for sample in samples]
        elif generated_attribute.id == "summary":
            return [
                {"summary": f"Summary of {sample['content']}"} for sample in samples
            ]
        return []

    mock_attribute_synthesizer.synthesize.side_effect = mock_synthesize

    synthesizer = DataSynthesizer(mock_generated_attributes, mock_attribute_synthesizer)

    result = synthesizer.synthesize(mock_dataset_plan_samples)

    # Verify that the second attribute synthesis received the updated samples
    # (including the content from the first attribute)
    calls = mock_attribute_synthesizer.synthesize.call_args_list
    second_call_samples = calls[1][0][0]

    # The second call should have received samples that include the content
    # from the first attribute synthesis
    assert "content" in second_call_samples[0]
    assert second_call_samples[0]["content"] == "Content for formal"

    # Verify final result includes both attributes
    assert result[0]["content"] == "Content for formal"
    assert result[0]["summary"] == "Summary of Content for formal"


def test_synthesize_updates_original_samples(
    mock_generated_attributes,
    mock_attribute_synthesizer,
):
    """Test that synthesize updates the original samples in place."""
    # Mock the attribute synthesizer to return content
    mock_attribute_synthesizer.synthesize.return_value = [
        {"content": "Generated content 1"},
        {"content": "Generated content 2"},
        {"content": "Generated content 3"},
    ]

    # Use only one generated attribute for simplicity
    single_generated_attribute = [mock_generated_attributes[0]]

    synthesizer = DataSynthesizer(
        single_generated_attribute, mock_attribute_synthesizer
    )

    # Create samples for this test specifically
    test_samples = [
        {"style": "formal", "topic": "technology"},
        {"style": "casual", "topic": "science"},
        {"style": "formal", "topic": "history"},
    ]

    original_samples = [sample.copy() for sample in test_samples]
    result = synthesizer.synthesize(test_samples)

    # Verify that original samples were updated
    assert test_samples == result
    assert test_samples != original_samples
    assert "content" in test_samples[0]
    assert "content" not in original_samples[0]
