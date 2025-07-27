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

import random
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs.params.synthesis_params import (
    AttributeCombination,
    DatasetSource,
    DocumentSegmentationParams,
    DocumentSource,
    ExampleSource,
    GeneralSynthesisParams,
    PermutableAttribute,
    PermutableAttributeValue,
    SegmentationStrategy,
)
from oumi.core.synthesis.dataset_planner import DatasetPlanner


@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set up a fixed random seed for deterministic tests."""
    random.seed(42)
    yield
    random.seed()  # Reset the seed after the test


@pytest.fixture
def planner():
    return DatasetPlanner()


@pytest.fixture
def mock_permutable_attributes():
    class MockPermutableAttribute1(PermutableAttribute):
        def __init__(self):
            super().__init__(
                id="attr1",
                attribute="test1",
                description="First test attribute",
                possible_values=[
                    PermutableAttributeValue(
                        id="value1",
                        value="value1",
                        description="First value",
                        sample_rate=0.5,
                    ),
                    PermutableAttributeValue(
                        id="value2",
                        value="value2",
                        description="Second value",
                        sample_rate=0.5,
                    ),
                ],
            )

        def get_value_distribution(self):
            return {"value1": 0.5, "value2": 0.5}

    class MockPermutableAttribute2(PermutableAttribute):
        def __init__(self):
            super().__init__(
                id="attr2",
                attribute="test2",
                description="Second test attribute",
                possible_values=[
                    PermutableAttributeValue(
                        id="valueA",
                        value="valueA",
                        description="Value A",
                        sample_rate=0.5,
                    ),
                    PermutableAttributeValue(
                        id="valueB",
                        value="valueB",
                        description="Value B",
                        sample_rate=0.5,
                    ),
                ],
            )

        def get_value_distribution(self):
            return {"valueA": 0.5, "valueB": 0.5}

    return [MockPermutableAttribute1(), MockPermutableAttribute2()]


@pytest.fixture
def mock_dataset_sources():
    """Mock dataset sources for testing."""
    return [
        DatasetSource(path="test_dataset1.jsonl"),
        DatasetSource(path="test_dataset2.jsonl"),
    ]


@pytest.fixture
def mock_dataset_data():
    """Mock data that would be returned by DatasetReader."""
    return [
        [
            {"col1": "value1", "col2": "valueA"},
            {"col1": "value2", "col2": "valueB"},
            {"col1": "value3", "col2": "valueC"},
        ],
        [
            {"col3": "data1", "col4": "dataA"},
            {"col3": "data2", "col4": "dataB"},
        ],
    ]


@pytest.fixture
def mock_example_sources():
    """Mock example sources for testing."""
    return [
        ExampleSource(
            examples=[
                {"example_attr1": "example_value1", "example_attr2": "example_valueA"},
                {"example_attr1": "example_value2", "example_attr2": "example_valueB"},
            ]
        )
    ]


@pytest.fixture
def mock_document_sources():
    """Mock document sources for testing."""
    return [
        DocumentSource(path="test_document1.pdf", id="doc1"),
        DocumentSource(path="test_document2.txt", id="doc2"),
    ]


@pytest.fixture
def mock_document_sources_with_segmentation():
    """Mock document sources with segmentation for testing."""
    return [
        DocumentSource(
            path="test_document1.pdf",
            id="doc1",
            segmentation_params=DocumentSegmentationParams(
                id="doc1_segment",
                segmentation_strategy=SegmentationStrategy.TOKENS,
                segment_length=512,
                segment_overlap=50,
                keep_original_text=False,
            ),
        ),
        DocumentSource(
            path="test_document2.txt",
            id="doc2",
            segmentation_params=DocumentSegmentationParams(
                id="doc2_segment",
                segmentation_strategy=SegmentationStrategy.TOKENS,
                segment_length=1024,
                segment_overlap=100,
                keep_original_text=True,
            ),
        ),
    ]


@pytest.fixture
def mock_document_data():
    """Mock document data that would be returned by DocumentReader."""
    return [
        "This is the content of document 1. It contains multiple sentences and "
        "paragraphs.",
        "This is the content of document 2. It has different text content from "
        "document 1.",
        "This is the content of document 3. It provides additional context for "
        "testing.",
    ]


@pytest.fixture
def mock_document_segments():
    """Mock document segments that would be returned by DocumentSegmenter."""
    return [
        ["This is segment 1 of document 1.", "This is segment 2 of document 1."],
        [
            "This is segment 1 of document 2.",
            "This is segment 2 of document 2.",
            "This is segment 3 of document 2.",
        ],
    ]


def test_plan_with_no_permutable_attributes(planner):
    params = GeneralSynthesisParams(permutable_attributes=None)
    with pytest.raises(ValueError, match="Empty sample created after planning"):
        _ = planner.plan(params, sample_count=5)


def test_plan_with_empty_permutable_attributes(planner, mock_permutable_attributes):
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    params.permutable_attributes = []
    with pytest.raises(ValueError, match="Empty sample created after planning"):
        _ = planner.plan(params, sample_count=5)


def test_plan_with_zero_samples(planner, mock_permutable_attributes):
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    with pytest.raises(ValueError, match="sample_count must be positive"):
        _ = planner.plan(params, sample_count=0)


def test_plan_with_negative_samples(planner, mock_permutable_attributes):
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    with pytest.raises(ValueError, match="sample_count must be positive"):
        planner.plan(params, sample_count=-1)


def test_plan_with_valid_permutable_attributes(planner, mock_permutable_attributes):
    """Test that the distribution matches expected values with a fixed seed."""
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    sample_count = 10
    result = planner.plan(params, sample_count=sample_count)

    # With seed 42, we know exactly what values we should get
    expected_values = [
        "value1",
        "value2",
        "value1",
        "value1",
        "value1",
        "value1",
        "value1",
        "value1",
        "value1",
        "value2",
    ]

    for i, sample in enumerate(result):
        value = list(sample.values())[0]
        assert value == expected_values[i], (
            f"Value at index {i} does not match expected value"
        )


def test_plan_with_valid_combination_sampling(planner, mock_permutable_attributes):
    """Test that combination sampling works with valid probabilities."""
    combination = {"attr1": "value1", "attr2": "valueA"}
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        combination_sampling=[
            AttributeCombination(combination=combination, sample_rate=0.8)
        ],
    )
    sample_count = 100
    result = planner.plan(params, sample_count=sample_count)

    # Count how many samples match our forced combination
    matching_samples = sum(
        1
        for sample in result
        if all(sample.get(attr) == val for attr, val in combination.items())
    )

    # With seed 42 and 0.8 probability, we expect 8 matches
    assert matching_samples == 78, (
        "Expected 78 samples to match the combination based on seed 42"
    )

    # Check that non-matching samples have the correct distribution
    non_matching_samples = [
        sample
        for sample in result
        if not all(sample.get(attr) == val for attr, val in combination.items())
    ]
    non_matching_attr_1_values = [
        sample.get("attr1") for sample in non_matching_samples
    ]
    non_matching_attr_2_values = [
        sample.get("attr2") for sample in non_matching_samples
    ]
    assert non_matching_attr_1_values.count("value1") == 8, (
        "Expected 8 non-matching samples with value1"
    )
    assert non_matching_attr_1_values.count("value2") == 14, (
        "Expected 14 non-matching samples with value2"
    )
    assert non_matching_attr_2_values.count("valueA") == 7, (
        "Expected 10 non-matching samples with valueA"
    )
    assert non_matching_attr_2_values.count("valueB") == 15, (
        "Expected 10 non-matching samples with valueB"
    )


def test_plan_with_multiple_combinations(planner, mock_permutable_attributes):
    """Test that multiple combinations are sampled correctly."""
    combinations = [
        AttributeCombination(
            combination={"attr1": "value1", "attr2": "valueA"}, sample_rate=0.3
        ),
        AttributeCombination(
            combination={"attr1": "value2", "attr2": "valueB"}, sample_rate=0.3
        ),
    ]
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        combination_sampling=combinations,
    )
    sample_count = 100
    result = planner.plan(params, sample_count=sample_count)

    # Count matches for each combination
    matches = [
        sum(
            1
            for sample in result
            if all(sample.get(attr) == val for attr, val in comb.combination.items())
        )
        for comb in combinations
    ]

    # With seed 42 and 0.3 probability each, we expect 26 and 29 matches
    expected_matches = [26, 29]
    for i, m in enumerate(matches):
        assert m == expected_matches[i], (
            f"Expected {expected_matches[i]} matches for combination {i}"
        )


def test_plan_resamples_on_combination_match(planner, mock_permutable_attributes):
    """Test that samples are redrawn if they accidentally match a combination."""
    # Set up a combination that would be very likely to occur randomly
    forbidden_combination = {"attr1": "value1", "attr2": "valueA"}
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        combination_sampling=[
            AttributeCombination(
                combination=forbidden_combination,
                sample_rate=0.0,  # Explicitly never sample this combination
            )
        ],
    )
    sample_count = 20
    result = planner.plan(params, sample_count=sample_count)

    # Check that none of the random samples match our forbidden combination
    matching_samples = sum(
        1
        for sample in result
        if all(sample.get(attr) == val for attr, val in forbidden_combination.items())
    )
    assert matching_samples == 0, (
        "Expected no samples to match the forbidden combination"
    )


def test_plan_with_no_dataset_sources(planner, mock_permutable_attributes):
    """Dataset plan should still be created if no dataset sources are provided."""
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_data=None,
    )
    result = planner.plan(params, sample_count=5)

    assert len(result) == 5
    # Should only contain permutable attributes
    for sample in result:
        assert "attr1" in sample
        assert "attr2" in sample
        assert len(sample) == 2


def test_plan_with_empty_dataset_sources(planner, mock_permutable_attributes):
    """Dataset plan should still be created if dataset sources are empty."""
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_data=None,  # Use None instead of [] to avoid validation error
    )
    result = planner.plan(params, sample_count=5)

    assert len(result) == 5
    # Should only contain permutable attributes
    for sample in result:
        assert "attr1" in sample
        assert "attr2" in sample
        assert len(sample) == 2


def test_plan_with_example_sources(
    planner, mock_permutable_attributes, mock_example_sources
):
    """Test that example sources are correctly included in the dataset plan."""
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_examples=mock_example_sources,
    )
    result = planner.plan(params, sample_count=5)

    assert len(result) == 5
    # Each sample should have both example attributes and permutable attributes
    for i, sample in enumerate(result):
        # Check example attributes (round-robin)
        example_index = i % 2
        expected_example_attr1 = ["example_value1", "example_value2"][example_index]
        expected_example_attr2 = ["example_valueA", "example_valueB"][example_index]
        assert sample["example_attr1"] == expected_example_attr1
        assert sample["example_attr2"] == expected_example_attr2

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have all 4 attributes (2 from examples + 2 permutable)
        assert len(sample) == 4


@patch("oumi.core.synthesis.dataset_planner.DatasetReader")
def test_plan_with_single_dataset_source(
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_dataset_sources,
):
    """Dataset plan should be created with a single dataset source."""
    mock_reader = Mock()
    mock_reader.read.return_value = [
        {"col1": "value1", "col2": "valueA"},
        {"col1": "value2", "col2": "valueB"},
        {"col1": "value3", "col2": "valueC"},
    ]
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_data=[mock_dataset_sources[0]],
    )
    result = planner.plan(params, sample_count=5)

    # Verify DatasetReader was called correctly
    mock_reader_class.assert_called_once()
    mock_reader.read.assert_called_once_with(mock_dataset_sources[0])

    assert len(result) == 5
    # Each sample should have both dataset attributes and permutable attributes
    for i, sample in enumerate(result):
        # Check dataset attributes (round-robin)
        dataset_index = i % 3
        expected_col1 = ["value1", "value2", "value3"][dataset_index]
        expected_col2 = ["valueA", "valueB", "valueC"][dataset_index]
        assert sample["col1"] == expected_col1
        assert sample["col2"] == expected_col2

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have all 4 attributes
        assert len(sample) == 4


@patch("oumi.core.synthesis.dataset_planner.DatasetReader")
def test_plan_with_multiple_dataset_sources(
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_dataset_sources,
    mock_dataset_data,
):
    """Dataset plan should be created with multiple dataset sources."""
    mock_reader = Mock()
    mock_reader.read.side_effect = mock_dataset_data
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_data=mock_dataset_sources,
    )
    result = planner.plan(params, sample_count=6)

    # Verify DatasetReader was called correctly
    mock_reader_class.assert_called_once()
    assert mock_reader.read.call_count == 2

    assert len(result) == 6
    # Each sample should have attributes from both datasets and permutable attributes
    for i, sample in enumerate(result):
        # Check first dataset attributes (round-robin)
        dataset1_index = i % 3
        expected_col1 = ["value1", "value2", "value3"][dataset1_index]
        expected_col2 = ["valueA", "valueB", "valueC"][dataset1_index]
        assert sample["col1"] == expected_col1
        assert sample["col2"] == expected_col2

        # Check second dataset attributes (round-robin)
        dataset2_index = i % 2
        expected_col3 = ["data1", "data2"][dataset2_index]
        expected_col4 = ["dataA", "dataB"][dataset2_index]
        assert sample["col3"] == expected_col3
        assert sample["col4"] == expected_col4

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have all 6 attributes (4 from datasets + 2 permutable)
        assert len(sample) == 6


@patch("oumi.core.synthesis.dataset_planner.DatasetReader")
def test_plan_with_dataset_sources_only(
    mock_reader_class, planner, mock_dataset_sources
):
    """Dataset plan should be created with no permutable attributes."""
    mock_reader = Mock()
    mock_reader.read.return_value = [
        {"col1": "value1", "col2": "valueA"},
        {"col1": "value2", "col2": "valueB"},
    ]
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=None,
        input_data=[mock_dataset_sources[0]],
    )
    result = planner.plan(params, sample_count=4)

    # Verify DatasetReader was called correctly
    mock_reader_class.assert_called_once()
    mock_reader.read.assert_called_once_with(mock_dataset_sources[0])

    assert len(result) == 4
    # Each sample should only have dataset attributes (round-robin)
    for i, sample in enumerate(result):
        dataset_index = i % 2
        expected_col1 = ["value1", "value2"][dataset_index]
        expected_col2 = ["valueA", "valueB"][dataset_index]
        assert sample["col1"] == expected_col1
        assert sample["col2"] == expected_col2

        # Should have only dataset attributes
        assert len(sample) == 2


# Document handling tests
def test_plan_with_no_document_sources(planner, mock_permutable_attributes):
    """Dataset plan should still be created if no document sources are provided."""
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=None,
    )
    result = planner.plan(params, sample_count=5)

    assert len(result) == 5
    # Should only contain permutable attributes
    for sample in result:
        assert "attr1" in sample
        assert "attr2" in sample
        assert len(sample) == 2


def test_plan_with_empty_document_sources(planner, mock_permutable_attributes):
    """Dataset plan should still be created if document sources are empty."""
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=None,  # Use None instead of [] to avoid validation error
    )
    result = planner.plan(params, sample_count=5)

    assert len(result) == 5
    # Should only contain permutable attributes
    for sample in result:
        assert "attr1" in sample
        assert "attr2" in sample
        assert len(sample) == 2


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
def test_plan_with_single_document_source_no_segmentation(
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_document_sources,
    mock_document_data,
):
    """Test plan with single document source without segmentation."""
    mock_reader = Mock()
    mock_reader.read.return_value = mock_document_data[:1]  # Only first document
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=[mock_document_sources[0]],
    )
    result = planner.plan(params, sample_count=3)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    mock_reader.read.assert_called_once_with(mock_document_sources[0].path)

    assert len(result) == 3
    # Each sample should have both document attributes and permutable attributes
    for i, sample in enumerate(result):
        # Check document attributes (round-robin) - only one document, so always index 0
        expected_doc_content = mock_document_data[0]
        assert sample["doc1"] == expected_doc_content

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have 3 attributes (1 from document + 2 permutable)
        assert len(sample) == 3


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
def test_plan_with_multiple_document_sources_no_segmentation(
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_document_sources,
    mock_document_data,
):
    """Test plan with multiple document sources without segmentation."""
    mock_reader = Mock()
    mock_reader.read.side_effect = [
        [mock_document_data[0]],  # Document 1
        [mock_document_data[1]],  # Document 2
    ]
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=mock_document_sources,
    )
    result = planner.plan(params, sample_count=4)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    assert mock_reader.read.call_count == 2

    assert len(result) == 4
    # Each sample should have attributes from both documents and permutable attributes
    for i, sample in enumerate(result):
        # Check first document attributes (round-robin) - only one document per source
        expected_doc1_content = mock_document_data[0]
        assert sample["doc1"] == expected_doc1_content

        # Check second document attributes (round-robin) - only one document per source
        expected_doc2_content = mock_document_data[1]
        assert sample["doc2"] == expected_doc2_content

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have 4 attributes (2 from documents + 2 permutable)
        assert len(sample) == 4


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
@patch("oumi.core.synthesis.dataset_planner.DocumentSegmenter")
def test_plan_with_document_source_with_segmentation(
    mock_segmenter_class,
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_document_sources_with_segmentation,
    mock_document_data,
    mock_document_segments,
):
    """Test plan with document source that has segmentation."""
    mock_reader = Mock()
    mock_reader.read.return_value = [mock_document_data[0]]
    mock_reader_class.return_value = mock_reader

    mock_segmenter = Mock()
    mock_segmenter.segment.return_value = mock_document_segments[0]
    mock_segmenter_class.return_value = mock_segmenter

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=[mock_document_sources_with_segmentation[0]],
    )
    result = planner.plan(params, sample_count=4)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    mock_reader.read.assert_called_once_with(
        mock_document_sources_with_segmentation[0].path
    )

    # Verify DocumentSegmenter was called correctly
    mock_segmenter_class.assert_called_once_with(
        mock_document_sources_with_segmentation[0].segmentation_params
    )
    mock_segmenter.segment.assert_called_once_with(mock_document_data[0])

    assert len(result) == 4
    # Each sample should have both document segment attributes and permutable attributes
    for i, sample in enumerate(result):
        # Check document segment attributes (round-robin)
        segment_index = i % len(mock_document_segments[0])
        expected_segment = mock_document_segments[0][segment_index]
        assert sample["doc1_segment"] == expected_segment

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have 3 attributes (1 from document segments + 2 permutable)
        assert len(sample) == 3


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
@patch("oumi.core.synthesis.dataset_planner.DocumentSegmenter")
def test_plan_with_document_source_with_segmentation_keep_original(
    mock_segmenter_class,
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_document_sources_with_segmentation,
    mock_document_data,
    mock_document_segments,
):
    """Test plan with document source that has segmentation and keeps original text."""
    mock_reader = Mock()
    mock_reader.read.return_value = [mock_document_data[0]]
    mock_reader_class.return_value = mock_reader

    mock_segmenter = Mock()
    mock_segmenter.segment.return_value = mock_document_segments[0]
    mock_segmenter_class.return_value = mock_segmenter

    # Use the second document source which has keep_original_text=True
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=[mock_document_sources_with_segmentation[1]],
    )
    result = planner.plan(params, sample_count=4)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    mock_reader.read.assert_called_once_with(
        mock_document_sources_with_segmentation[1].path
    )

    # Verify DocumentSegmenter was called correctly
    mock_segmenter_class.assert_called_once_with(
        mock_document_sources_with_segmentation[1].segmentation_params
    )
    mock_segmenter.segment.assert_called_once_with(mock_document_data[0])

    assert len(result) == 4
    # Each sample should have both document segment attributes, original document, and
    # permutable attributes
    for i, sample in enumerate(result):
        # Check document segment attributes (round-robin)
        segment_index = i % len(mock_document_segments[0])
        expected_segment = mock_document_segments[0][segment_index]
        assert sample["doc2_segment"] == expected_segment

        # Check original document is preserved
        assert sample["doc2"] == mock_document_data[0]

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have 4 attributes
        # (1 from document segments + 1 original document + 2 permutable)
        assert len(sample) == 4


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
@patch("oumi.core.synthesis.dataset_planner.DocumentSegmenter")
def test_plan_with_mixed_document_sources(
    mock_segmenter_class,
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_document_data,
    mock_document_segments,
):
    """Test plan with mix of segmented and non-segmented document sources."""
    mock_reader = Mock()
    mock_reader.read.side_effect = [
        [mock_document_data[0]],  # Document 1 (non-segmented)
        [mock_document_data[1]],  # Document 2 (segmented)
    ]
    mock_reader_class.return_value = mock_reader

    mock_segmenter = Mock()
    mock_segmenter.segment.return_value = mock_document_segments[1]
    mock_segmenter_class.return_value = mock_segmenter

    # Create mixed document sources
    mixed_sources = [
        DocumentSource(path="doc1.pdf", id="doc1"),  # No segmentation
        DocumentSource(
            path="doc2.txt",
            id="doc2",
            segmentation_params=DocumentSegmentationParams(
                id="doc2_segment",
                segment_length=512,
                keep_original_text=False,
            ),
        ),  # With segmentation
    ]

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=mixed_sources,
    )
    result = planner.plan(params, sample_count=6)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    assert mock_reader.read.call_count == 2

    # Verify DocumentSegmenter was called correctly (only for the segmented document)
    mock_segmenter_class.assert_called_once_with(mixed_sources[1].segmentation_params)
    mock_segmenter.segment.assert_called_once_with(mock_document_data[1])

    assert len(result) == 6
    # Each sample should have attributes from both documents and permutable attributes
    for i, sample in enumerate(result):
        # Check first document attributes (non-segmented)
        assert sample["doc1"] == mock_document_data[0]

        # Check second document segment attributes (segmented, round-robin)
        segment_index = i % len(mock_document_segments[1])
        expected_segment = mock_document_segments[1][segment_index]
        assert sample["doc2_segment"] == expected_segment

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have 4 attributes
        # (1 from non-segmented document + 1 from segmented document + 2 permutable)
        assert len(sample) == 4


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
def test_plan_with_document_sources_only(
    mock_reader_class, planner, mock_document_sources, mock_document_data
):
    """Test plan with only document sources and no permutable attributes."""
    mock_reader = Mock()
    mock_reader.read.side_effect = [
        [mock_document_data[0]],  # Document 1
        [mock_document_data[1]],  # Document 2
    ]
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=None,
        input_documents=mock_document_sources,
    )
    result = planner.plan(params, sample_count=3)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    assert mock_reader.read.call_count == 2

    assert len(result) == 3
    # Each sample should only have document attributes
    for i, sample in enumerate(result):
        # Check document attributes
        assert sample["doc1"] == mock_document_data[0]
        assert sample["doc2"] == mock_document_data[1]

        # Should have only document attributes
        assert len(sample) == 2


@patch("oumi.core.synthesis.dataset_planner.DocumentReader")
def test_plan_with_combined_sources_including_documents(
    mock_reader_class,
    planner,
    mock_permutable_attributes,
    mock_document_sources,
    mock_document_data,
    mock_example_sources,
):
    """Test plan with combination of documents, examples, and permutable attributes."""
    mock_reader = Mock()
    mock_reader.read.return_value = [mock_document_data[0]]
    mock_reader_class.return_value = mock_reader

    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        input_documents=[mock_document_sources[0]],
        input_examples=mock_example_sources,
    )
    result = planner.plan(params, sample_count=4)

    # Verify DocumentReader was called correctly
    mock_reader_class.assert_called_once()
    mock_reader.read.assert_called_once_with(mock_document_sources[0].path)

    assert len(result) == 4
    # Each sample should have attributes from all sources
    for i, sample in enumerate(result):
        # Check document attributes
        assert sample["doc1"] == mock_document_data[0]

        # Check example attributes (round-robin)
        example_index = i % 2
        expected_example_attr1 = ["example_value1", "example_value2"][example_index]
        expected_example_attr2 = ["example_valueA", "example_valueB"][example_index]
        assert sample["example_attr1"] == expected_example_attr1
        assert sample["example_attr2"] == expected_example_attr2

        # Check permutable attributes are present
        assert "attr1" in sample
        assert "attr2" in sample

        # Should have 5 attributes (1 from document + 2 from examples + 2 permutable)
        assert len(sample) == 5
