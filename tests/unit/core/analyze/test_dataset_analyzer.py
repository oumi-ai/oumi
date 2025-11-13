"""Unit tests for DatasetAnalyzer."""

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import jsonlines
import pandas as pd
import pytest

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.dataset_analyzer import (
    DatasetAnalyzer,
)
from oumi.core.configs import AnalyzeConfig, DatasetSource, SampleAnalyzerParams
from oumi.core.datasets import BaseMapDataset
from oumi.datasets import TextSftJsonLinesDataset


def check_no_nans(obj):
    """Recursively check for NaN values in nested dictionaries."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Only check for NaN in numeric values, not lists or strings
            if isinstance(value, (int, float)) and pd.isna(value):
                raise AssertionError(f"Found NaN value in key '{key}': {value}")
            check_no_nans(value)
    elif isinstance(obj, list):
        for item in obj:
            check_no_nans(item)
    elif isinstance(obj, (int, float)) and pd.isna(obj):
        raise AssertionError(f"Found NaN value: {obj}")
    else:
        # Other types (str, bool, None, etc.) are not checked for NaN
        pass


class MockSampleAnalyzer:
    """Mock sample analyzer for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.analyze_calls = []
        # Extract analyzer ID from config
        self.analyzer_id = kwargs.get("analyzer_id", "mock")

    def analyze_sample(
        self, df: pd.DataFrame, schema: Optional[dict] = None
    ) -> pd.DataFrame:
        """
        Mock analysis that adds analyzer metrics to the DataFrame.
        """
        self.analyze_calls.append(df)

        result_df = df.copy()

        # Add mock analyzer metrics for text content columns
        if schema:
            text_columns = [
                col
                for col, config in schema.items()
                if config.get("content_type") == ContentType.TEXT and col in df.columns
            ]

            for text_col in text_columns:
                # Add char_count and word_count metrics for each text column
                result_df[f"{text_col}_{self.analyzer_id}_char_count"] = (
                    df[text_col].astype(str).str.len()
                )
                result_df[f"{text_col}_{self.analyzer_id}_word_count"] = (
                    df[text_col].astype(str).str.split().str.len()
                )

        return result_df


class MockFailingAnalyzer:
    """Mock analyzer that always fails."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def analyze_sample(
        self, df: pd.DataFrame, schema: Optional[dict] = None
    ) -> pd.DataFrame:
        raise ValueError("Analyzer failed")


class MockDataset(BaseMapDataset):
    """Mock dataset for testing."""

    def __init__(self):
        super().__init__(dataset_name="mock_dataset")
        self._data = pd.DataFrame({"dummy": [1, 2, 3, 4, 5]})

    def transform(self, sample: pd.Series) -> dict:
        """Transform a sample to the expected format."""
        return {"dummy": sample["dummy"]}

    def __len__(self):
        return 5


class MockRegistry:
    """Mock registry for testing."""

    def get_sample_analyzer(self, analyzer_id: str):
        """Get a mock analyzer class that remembers its ID."""
        if analyzer_id == "failing_analyzer":
            return MockFailingAnalyzer

        # Create a subclass that remembers the analyzer_id
        class SpecificMockAnalyzer(MockSampleAnalyzer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.analyzer_id = analyzer_id  # Use the registry ID

        return SpecificMockAnalyzer


@pytest.fixture
def test_data():
    """Sample conversation data for testing."""
    return [
        {
            "conversation_id": "conv_1",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                    "id": "msg_1_0",
                },
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you!",
                    "id": "msg_1_1",
                },
            ],
        },
        {
            "conversation_id": "conv_2",
            "messages": [
                {
                    "role": "user",
                    "content": "What is 2+2?",
                    "id": "msg_2_0",
                },
                {
                    "role": "assistant",
                    "content": "2+2 equals 4.",
                    "id": "msg_2_1",
                },
            ],
        },
        {
            "conversation_id": "conv_3",
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke",
                    "id": "msg_3_0",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Why don't scientists trust atoms? "
                        "Because they make up everything!"
                    ),
                    "id": "msg_3_1",
                },
            ],
        },
        {
            "conversation_id": None,
            "messages": [
                {
                    "role": "user",
                    "content": "Test message without conversation ID",
                    "id": None,
                },
            ],
        },
        {
            "conversation_id": "conv_5",
            "messages": [],
        },
    ]


@pytest.fixture
def single_conversation_test_data():
    """Single conversation data for testing."""
    return [
        {
            "conversation_id": "conv_1",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                    "id": "msg_1_0",
                }
            ],
        },
    ]


@pytest.fixture
def test_data_path(test_data):
    """Create a temporary JSONL file with test data."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(test_data)

    yield Path(f.name)
    Path(f.name).unlink()  # Cleanup temp file


@pytest.fixture
def single_conversation_test_data_path(single_conversation_test_data):
    """Create a temporary JSONL file with single conversation test data."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(single_conversation_test_data)

    yield Path(f.name)
    Path(f.name).unlink()  # Cleanup temp file


@pytest.fixture
def mock_config():
    """Create a mock analyzer configuration."""
    return AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=2,
        output_path="./test_output",
        analyzers=[
            SampleAnalyzerParams(
                id="length_analyzer",
                params={"char_count": True, "word_count": True},
            ),
            SampleAnalyzerParams(id="analyzer_2", params={"analyzer_id": "analyzer_2"}),
        ],
    )


def create_analyzer_with_jsonl_dataset(test_data_path, config):
    """Helper function to create analyzer with JSONL dataset."""
    # Create a real TextSftJsonLinesDataset from the JSONL file
    dataset = TextSftJsonLinesDataset(dataset_path=test_data_path)

    with patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry()):
        with patch(
            "oumi.core.analyze.dataset_analyzer.load_dataset_from_config"
        ) as mock_load:
            mock_load.return_value = dataset
            analyzer = DatasetAnalyzer(config)
            return analyzer, dataset


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
def test_analyzer_initialization(mock_load, mock_config):
    """Test DatasetAnalyzer initialization."""
    mock_load.return_value = "mock_dataset"

    analyzer = DatasetAnalyzer(mock_config)

    # Test basic initialization
    assert analyzer.config == mock_config
    assert analyzer.dataset_name == "text_sft"
    assert analyzer.split == "train"

    # Test that analyzers were initialized correctly
    assert len(analyzer.sample_analyzers) == 2
    assert "length_analyzer" in analyzer.sample_analyzers
    assert "analyzer_2" in analyzer.sample_analyzers


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
def test_analyzer_initialization_with_dataset(mock_load, mock_config, test_data_path):
    """Test DatasetAnalyzer initialization with optional dataset parameter."""
    # Create a real dataset from test data
    dataset = TextSftJsonLinesDataset(dataset_path=test_data_path)

    # Create a config with DIRECT mode for this test
    direct_config = AnalyzeConfig(
        dataset_source=DatasetSource.DIRECT,
        dataset_name="text_sft",
        sample_count=2,
        output_path="./test_output",
        analyzers=mock_config.analyzers,
    )

    # Test initialization with provided dataset
    analyzer = DatasetAnalyzer(direct_config, dataset=dataset)

    # Test basic initialization
    assert analyzer.config == direct_config
    assert analyzer.dataset_name == "text_sft"

    # Test that the provided dataset was used instead of loading from config
    assert analyzer.dataset == dataset
    # Check dataset size (only for map datasets that support len())
    from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

    if not isinstance(analyzer.dataset, BaseIterableDataset):
        assert len(analyzer.dataset) == 5  # Should have 5 conversations from test data

    # Test that load_dataset_from_config was not called
    mock_load.assert_not_called()

    # Test that analyzers were initialized correctly
    assert len(analyzer.sample_analyzers) == 2
    assert "length_analyzer" in analyzer.sample_analyzers
    assert "analyzer_2" in analyzer.sample_analyzers


def test_dataset_source_direct_with_dataset_success():
    """Test that DatasetSource.DIRECT works when dataset is provided."""

    dataset = MockDataset()

    config = AnalyzeConfig(
        dataset_source=DatasetSource.DIRECT,
        dataset_name="test_dataset",
        analyzers=[SampleAnalyzerParams(id="test_analyzer", params={})],
    )

    # This should work without error
    analyzer = DatasetAnalyzer(config, dataset=dataset)
    assert analyzer.dataset == dataset
    assert analyzer.dataset_name == "test_dataset"


def test_dataset_source_direct_without_dataset_failure():
    """Test that DatasetSource.DIRECT fails when no dataset is provided."""
    config = AnalyzeConfig(
        dataset_source=DatasetSource.DIRECT,
        dataset_name="test_dataset",
        analyzers=[SampleAnalyzerParams(id="test_analyzer", params={})],
    )

    with pytest.raises(
        ValueError,
        match="Config specifies dataset_source=DatasetSource.DIRECT but no dataset "
        "was provided",
    ):
        DatasetAnalyzer(config)


def test_dataset_source_config_with_dataset_failure():
    """Test that DatasetSource.CONFIG fails when dataset is provided."""

    dataset = MockDataset()

    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,
        dataset_name="test_dataset",
        analyzers=[SampleAnalyzerParams(id="test_analyzer", params={})],
    )

    with pytest.raises(
        ValueError,
        match="Dataset provided but config.dataset_source is 'config'. When using "
        "DatasetSource.CONFIG, do not pass a dataset to the constructor",
    ):
        DatasetAnalyzer(config, dataset=dataset)


def test_analyze_dataset_integration(test_data_path, mock_config):
    """Test DatasetAnalyzer analysis integration."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    # Test result structure
    assert results.dataset_name == "text_sft"
    assert results.total_conversations == 5  # Total in test data
    assert results.conversations_analyzed == 2  # Limited by sample_count

    # Test that analyzers were used correctly by checking the DataFrame
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    assert len(analysis_df) > 0

    # Check that analyzer metrics are present in the DataFrame
    message_columns = [col for col in analysis_df.columns if col.startswith("message_")]
    assert len(message_columns) > 0


def test_analyze_dataset_with_sample_limit(test_data_path, mock_config):
    """Test analysis with sample count limit."""
    # Create config with sample_count=1 (only analyze first conversation)
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=1,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 1

    # Test that only one conversation was analyzed
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    unique_conversations = analysis_df["conversation_index"].unique()
    assert len(unique_conversations) == 1
    assert unique_conversations[0] == 0


def test_analyze_dataset_analyzer_failure(test_data_path):
    """Test analysis when an analyzer fails."""
    # Create config with failing analyzer
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=2,  # Limit to first 2 conversations
        analyzers=[
            SampleAnalyzerParams(id="failing_analyzer", params={}),
        ],
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    # Should still complete analysis even with failing analyzer
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    assert len(analysis_df) > 0

    # Should not have analyzer metrics due to failure
    failing_analyzer_columns = [
        col for col in analysis_df.columns if "failing_analyzer" in col
    ]
    assert len(failing_analyzer_columns) == 0


def test_analyze_dataset_no_analyzers(test_data_path):
    """Test that DatasetAnalyzer raises an error when no analyzers are configured."""
    # Create config with no analyzers
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        analyzers=[],
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)

    # Should raise an error when trying to analyze without analyzers
    with pytest.raises(ValueError, match="No analyzers configured for analysis"):
        analyzer.analyze_dataset()


def test_analyze_dataset_sample_count_none(test_data_path, mock_config):
    """Test analysis with sample_count=None (analyze all conversations)."""
    # Create config with sample_count=None
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=None,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 5

    # Test that all conversations were analyzed
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    unique_conversations = analysis_df["conversation_index"].unique()
    assert len(unique_conversations) == 5  # All conversations analyzed


def test_analyze_dataset_sample_count_zero(test_data_path, mock_config):
    """Test analysis with sample_count=0 raises ValueError."""
    # Create config with sample_count=0
    with pytest.raises(ValueError, match="`sample_count` must be greater than 0."):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name="text_sft",
            split="train",
            sample_count=0,
            analyzers=mock_config.analyzers,
        )


def test_analyze_dataset_sample_count_negative(test_data_path, mock_config):
    """Test analysis with negative sample_count raises ValueError."""
    # Create config with negative sample_count
    with pytest.raises(ValueError, match="`sample_count` must be greater than 0."):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name="text_sft",
            split="train",
            sample_count=-5,
            analyzers=mock_config.analyzers,
        )


def test_analyze_dataset_sample_count_exceeds_total(test_data_path, mock_config):
    """Test analysis when sample_count exceeds total conversations."""
    # Create config with sample_count exceeding total
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=10,  # More than total conversations
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 5  # Should not exceed total

    # Test that all conversations were analyzed
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    unique_conversations = analysis_df["conversation_index"].unique()
    assert len(unique_conversations) == 5  # Should not exceed total


def test_analyze_dataset_missing_conversation_id(test_data_path, mock_config):
    """Test analysis when conversation_id is None."""
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=4,  # Include the conversation with null ID
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    # Test that the conversation with missing ID was handled correctly
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None

    # Check that conversation_id fallback was used
    conv_3_rows = analysis_df[analysis_df["conversation_id"] == "conv_3"]
    assert len(conv_3_rows) > 0


def test_analyze_dataset_missing_message_id(test_data_path, mock_config):
    """Test analysis when message_id is None."""
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=4,  # Include the conversation with null message ID
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    # Test that the message with missing ID was handled correctly
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None

    # Check that message_id fallback was used
    msg_3_0_rows = analysis_df[analysis_df["message_id"] == "msg_3_0"]
    assert len(msg_3_0_rows) > 0


def test_analyze_dataset_empty_conversation(test_data_path, mock_config):
    """Test analysis with empty conversation."""
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=5,  # Include the empty conversation
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.analysis_results
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 5

    # Test that all conversations were analyzed including empty ones
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    unique_conversations = analysis_df["conversation_index"].unique()
    assert len(unique_conversations) == 5  # All conversations analyzed


def test_analyze_dataset_analyzer_calls(test_data_path, mock_config):
    """Test that analyzers are called correctly."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Check that the mock analyzer was called
    mock_analyzer = analyzer.sample_analyzers["length_analyzer"]
    assert len(mock_analyzer.analyze_calls) == 2  # Called for each conversation


def test_query_method(test_data_path, mock_config):
    """Test the query method functionality."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Test basic query
    results = analyzer.query("role == 'user'")
    assert len(results) > 0
    assert all(row["role"] == "user" for _, row in results.iterrows())

    # Test query with analyzer metrics
    results = analyzer.query("text_content_length_analyzer_char_count > 10")
    assert len(results) > 0
    assert all(
        row["text_content_length_analyzer_char_count"] > 10
        for _, row in results.iterrows()
    )


def test_query_with_empty_results(test_data_path, mock_config):
    """Test behavior when query returns no results."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Use a query that should return no results
    query_results = analyzer.query("role == 'nonexistent_role'")

    # Should return an empty DataFrame
    assert isinstance(query_results, pd.DataFrame)
    assert len(query_results) == 0


def test_query_complex_expressions_examples(test_data_path, mock_config):
    """Test query method with complex expressions - shows usage examples."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Test various query expressions with proper validation
    queries = [
        "text_content_length_analyzer_word_count < 10",  # Short messages
        "role == 'assistant'",  # Filter for assistant messages
        "role == 'user' and text_content_length_analyzer_word_count > 5",  # Long
        # messages
        "role == 'user' or role == 'assistant'",  # Any user or assistant messages
        # Medium-length
        "text_content_length_analyzer_char_count > 10 and "
        "text_content_length_analyzer_word_count < 20",
    ]

    for query in queries:
        try:
            results = analyzer.query(query)
            assert isinstance(results, pd.DataFrame)
            # Query should not raise an exception and should return valid DataFrame
        except Exception as e:
            pytest.fail(f"Query '{query}' failed: {e}")


def test_filter_method(test_data_path, mock_config):
    """Test the filter method returns dataset with correct interface."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Filter returns dataset
    filter_results = analyzer.filter("role == 'user'")

    # Test that filtered dataset has required methods
    assert hasattr(filter_results, "conversation")
    assert hasattr(filter_results, "dataset_name")

    # Check length only for map datasets
    from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

    if not isinstance(filter_results, BaseIterableDataset):
        assert hasattr(filter_results, "__len__")
        assert len(filter_results) > 0
        # Test that we can access conversations
        if len(filter_results) > 0:
            first_conv = filter_results.conversation(0)
            assert hasattr(first_conv, "messages")
            assert len(first_conv.messages) > 0
    else:
        # For iterable datasets, just test that we can get a conversation
        first_conv = filter_results.conversation(0)
        assert hasattr(first_conv, "messages")
        assert len(first_conv.messages) > 0


def test_class_preservation_in_filtered_dataset(test_data_path, mock_config):
    """Test that filtered dataset preserves the original class."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Get original and filtered datasets
    original_dataset = analyzer.dataset
    filtered_dataset = analyzer.filter("role == 'user'")

    # Test class preservation
    original_class = type(original_dataset)
    filtered_class = type(filtered_dataset)

    # Filtered dataset should inherit from original class
    assert issubclass(filtered_class, original_class)

    # Both should have the same base functionality
    assert hasattr(original_dataset, "conversation")
    assert hasattr(filtered_dataset, "conversation")


def test_filtered_dataset_naming(test_data_path, mock_config):
    """Test that filtered datasets have appropriate names."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Test that filtered dataset has appropriate name
    filtered_dataset = analyzer.filter("role == 'user'")
    assert filtered_dataset.dataset_name.endswith("_filtered")


def test_empty_filter_results(test_data_path, mock_config):
    """Test behavior when filter returns no results."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Use a filter that should return no results
    filtered_dataset = analyzer.filter("role == 'nonexistent_role'")

    # Should return an empty dataset (check only for map datasets)
    from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

    if not isinstance(filtered_dataset, BaseIterableDataset):
        assert len(filtered_dataset) == 0
    assert hasattr(filtered_dataset, "conversation")
    assert hasattr(filtered_dataset, "dataset_name")


def test_invalid_expressions(test_data_path, mock_config):
    """Test that invalid expressions raise appropriate errors."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Test various invalid expressions
    invalid_expressions = [
        "invalid_column == 'value'",  # Non-existent column
        "role == 'user' and invalid_column > 5",  # Invalid column in compound
        "role == 'user' or invalid_column < 10",  # Invalid column in OR expression
    ]

    for expression in invalid_expressions:
        # Both query and filter should fail with the same invalid expression
        with pytest.raises((ValueError, KeyError)):
            analyzer.query(expression)

        with pytest.raises((ValueError, KeyError)):
            analyzer.filter(expression)


def test_generate_analysis_summary(test_data_path, mock_config):
    """Test the _generate_analysis_summary method."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)

    # Test that summary is not available before analysis
    with pytest.raises(RuntimeError, match="Analysis has not been run yet"):
        _ = analyzer.analysis_summary

    # Run analysis to generate summary
    analyzer.analyze_dataset()

    # Test that summary is now available
    summary = analyzer.analysis_summary
    assert summary is not None
    assert isinstance(summary, dict)

    # Test summary structure
    expected_keys = [
        "dataset_overview",
        "message_level_summary",
        "conversation_level_summary",
        "conversation_turns",
    ]
    for key in expected_keys:
        assert key in summary

    # Test dataset overview
    overview = summary["dataset_overview"]
    assert overview["dataset_name"] == "text_sft"
    assert overview["total_conversations"] == 5
    assert overview["conversations_analyzed"] == 2
    assert "dataset_coverage_percentage" in overview
    assert "total_messages" in overview
    assert "analyzers_used" in overview

    # Test message level summary - analyzer names with underscores get split
    message_summary = summary["message_level_summary"]
    # The analyzer names get split on underscores, so check for the actual keys
    assert len(message_summary) > 0
    # Check that we have some analyzer metrics
    for analyzer_name, metrics in message_summary.items():
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    # Test conversation level summary - may be empty if no conversation-level metrics
    conversation_summary = summary["conversation_level_summary"]
    assert isinstance(conversation_summary, dict)  # Should be a dict, but may be empty

    # Test conversation turns statistics (now at top level)
    turns_stats = summary["conversation_turns"]
    assert "count" in turns_stats
    assert "mean" in turns_stats
    assert "std" in turns_stats
    assert "min" in turns_stats
    assert "max" in turns_stats
    assert "median" in turns_stats


def test_generate_analysis_summary_single_conversation_no_nan(
    single_conversation_test_data_path, mock_config
):
    """Test single conversation and no NaN values."""
    # Create a config with sample_count=1 to only analyze 1 conversation
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,
        dataset_name="text_sft",
        split="train",
        sample_count=1,
        output_path="./test_output",
        analyzers=mock_config.analyzers,
    )
    analyzer, _ = create_analyzer_with_jsonl_dataset(
        single_conversation_test_data_path, config
    )

    # Run analysis to generate summary
    analyzer.analyze_dataset()

    # Test that summary is now available
    summary = analyzer.analysis_summary
    assert summary is not None
    assert isinstance(summary, dict)

    # Test dataset overview
    overview = summary["dataset_overview"]
    assert overview["dataset_name"] == "text_sft"
    assert overview["total_conversations"] == 1
    assert overview["conversations_analyzed"] == 1

    # Test that there are no NaN values in any statistics
    check_no_nans(summary)

    # Test that std is 0.0 for single conversation (since there's no variance)
    conversation_summary = summary["conversation_level_summary"]
    for analyzer_name, metrics in conversation_summary.items():
        for metric_name, stats in metrics.items():
            if isinstance(stats, dict) and "std" in stats:
                assert stats["std"] == 0.0, (
                    f"Expected std=0.0 for single conversation in "
                    f"{analyzer_name}.{metric_name}, got {stats['std']}"
                )

    # Verify conversation turns std is also 0.0 for single conversation
    turns_stats = summary["conversation_turns"]
    assert turns_stats["std"] == 0.0, (
        f"Expected conversation_turns std=0.0 for single conversation, "
        f"got {turns_stats['std']}"
    )


def test_analyzer_with_tokenizer(test_data_path):
    """Test that tokenizer is properly built from config and passed to analyzers."""
    config = AnalyzeConfig(
        dataset_source=DatasetSource.CONFIG,  # Required field
        dataset_name="text_sft",
        split="train",
        sample_count=2,
        tokenizer_config={
            "model_name": "gpt2"
        },  # This will be used to build a real tokenizer
        analyzers=[
            SampleAnalyzerParams(
                id="length_analyzer",
                params={"char_count": True, "word_count": True, "token_count": True},
            ),
        ],
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)

    # Verify that tokenizer was built from config
    assert analyzer.tokenizer is not None
    assert hasattr(analyzer.tokenizer, "encode")

    # Run analysis to trigger tokenizer usage
    analyzer.analyze_dataset()

    # Check that analysis completed successfully
    results = analyzer.analysis_results
    assert results is not None

    # Check that we have messages in the results
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    assert len(analysis_df) > 0
