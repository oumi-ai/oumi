"""Unit tests for DatasetAnalyzer."""

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jsonlines
import pandas as pd
import pytest

from oumi.core.analyze.dataset_analyzer import (
    ConversationAnalysisResult,
    DatasetAnalyzer,
    MessageAnalysisResult,
    SampleAnalysisResult,
)
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams
from oumi.datasets import TextSftJsonLinesDataset


class MockSampleAnalyzer:
    """Mock sample analyzer for testing."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.analyze_calls = []
        # Extract analyzer ID from config
        self.analyzer_id = kwargs.get("analyzer_id", "mock")

    def compute_metrics(self, conversation, tokenizer=None) -> SampleAnalysisResult:
        """
        Mock analysis that returns both message-level and conversation-level metrics.
        """
        self.analyze_calls.append(conversation)

        # Compute message-level metrics
        message_results = []
        for msg_idx, message in enumerate(conversation.messages):
            if isinstance(message.content, str):
                text_content = message.content
            else:
                text_content = message.compute_flattened_text_content()

            # Create message metrics
            message_metrics = {
                "char_count": len(text_content),
                "word_count": len(text_content.split()),
                "analyzer_id": self.analyzer_id,
            }

            # Create MessageAnalysisResult
            message_result = MessageAnalysisResult(
                message_index=msg_idx,
                role=message.role.value,
                message_id=message.id or f"msg_{msg_idx}",
                text_content=text_content,
                analyzer_metrics=message_metrics,
            )
            message_results.append(message_result)

        # Compute conversation-level metrics
        conversation_text = ""
        for message in conversation.messages:
            if isinstance(message.content, str):
                text_content = message.content
            else:
                text_content = message.compute_flattened_text_content()
            conversation_text += f"{message.role.value}: {text_content}\n"

        # Create conversation metrics
        conversation_metrics = {
            "char_count": len(conversation_text),
            "word_count": len(conversation_text.split()),
            "analyzer_id": self.analyzer_id,
        }

        # Use tokenizer if provided (to verify it's passed correctly)
        if tokenizer:
            tokenizer.encode(conversation_text, add_special_tokens=False)

        # Create ConversationAnalysisResult
        conversation_result = ConversationAnalysisResult(
            analyzer_metrics=conversation_metrics,
        )

        # Create and return SampleAnalysisResult
        return SampleAnalysisResult(
            conversation_id=conversation.conversation_id or "unknown",
            conversation_index=0,  # Single conversation
            messages=message_results,
            conversation=conversation_result,
        )


class MockFailingAnalyzer:
    """Mock analyzer that always fails."""

    def __init__(self, **kwargs):
        self.config = kwargs

    def compute_metrics(self, conversation, tokenizer=None) -> dict[str, Any]:
        raise ValueError("Analyzer failed")


class MockRegistry:
    """Mock registry for testing."""

    def get_sample_analyzer(self, analyzer_id: str):
        """Get a mock analyzer class."""
        if analyzer_id == "failing_analyzer":
            return MockFailingAnalyzer
        return MockSampleAnalyzer


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
def test_data_path(test_data):
    """Create a temporary JSONL file with test data."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        with jsonlines.Writer(f) as writer:
            writer.write_all(test_data)

    yield Path(f.name)
    Path(f.name).unlink()  # Cleanup temp file


@pytest.fixture
def mock_config():
    """Create a mock analyzer configuration."""
    return AnalyzeConfig(
        dataset_name="text_sft",
        split="train",
        sample_count=2,
        output_path="./test_output",
        analyzers=[
            SampleAnalyzerParams(
                id="text_length_analyzer",
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
    assert "text_length_analyzer" in analyzer.sample_analyzers
    assert "analyzer_2" in analyzer.sample_analyzers


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
    config = AnalyzeConfig(dataset_name="text_sft", analyzers=[])

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)

    # Should raise an error when trying to analyze without analyzers
    with pytest.raises(ValueError, match="No analyzers configured for analysis"):
        analyzer.analyze_dataset()


def test_analyze_dataset_sample_count_none(test_data_path, mock_config):
    """Test analysis with sample_count=None (analyze all conversations)."""
    # Create config with sample_count=None
    config = AnalyzeConfig(
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
    config = AnalyzeConfig(
        dataset_name="text_sft",
        split="train",
        sample_count=0,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    with pytest.raises(ValueError, match="sample_count must be positive"):
        analyzer.analyze_dataset()


def test_analyze_dataset_sample_count_negative(test_data_path, mock_config):
    """Test analysis with negative sample_count raises ValueError."""
    # Create config with negative sample_count
    config = AnalyzeConfig(
        dataset_name="text_sft",
        split="train",
        sample_count=-5,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    with pytest.raises(ValueError, match="sample_count must be positive"):
        analyzer.analyze_dataset()


def test_analyze_dataset_sample_count_exceeds_total(test_data_path, mock_config):
    """Test analysis when sample_count exceeds total conversations."""
    # Create config with sample_count exceeding total
    config = AnalyzeConfig(
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
    mock_analyzer = analyzer.sample_analyzers["text_length_analyzer"]
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
    results = analyzer.query("message_text_length_analyzer_char_count > 10")
    assert len(results) > 0
    assert all(
        row["message_text_length_analyzer_char_count"] > 10
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
        "message_text_length_analyzer_word_count < 10",  # Filter for short messages
        "role == 'assistant'",  # Filter for assistant messages
        "role == 'user' and message_text_length_analyzer_word_count > 5",  # Long user
        # messages
        "role == 'user' or role == 'assistant'",  # Any user or assistant messages
        # Medium-length
        "message_text_length_analyzer_char_count > 10 and "
        "message_text_length_analyzer_word_count < 20",
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
    assert hasattr(filter_results, "__len__")
    assert hasattr(filter_results, "dataset_name")
    assert len(filter_results) > 0

    # Test that we can access conversations
    if len(filter_results) > 0:
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

    # Should return an empty dataset
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


def test_analyzer_with_tokenizer(test_data_path):
    """Test that tokenizer is properly passed to analyzers."""
    from unittest.mock import Mock

    # Create a mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [1, 2, 3]  # 3 tokens

    # Create config with tokenizer
    config = AnalyzeConfig(
        dataset_name="text_sft",
        split="train",
        sample_count=2,
        tokenizer=mock_tokenizer,
        analyzers=[
            SampleAnalyzerParams(
                id="text_length_analyzer",
                params={"char_count": True, "word_count": True, "token_count": True},
            ),
        ],
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)

    # Run analysis to trigger tokenizer usage
    analyzer.analyze_dataset()

    # Check that analysis completed successfully
    results = analyzer.analysis_results
    assert results is not None

    # Check that we have messages in the results
    analysis_df = analyzer.analysis_df
    assert analysis_df is not None
    assert len(analysis_df) > 0

    # Verify that the mock tokenizer was actually called
    assert mock_tokenizer.encode.call_count > 0
