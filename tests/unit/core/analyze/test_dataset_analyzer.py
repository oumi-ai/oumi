"""Unit tests for DatasetAnalyzer."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import jsonlines
import pytest

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams
from oumi.datasets import TextSftJsonLinesDataset


class MockSampleAnalyzer:
    """Mock sample analyzer for testing."""

    def __init__(self, config: dict):
        self.config = config
        self.analyze_calls = []

    def analyze_message(self, text_content: str) -> dict:
        """Mock analysis that returns basic metrics."""
        self.analyze_calls.append(text_content)
        return {
            "char_count": len(text_content),
            "word_count": len(text_content.split()),
            "analyzer_id": self.config.get("id", "mock"),
        }


class MockFailingAnalyzer:
    """Mock analyzer that always fails."""

    def __init__(self, config: dict):
        self.config = config

    def analyze_message(self, text_content: str) -> dict:
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
                config={"char_count": True, "word_count": True},
            ),
            SampleAnalyzerParams(id="analyzer_2", config={"analyzer_id": "analyzer_2"}),
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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    # Test result structure
    assert results.dataset_name == "text_sft"
    assert results.total_conversations == 5  # Total in test data
    assert results.conversations_analyzed == 2  # Limited by sample_count
    assert results.total_messages == 4  # 2 messages from each of 2 conversations

    # Test that analyzers were used correctly
    messages = results.messages
    assert len(messages) == 4

    # Check first message has analyzer metrics
    first_message = messages[0]
    assert "text_length_analyzer_char_count" in first_message.analyzer_metrics
    assert "text_length_analyzer_word_count" in first_message.analyzer_metrics
    assert "analyzer_2_char_count" in first_message.analyzer_metrics
    assert "analyzer_2_word_count" in first_message.analyzer_metrics


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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 1
    assert results.total_messages == 2  # Only 2 messages from first conversation

    messages = results.messages
    assert len(messages) == 2
    assert all(msg.conversation_index == 0 for msg in messages)


def test_analyze_dataset_analyzer_failure(test_data_path):
    """Test analysis when an analyzer fails."""
    # Create config with failing analyzer
    config = AnalyzeConfig(
        dataset_name="text_sft",
        split="train",
        sample_count=2,  # Limit to first 2 conversations
        analyzers=[
            SampleAnalyzerParams(id="failing_analyzer", config={}),
        ],
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    analyzer.analyze_dataset()
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    # Should still complete analysis even with failing analyzer
    assert results.total_messages == 4
    assert len(results.messages) == 4

    # Should not have analyzer metrics due to failure
    first_message = results.messages[0]
    assert "failing_analyzer_char_count" not in first_message.analyzer_metrics


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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 5
    assert results.total_messages == 7  # Total messages in all conversations


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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 5  # Should not exceed total
    assert results.total_messages == 7


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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    # Find the message with missing conversation ID
    null_conv_message = None
    for msg in results.messages:
        if msg.text_content == "Test message without conversation ID":
            null_conv_message = msg
            break

    assert null_conv_message is not None
    assert null_conv_message.conversation_id == "conv_3"  # Should use fallback


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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    # Find the message with missing message ID
    null_msg_message = None
    for msg in results.messages:
        if msg.text_content == "Test message without conversation ID":
            null_msg_message = msg
            break

    assert null_msg_message is not None
    assert null_msg_message.message_id == "msg_3_0"  # Should use fallback


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
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    assert results.total_conversations == 5
    assert results.conversations_analyzed == 5
    assert results.total_messages == 7  # Empty conversation contributes 0 messages


def test_analyze_dataset_analyzer_calls(test_data_path, mock_config):
    """Test that analyzers are called with correct parameters."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Check that analyzers were called for each message
    text_length_analyzer = analyzer.sample_analyzers["text_length_analyzer"]
    analyzer_2 = analyzer.sample_analyzers["analyzer_2"]

    assert len(text_length_analyzer.analyze_calls) == 4
    assert len(analyzer_2.analyze_calls) == 4

    # Check first call parameters
    text_content = text_length_analyzer.analyze_calls[0]
    assert text_content == "Hello, how are you?"


def test_analyze_dataset_metric_prefixing(test_data_path, mock_config):
    """Test that analyzer metrics are properly prefixed to avoid conflicts."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()
    results = analyzer.get_analysis_results()
    assert results is not None  # Type assertion for linter

    first_message = results.messages[0]
    # Check that metrics are prefixed with analyzer ID
    assert "text_length_analyzer_char_count" in first_message.analyzer_metrics
    assert "text_length_analyzer_word_count" in first_message.analyzer_metrics
    assert "analyzer_2_char_count" in first_message.analyzer_metrics
    assert "analyzer_2_word_count" in first_message.analyzer_metrics


def test_query_and_filter_methods(test_data_path, mock_config):
    """Test the new query() and filter() methods."""
    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, mock_config)
    analyzer.analyze_dataset()

    # Test query() method - should return DataFrame with analysis results
    query_results = analyzer.query("role == 'user'")
    assert len(query_results) == 2  # 2 user messages in test data
    # Should have analysis columns
    assert "text_length_analyzer_char_count" in query_results.columns
    assert "text_length_analyzer_word_count" in query_results.columns

    # Test filter() method - should return dataset object
    filter_results = analyzer.filter("role == 'user'")
    assert len(filter_results) == 2  # 2 conversations with user messages
    assert hasattr(filter_results, "conversation")  # Should have conversation method
    assert hasattr(filter_results, "dataset_name")  # Should have dataset_name property

    # Test that filtered dataset has the same interface as original
    filtered_conversation = filter_results.conversation(0)
    assert hasattr(filtered_conversation, "messages")  # Same interface


def test_huggingface_dataset_filtering(test_data_path, mock_config):
    """Test unified filtering functionality for HuggingFace datasets."""
    # Create a mock HF dataset by setting dataset_path to None and dataset_name with "/"
    from oumi.core.datasets import BaseMapDataset

    class MockHFDataset(BaseMapDataset):
        def __init__(self):
            self.dataset_path = None
            self.dataset_name = "tatsu-lab/alpaca"
            self.dataset_subset = None
            self.split = "train"
            self.trust_remote_code = False

        def __len__(self):
            return 5

        def conversation(self, idx):
            # Return a mock conversation
            from oumi.core.types.conversation import Conversation, Message, Role

            return Conversation(
                messages=[
                    Message(role=Role.USER, content=f"Test message {idx}"),
                    Message(role=Role.ASSISTANT, content=f"Test response {idx}"),
                ]
            )

        def transform_conversation(self, example):
            # Mock transform method
            from oumi.core.types.conversation import Conversation, Message, Role

            return Conversation(
                messages=[
                    Message(role=Role.USER, content="Test"),
                    Message(role=Role.ASSISTANT, content="Response"),
                ]
            )

        def transform(self, sample):
            """Required abstract method implementation."""
            return sample

    # Create analyzer with mock HF dataset - bypass normal initialization
    analyzer = DatasetAnalyzer.__new__(DatasetAnalyzer)
    analyzer.config = mock_config
    analyzer.dataset_name = mock_config.dataset_name
    analyzer.split = mock_config.split
    analyzer.dataset = MockHFDataset()
    analyzer.sample_analyzers = analyzer._initialize_sample_analyzers()
    analyzer._analysis_results = None
    analyzer._analysis_df = None

    # Mock the analysis results
    from oumi.core.analyze.dataset_analyzer import (
        DatasetAnalysisResult,
        MessageAnalysisResult,
    )

    analyzer._analysis_results = DatasetAnalysisResult(
        dataset_name="tatsu-lab/alpaca",
        total_conversations=5,
        conversations_analyzed=5,
        total_messages=10,
        messages=[
            MessageAnalysisResult(
                conversation_id="conv_1",
                conversation_index=0,
                message_index=0,
                role="user",
                message_id="msg_1_0",
                text_content="Test message 0",
                analyzer_metrics={
                    "text_length_analyzer_char_count": 13,
                    "text_length_analyzer_word_count": 3,
                },
            ),
            MessageAnalysisResult(
                conversation_id="conv_1",
                conversation_index=0,
                message_index=1,
                role="assistant",
                message_id="msg_1_1",
                text_content="Test response 0",
                analyzer_metrics={
                    "text_length_analyzer_char_count": 14,
                    "text_length_analyzer_word_count": 2,
                },
            ),
        ],
    )
    analyzer._analysis_df = analyzer._analysis_results.to_dataframe()

    # Test _is_huggingface_dataset method
    assert analyzer._is_huggingface_dataset() is True

    # Test filtering with HF dataset
    filter_results = analyzer.filter("role == 'user'")

    # Should return a dataset object
    assert hasattr(filter_results, "conversation")
    assert hasattr(filter_results, "__len__")
    assert hasattr(filter_results, "dataset_name")

    # Should have the expected interface
    assert filter_results.dataset_name == "tatsu-lab/alpaca_filtered"


def test_custom_dataset_filtering(test_data_path, mock_config):
    """Test unified filtering functionality for custom datasets."""
    # Create a mock custom dataset by setting dataset_path
    from oumi.core.datasets import BaseMapDataset

    class MockCustomDataset(BaseMapDataset):
        def __init__(self):
            self.dataset_path = "/path/to/local/data.jsonl"
            self.dataset_name = "custom_dataset"
            self.dataset_subset = None
            self.split = "train"
            self.trust_remote_code = False

        def __len__(self):
            return 5

        def conversation(self, idx):
            # Return a mock conversation
            from oumi.core.types.conversation import Conversation, Message, Role

            return Conversation(
                messages=[
                    Message(role=Role.USER, content=f"Test message {idx}"),
                    Message(role=Role.ASSISTANT, content=f"Test response {idx}"),
                ]
            )

        def transform(self, sample):
            """Required abstract method implementation."""
            return sample

    # Create analyzer with mock custom dataset - bypass normal initialization
    analyzer = DatasetAnalyzer.__new__(DatasetAnalyzer)
    analyzer.config = mock_config
    analyzer.dataset_name = mock_config.dataset_name
    analyzer.split = mock_config.split
    analyzer.dataset = MockCustomDataset()
    analyzer.sample_analyzers = analyzer._initialize_sample_analyzers()
    analyzer._analysis_results = None
    analyzer._analysis_df = None

    # Mock the analysis results
    from oumi.core.analyze.dataset_analyzer import (
        DatasetAnalysisResult,
        MessageAnalysisResult,
    )

    analyzer._analysis_results = DatasetAnalysisResult(
        dataset_name="custom_dataset",
        total_conversations=5,
        conversations_analyzed=5,
        total_messages=10,
        messages=[
            MessageAnalysisResult(
                conversation_id="conv_1",
                conversation_index=0,
                message_index=0,
                role="user",
                message_id="msg_1_0",
                text_content="Test message 0",
                analyzer_metrics={
                    "text_length_analyzer_char_count": 13,
                    "text_length_analyzer_word_count": 3,
                },
            ),
            MessageAnalysisResult(
                conversation_id="conv_1",
                conversation_index=0,
                message_index=1,
                role="assistant",
                message_id="msg_1_1",
                text_content="Test response 0",
                analyzer_metrics={
                    "text_length_analyzer_char_count": 14,
                    "text_length_analyzer_word_count": 2,
                },
            ),
        ],
    )
    analyzer._analysis_df = analyzer._analysis_results.to_dataframe()

    # Test _is_huggingface_dataset method
    assert analyzer._is_huggingface_dataset() is False

    # Test filtering with custom dataset
    filter_results = analyzer.filter("role == 'user'")

    # Should return a dataset object
    assert hasattr(filter_results, "conversation")
    assert hasattr(filter_results, "__len__")
    assert hasattr(filter_results, "dataset_name")

    # Should have the expected interface
    assert filter_results.dataset_name == "custom_dataset_filtered"
