"""Unit tests for DatasetAnalyzer."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import jsonlines
import pytest

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
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

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

    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

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
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

    # Find the message with missing message ID
    null_msg = None
    for msg in results.messages:
        if msg.text_content == "Test message without conversation ID":
            null_msg = msg
            break

    assert null_msg is not None
    assert null_msg.message_id == "msg_3_0"  # Should use fallback


def test_analyze_dataset_empty_conversation(test_data_path, mock_config):
    """Test analysis with conversation containing no messages."""
    config = AnalyzeConfig(
        dataset_name="text_sft",
        split="train",
        sample_count=5,  # Include the empty conversation
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_jsonl_dataset(test_data_path, config)
    results = analyzer.analyze_dataset()

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
    results = analyzer.analyze_dataset()

    first_message = results.messages[0]

    # Check that metrics are prefixed with analyzer ID
    assert "text_length_analyzer_char_count" in first_message.analyzer_metrics
    assert "text_length_analyzer_word_count" in first_message.analyzer_metrics
    assert "text_length_analyzer_analyzer_id" in first_message.analyzer_metrics
    assert "analyzer_2_char_count" in first_message.analyzer_metrics
    assert "analyzer_2_word_count" in first_message.analyzer_metrics
    assert "analyzer_2_analyzer_id" in first_message.analyzer_metrics

    # Check that values are different (different analyzer IDs)
    assert (
        first_message.analyzer_metrics["text_length_analyzer_analyzer_id"]
        == "text_length_analyzer"
    )
    assert first_message.analyzer_metrics["analyzer_2_analyzer_id"] == "analyzer_2"
