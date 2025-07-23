"""Unit tests for DatasetAnalyzer."""

from unittest.mock import patch

import pytest

from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams


class MockRegistry:
    """Mock registry for testing."""

    def get_sample_analyzer(self, analyzer_id: str):
        """Get a mock analyzer class."""
        return MockSampleAnalyzer


class MockSampleAnalyzer:
    """Mock sample analyzer for testing."""

    def __init__(self, config: dict):
        self.config = config
        self.analyze_calls = []

    def analyze_message(self, text_content: str, message_metadata: dict) -> dict:
        """Mock analysis that returns basic metrics."""
        self.analyze_calls.append((text_content, message_metadata))
        return {
            "char_count": len(text_content),
            "word_count": len(text_content.split()),
            "analyzer_id": self.config.get("id", "mock"),
        }


class MockFailingAnalyzerRegistry:
    """Mock registry that returns failing analyzers."""

    def get_sample_analyzer(self, analyzer_id: str):
        return MockFailingAnalyzer


class MockFailingAnalyzer:
    """Mock analyzer that always fails."""

    def __init__(self, config: dict):
        self.config = config

    def analyze_message(self, text_content: str, message_metadata: dict) -> dict:
        raise ValueError("Analyzer failed")


@pytest.fixture
def mock_config():
    """Create a mock analyzer configuration."""
    return AnalyzeConfig(
        dataset_name="test_dataset",
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


@pytest.fixture
def conversations():
    """Create conversations with multiple messages for testing."""
    from tests.unit.utils.test_analysis_utils import MockConversation, MockMessage

    return [
        MockConversation(
            "conv_1",
            [
                MockMessage("Hello, how are you?", "user", "msg_1_0"),
                MockMessage("I'm doing well, thank you!", "assistant", "msg_1_1"),
            ],
        ),
        MockConversation(
            "conv_2",
            [
                MockMessage("What is 2+2?", "user", "msg_2_0"),
                MockMessage("2+2 equals 4.", "assistant", "msg_2_1"),
            ],
        ),
    ]


def create_analyzer_with_dataset(
    conversations, config, registry_class: type = MockRegistry
):
    """Helper function to create analyzer with mock dataset."""
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
    from tests.unit.utils.test_analysis_utils import MockDataset

    mock_dataset = MockDataset(conversations)

    with patch("oumi.core.analyze.dataset_analyzer.REGISTRY", registry_class()):
        with patch(
            "oumi.core.analyze.dataset_analyzer.load_dataset_from_config"
        ) as mock_load:
            mock_load.return_value = mock_dataset
            analyzer = DatasetAnalyzer(config)
            return analyzer, mock_dataset


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
def test_analyzer_initialization(mock_load, mock_config):
    """Test DatasetAnalyzer initialization."""
    mock_load.return_value = "mock_dataset"

    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    analyzer = DatasetAnalyzer(mock_config)

    # Test basic initialization
    assert analyzer.config == mock_config
    assert analyzer.dataset_name == "test_dataset"
    assert analyzer.split == "train"

    # Test that analyzers were initialized correctly
    assert len(analyzer.sample_analyzers) == 2
    assert "text_length_analyzer" in analyzer.sample_analyzers
    assert "analyzer_2" in analyzer.sample_analyzers


def test_analyze_dataset_integration(conversations, mock_config):
    """Test DatasetAnalyzer analysis integration."""
    analyzer, _ = create_analyzer_with_dataset(conversations, mock_config)
    results = analyzer.analyze_dataset()

    # Test result structure
    assert results["dataset_name"] == "test_dataset"
    assert "sample_level_results" in results

    sample_results = results["sample_level_results"]
    assert sample_results["total_conversations"] == 2
    assert sample_results["conversations_analyzed"] == 2
    assert sample_results["total_messages"] == 4

    # Test that analyzers were used correctly
    messages = sample_results["messages"]
    assert len(messages) == 4

    # Check first message has analyzer metrics
    first_message = messages[0]
    assert "text_length_analyzer_char_count" in first_message
    assert "text_length_analyzer_word_count" in first_message
    assert "analyzer_2_char_count" in first_message
    assert "analyzer_2_word_count" in first_message


def test_analyze_dataset_with_sample_limit(conversations, mock_config):
    """Test analysis with sample count limit."""
    # Create config with sample_count=1 (only analyze first conversation)
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=1,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_dataset(conversations, config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert sample_results["total_conversations"] == 2
    assert sample_results["conversations_analyzed"] == 1
    assert (
        sample_results["total_messages"] == 2
    )  # Only 2 messages from first conversation

    messages = sample_results["messages"]
    assert len(messages) == 2
    assert all(msg["conversation_index"] == 0 for msg in messages)


def test_analyze_dataset_analyzer_failure(conversations):
    """Test analysis when an analyzer fails."""
    # Create config with failing analyzer
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        analyzers=[
            SampleAnalyzerParams(id="failing_analyzer", config={}),
        ],
    )

    analyzer, _ = create_analyzer_with_dataset(
        conversations, config, MockFailingAnalyzerRegistry
    )
    results = analyzer.analyze_dataset()

    # Should still complete analysis even with failing analyzer
    sample_results = results["sample_level_results"]
    assert sample_results["total_messages"] == 4
    assert len(sample_results["messages"]) == 4

    # Should not have analyzer metrics due to failure
    first_message = sample_results["messages"][0]
    assert "failing_analyzer_char_count" not in first_message


def test_analyze_dataset_no_analyzers():
    """Test that DatasetAnalyzer raises an error when no analyzers are configured."""
    # Create config with no analyzers
    config = AnalyzeConfig(dataset_name="test_dataset", analyzers=[])

    analyzer, _ = create_analyzer_with_dataset([], config)

    # Should raise an error when trying to analyze without analyzers
    with pytest.raises(ValueError, match="No analyzers configured for analysis"):
        analyzer.analyze_dataset()


def test_analyze_dataset_sample_count_none(conversations, mock_config):
    """Test analysis with sample_count=None (analyze all conversations)."""
    # Create config with sample_count=None
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=None,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_dataset(conversations, config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert sample_results["total_conversations"] == 2
    assert sample_results["conversations_analyzed"] == 2
    assert sample_results["total_messages"] == 4


def test_analyze_dataset_sample_count_zero(conversations, mock_config):
    """Test analysis with sample_count=0 (analyze no conversations)."""
    # Create config with sample_count=0
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=0,
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_dataset(conversations, config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert sample_results["total_conversations"] == 2
    assert sample_results["conversations_analyzed"] == 0
    assert sample_results["total_messages"] == 0
    assert sample_results["messages"] == []


def test_analyze_dataset_sample_count_exceeds_total(conversations, mock_config):
    """Test analysis when sample_count exceeds total conversations."""
    # Create config with sample_count exceeding total
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=10,  # More than total conversations
        analyzers=mock_config.analyzers,
    )

    analyzer, _ = create_analyzer_with_dataset(conversations, config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert sample_results["total_conversations"] == 2
    assert sample_results["conversations_analyzed"] == 2  # Should not exceed total
    assert sample_results["total_messages"] == 4


def test_analyze_dataset_multimodal_content(mock_config):
    """Test analysis with multimodal content (non-string content)."""
    from tests.unit.utils.test_analysis_utils import MockConversation, MockMessage

    # Create message with non-string content
    mock_message = MockMessage("test content", "user")
    # Override content to be a dict for multimodal testing
    mock_message.content = {"text": "Hello world", "image": "image_data"}  # type: ignore

    conversation = MockConversation("conv_1", [mock_message])
    analyzer, _ = create_analyzer_with_dataset([conversation], mock_config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert len(sample_results["messages"]) == 1
    message = sample_results["messages"][0]
    assert (
        message["text_content"]
        == "flattened_{'text': 'Hello world', 'image': 'image_data'}"
    )  # Uses compute_flattened_text_content


def test_analyze_dataset_missing_conversation_id(mock_config):
    """Test analysis when conversation_id is None."""
    from tests.unit.utils.test_analysis_utils import MockConversation, MockMessage

    conversation = MockConversation(None, [MockMessage("Hello", "user")])
    analyzer, _ = create_analyzer_with_dataset([conversation], mock_config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert len(sample_results["messages"]) == 1
    message = sample_results["messages"][0]
    assert message["conversation_id"] == "conv_0"  # Should use fallback


def test_analyze_dataset_missing_message_id(mock_config):
    """Test analysis when message_id is None."""
    from tests.unit.utils.test_analysis_utils import MockConversation, MockMessage

    message = MockMessage("Hello", "user")
    message.id = None
    conversation = MockConversation("conv_1", [message])
    analyzer, _ = create_analyzer_with_dataset([conversation], mock_config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert len(sample_results["messages"]) == 1
    message_data = sample_results["messages"][0]
    assert message_data["message_id"] == "msg_0_0"  # Should use fallback


def test_analyze_dataset_empty_dataset(mock_config):
    """Test analysis with empty dataset."""
    analyzer, _ = create_analyzer_with_dataset([], mock_config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert sample_results["dataset_name"] == "test_dataset"
    assert sample_results["total_conversations"] == 0
    assert sample_results["conversations_analyzed"] == 0
    assert sample_results["total_messages"] == 0
    assert sample_results["messages"] == []


def test_analyze_dataset_empty_conversation(mock_config):
    """Test analysis with conversation containing no messages."""
    from tests.unit.utils.test_analysis_utils import MockConversation

    conversation = MockConversation("conv_1", [])
    analyzer, _ = create_analyzer_with_dataset([conversation], mock_config)
    results = analyzer.analyze_dataset()

    sample_results = results["sample_level_results"]
    assert sample_results["total_conversations"] == 1
    assert sample_results["conversations_analyzed"] == 1
    assert sample_results["total_messages"] == 0
    assert sample_results["messages"] == []


def test_analyze_dataset_analyzer_calls(multi_message_conversations, mock_config):
    """Test that analyzers are called with correct parameters."""
    analyzer, _ = create_analyzer_with_dataset(multi_message_conversations, mock_config)
    analyzer.analyze_dataset()

    # Check that analyzers were called for each message
    text_length_analyzer = analyzer.sample_analyzers["text_length_analyzer"]
    analyzer_2 = analyzer.sample_analyzers["analyzer_2"]

    assert len(text_length_analyzer.analyze_calls) == 4
    assert len(analyzer_2.analyze_calls) == 4

    # Check first call parameters
    text_content, metadata = text_length_analyzer.analyze_calls[0]
    assert text_content == "Hello, how are you?"
    assert metadata["conversation_id"] == "conv_1"
    assert metadata["conversation_index"] == 0
    assert metadata["message_index"] == 0
    assert metadata["role"] == "user"


def test_analyze_dataset_metric_prefixing(multi_message_conversations, mock_config):
    """Test that analyzer metrics are properly prefixed to avoid conflicts."""
    analyzer, _ = create_analyzer_with_dataset(multi_message_conversations, mock_config)
    results = analyzer.analyze_dataset()

    first_message = results["sample_level_results"]["messages"][0]

    # Check that metrics are prefixed with analyzer ID
    assert "text_length_analyzer_char_count" in first_message
    assert "text_length_analyzer_word_count" in first_message
    assert "text_length_analyzer_analyzer_id" in first_message
    assert "analyzer_2_char_count" in first_message
    assert "analyzer_2_word_count" in first_message
    assert "analyzer_2_analyzer_id" in first_message

    # Check that values are different (different analyzer IDs)
    assert first_message["text_length_analyzer_analyzer_id"] == "text_length_analyzer"
    assert first_message["analyzer_2_analyzer_id"] == "analyzer_2"
