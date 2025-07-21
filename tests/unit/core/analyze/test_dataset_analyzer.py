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


class MockDataset:
    """Mock dataset for testing."""

    def __init__(self, conversations=None):
        self.conversations = conversations or [
            {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you!"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4."},
                ]
            },
        ]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]


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
                id="length", config={"char_count": True, "word_count": True}
            ),
            SampleAnalyzerParams(id="mock", config={"analyzer_id": "mock"}),
        ],
    )


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    return MockDataset()


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
@patch("oumi.core.analyze.dataset_analyzer.compute_sample_level_analysis")
def test_analyzer_initialization(mock_compute, mock_load, mock_config):
    """Test DatasetAnalyzer initialization with mocked dependencies."""
    mock_load.return_value = MockDataset()
    mock_compute.return_value = {
        "total_conversations": 2,
        "conversations_analyzed": 2,
        "total_messages": 4,
        "messages": [],
    }

    # Import here to avoid import issues
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    analyzer = DatasetAnalyzer(mock_config)

    assert analyzer.config == mock_config
    assert analyzer.dataset_name == "test_dataset"
    assert analyzer.split == "train"


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
@patch("oumi.core.analyze.dataset_analyzer.compute_sample_level_analysis")
def test_analyze_dataset(mock_compute, mock_load, mock_config):
    """Test dataset analysis functionality."""
    mock_load.return_value = MockDataset()
    mock_compute.return_value = {
        "dataset_name": "test_dataset",
        "total_conversations": 2,
        "conversations_analyzed": 2,
        "total_messages": 4,
        "messages": [
            {"role": "user", "content": "Hello", "char_count": 5},
            {"role": "assistant", "content": "Hi", "char_count": 2},
        ],
    }

    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    analyzer = DatasetAnalyzer(mock_config)
    results = analyzer.analyze_dataset()

    assert results["dataset_name"] == "test_dataset"
    assert "sample_level_results" in results
    assert results["sample_level_results"]["total_messages"] == 4


def test_analyzer_config_validation():
    """Test analyzer configuration validation."""
    # Test with invalid config (missing required fields)
    with pytest.raises(ValueError):
        AnalyzeConfig(
            dataset_name="",  # Empty dataset name should raise error
            analyzers=[],
        )


def test_sample_analyzer_config_validation():
    """Test sample analyzer configuration validation."""
    # Test with valid config
    config = SampleAnalyzerParams(id="test_analyzer", config={"param1": "value1"})
    assert config.id == "test_analyzer"
    assert config.config["param1"] == "value1"


def test_dataset_analyzer_config_defaults():
    """Test AnalyzeConfig default values."""
    config = AnalyzeConfig(dataset_name="test_dataset", analyzers=[])

    assert config.dataset_name == "test_dataset"
    assert config.split == "train"  # default value
    assert config.sample_count is None  # default value
    assert config.output_path == "."  # default value
