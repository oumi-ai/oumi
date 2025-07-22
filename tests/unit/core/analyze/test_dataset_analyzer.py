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
    assert "length" in analyzer.sample_analyzers
    assert "mock" in analyzer.sample_analyzers


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
@patch("oumi.core.analyze.dataset_analyzer.compute_sample_level_analysis")
def test_analyze_dataset_integration(mock_compute, mock_load, mock_config):
    """Test DatasetAnalyzer analysis integration."""
    mock_load.return_value = "mock_dataset"
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

    # Test result structure
    assert results["dataset_name"] == "test_dataset"
    assert "sample_level_results" in results
    assert results["sample_level_results"]["total_messages"] == 4

    # Test that compute_sample_level_analysis was called with correct parameters
    mock_compute.assert_called_once()
    call_args = mock_compute.call_args
    assert call_args[0][0] == mock_load.return_value  # dataset
    assert call_args[0][1] == mock_config  # config
    assert isinstance(call_args[0][2], dict)  # sample_analyzers dict

    # Test that analyzers were passed correctly
    sample_analyzers = call_args[0][2]
    assert "length" in sample_analyzers
    assert "mock" in sample_analyzers
    assert len(sample_analyzers) == 2


@patch("oumi.core.analyze.dataset_analyzer.REGISTRY", MockRegistry())
@patch("oumi.core.analyze.dataset_analyzer.load_dataset_from_config")
def test_analyze_dataset_no_analyzers(mock_load):
    """Test that DatasetAnalyzer raises an error when no analyzers are configured."""
    mock_load.return_value = "mock_dataset"

    # Create config with no analyzers
    config = AnalyzeConfig(dataset_name="test_dataset", analyzers=[])

    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    analyzer = DatasetAnalyzer(config)

    # Should raise an error when trying to analyze without analyzers
    with pytest.raises(ValueError, match="No analyzers configured for analysis"):
        analyzer.analyze_dataset()
