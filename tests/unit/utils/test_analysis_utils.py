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

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import jsonlines
import pandas as pd
import pytest

from oumi.core.configs.analyze_config import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.datasets import TextSftJsonLinesDataset, VLJsonlinesDataset
from oumi.utils.analysis_utils import (
    DistributionAnalysisResult,
    DistributionType,
    ModeStatistics,
    compute_multimodal_outliers,
    compute_statistics,
    compute_statistics_with_distribution,
    detect_distribution_type,
    load_dataset_from_config,
)


@pytest.fixture
def mock_dataset_class_and_instance():
    """Fixture to create mock dataset class and instance."""
    mock_dataset_class = Mock()
    mock_dataset_instance = Mock(spec=BaseMapDataset)
    mock_dataset_class.return_value = mock_dataset_instance
    return mock_dataset_class, mock_dataset_instance


@pytest.fixture
def mock_registry():
    """Fixture to patch the registry."""
    with patch("oumi.utils.analysis_utils.REGISTRY") as mock_registry:
        yield mock_registry


@pytest.fixture
def sample_conversation_data():
    """Sample conversation format data."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What's the weather like?"},
                {
                    "role": "assistant",
                    "content": "I don't have access to real-time weather data.",
                },
            ]
        },
    ]


@pytest.fixture
def sample_alpaca_data():
    """Sample alpaca format data."""
    return [
        {
            "instruction": "What's the weather like in Seattle today?",
            "input": "",
            "output": "I apologize, but I don't have access to real-time weather "
            "information for Seattle.",
        },
        {
            "instruction": "Compute the average of the presented numbers.",
            "input": "5, 6, 10",
            "output": "The average for the numbers: 5, 6, 10 can be computed by "
            "adding first all of them, and then dividing this sum by their total "
            "number. First, 5+6+10 = 21. Then, 21 / 3 = 7. The average is 7.",
        },
    ]


@pytest.fixture
def sample_vision_language_data():
    """Sample vision-language data."""
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "content": "https://example.com/image_of_dog.jpg",
                        },
                        {"type": "text", "content": "What breed is this dog?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "This appears to be a Shih Tzu puppy.",
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "content": "https://example.com/scenic_view.jpg",
                        },
                        {"type": "text", "content": "Describe this image:"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "A scenic view of the puget sound with mountains in "
                    "the background.",
                },
            ]
        },
    ]


@pytest.fixture
def temp_conversation_file(sample_conversation_data):
    """Create a temporary conversation format file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_conversation_data, f)
        f.flush()  # Ensure data is written to disk
        yield f.name
    Path(f.name).unlink()  # Cleanup


@pytest.fixture
def temp_alpaca_file(sample_alpaca_data):
    """Create a temporary alpaca format file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_alpaca_data, f)
        f.flush()  # Ensure data is written to disk
        yield f.name
    Path(f.name).unlink()  # Cleanup


@pytest.fixture
def temp_vision_language_file(sample_vision_language_data):
    """Create a temporary vision-language format file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        with jsonlines.open(f.name, mode="w") as writer:
            writer.write_all(sample_vision_language_data)
        yield f.name
    Path(f.name).unlink()  # Cleanup


def test_load_dataset_from_config_success(
    mock_dataset_class_and_instance, mock_registry
):
    """Test successful dataset loading."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_dataset_class, mock_dataset_instance = mock_dataset_class_and_instance
    mock_registry.get_dataset.return_value = mock_dataset_class

    result = load_dataset_from_config(config)

    assert result == mock_dataset_instance
    assert mock_registry.get_dataset.called


def test_load_dataset_from_config_missing_dataset_info():
    """Test error handling when no dataset info is provided."""
    config = AnalyzeConfig(split="train")

    with pytest.raises(
        ValueError, match="Either dataset_name or dataset_path must be provided"
    ):
        load_dataset_from_config(config)


def test_load_dataset_from_config_dataset_not_registered(mock_registry):
    """Test error handling when dataset is not found in registry."""
    config = AnalyzeConfig(
        dataset_name="nonexistent_dataset",
        split="train",
    )

    mock_registry.get_dataset.return_value = None

    with pytest.raises(
        NotImplementedError,
        match=(
            "Dataset 'nonexistent_dataset' is not registered in the REGISTRY. "
            "Loading from HuggingFace Hub is not yet implemented."
        ),
    ):
        load_dataset_from_config(config)


def test_load_dataset_from_config_for_non_basemapdataset(mock_registry):
    """Test error handling when dataset class doesn't inherit from BaseMapDataset."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_dataset_class = Mock()
    mock_dataset_instance = Mock()  # Not a BaseMapDataset
    mock_dataset_class.return_value = mock_dataset_instance

    mock_registry.get_dataset.return_value = mock_dataset_class

    with pytest.raises(
        NotImplementedError,
        match=(
            "Dataset type .* is not supported for analysis. "
            "Please use a dataset that inherits from BaseMapDataset."
        ),
    ):
        load_dataset_from_config(config)


def test_load_dataset_from_config_registry_exception(mock_registry):
    """Test error handling when registry.get_dataset raises an exception."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_registry.get_dataset.side_effect = Exception("Registry error")

    with pytest.raises(Exception, match="Registry error"):
        load_dataset_from_config(config)


def test_load_dataset_from_config_with_processor_parameters(
    mock_dataset_class_and_instance, mock_registry
):
    """Test dataset loading with processor parameters."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        processor_name="Salesforce/blip2-opt-2.7b",
        processor_kwargs={"image_size": 224},
        trust_remote_code=True,
    )

    mock_dataset_class, mock_dataset_instance = mock_dataset_class_and_instance
    mock_registry.get_dataset.return_value = mock_dataset_class

    result = load_dataset_from_config(config)

    # Verify the dataset was called with processor parameters
    mock_dataset_class.assert_called_once()
    call_kwargs = mock_dataset_class.call_args[1]
    assert call_kwargs["processor_name"] == "Salesforce/blip2-opt-2.7b"
    assert call_kwargs["processor_kwargs"] == {"image_size": 224}
    assert call_kwargs["trust_remote_code"] is True
    assert result == mock_dataset_instance


def test_load_dataset_from_config_with_tokenizer(
    mock_dataset_class_and_instance, mock_registry
):
    """Test dataset loading with provided tokenizer."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_tokenizer = Mock()
    mock_dataset_class, mock_dataset_instance = mock_dataset_class_and_instance
    mock_registry.get_dataset.return_value = mock_dataset_class

    result = load_dataset_from_config(config, tokenizer=mock_tokenizer)

    # Verify the dataset was called with tokenizer
    mock_dataset_class.assert_called_once()
    call_kwargs = mock_dataset_class.call_args[1]
    assert call_kwargs["tokenizer"] == mock_tokenizer
    assert result == mock_dataset_instance


def test_load_dataset_from_config_without_tokenizer(
    mock_dataset_class_and_instance, mock_registry
):
    """Test dataset loading without tokenizer."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_dataset_class, mock_dataset_instance = mock_dataset_class_and_instance
    mock_registry.get_dataset.return_value = mock_dataset_class

    result = load_dataset_from_config(config)

    # Verify the dataset was called without tokenizer
    mock_dataset_class.assert_called_once()
    call_kwargs = mock_dataset_class.call_args[1]
    assert "tokenizer" not in call_kwargs
    assert result == mock_dataset_instance


# Custom dataset loading tests
def test_load_custom_dataset_conversation_format(temp_conversation_file):
    """Test loading custom dataset in conversation format (auto-detected)."""
    config = AnalyzeConfig(
        dataset_path=temp_conversation_file,
        is_multimodal=False,  # Explicitly set as text-only
    )

    dataset = load_dataset_from_config(config)

    assert isinstance(dataset, TextSftJsonLinesDataset)
    assert len(dataset) == 2

    # Check first conversation
    conv1 = dataset.conversation(0)
    assert len(conv1.messages) == 2
    assert conv1.messages[0].role == "user"
    assert conv1.messages[0].content == "Hello, how are you?"
    assert conv1.messages[1].role == "assistant"
    assert conv1.messages[1].content == "I'm doing well, thank you!"


def test_load_custom_dataset_alpaca_format(temp_alpaca_file):
    """Test loading custom dataset in alpaca format (auto-detected)."""
    config = AnalyzeConfig(
        dataset_path=temp_alpaca_file,
        is_multimodal=False,  # Explicitly set as text-only
    )

    dataset = load_dataset_from_config(config)

    assert isinstance(dataset, TextSftJsonLinesDataset)
    assert len(dataset) == 2

    # Check conversation structure
    conv1 = dataset.conversation(0)
    assert len(conv1.messages) == 2
    assert conv1.messages[0].role == "user"
    assert conv1.messages[1].role == "assistant"


def test_load_custom_dataset_multi_modal(temp_vision_language_file):
    """Test loading custom multimodal dataset with processor."""
    # Create a mock tokenizer with required attributes
    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0  # Set a valid pad_token_id
    mock_tokenizer.convert_tokens_to_ids.return_value = 0

    config = AnalyzeConfig(
        dataset_path=temp_vision_language_file,
        processor_name="HuggingFaceTB/SmolVLM-256M-Instruct",  # Processor provided
        is_multimodal=True,  # Explicitly mark as multimodal
    )

    dataset = load_dataset_from_config(config, tokenizer=mock_tokenizer)
    assert isinstance(dataset, VLJsonlinesDataset)
    assert len(dataset) == 2


def test_load_custom_dataset_text(temp_conversation_file):
    """Test that text-only datasets are correctly detected and loaded as
    TextSftJsonLinesDataset."""
    config = AnalyzeConfig(
        dataset_path=temp_conversation_file,
        is_multimodal=False,  # Explicitly set as text-only
    )

    dataset = load_dataset_from_config(config)

    # Should be detected as text-only and loaded as TextSftJsonLinesDataset
    assert isinstance(dataset, TextSftJsonLinesDataset)
    assert len(dataset) == 2


def test_load_custom_dataset_file_not_found():
    """Test error handling when custom dataset file doesn't exist."""
    config = AnalyzeConfig(
        dataset_path="nonexistent_file.json",
        is_multimodal=False,  # Required for custom datasets
    )

    with pytest.raises(
        FileNotFoundError, match="Dataset file not found: nonexistent_file.json"
    ):
        load_dataset_from_config(config)


def test_load_custom_dataset_directory_path():
    """Test error handling when custom dataset path is a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AnalyzeConfig(
            dataset_path=temp_dir,
            is_multimodal=False,  # Required for custom datasets
        )

        with pytest.raises(
            ValueError, match="Dataset path must be a file, not a directory"
        ):
            load_dataset_from_config(config)


def test_compute_statistics_empty_series():
    """Test compute_statistics with empty pandas Series."""
    empty_series = pd.Series([], dtype=float)
    result = compute_statistics(empty_series)

    expected = {
        "count": 0,
        "mean": 0.0,
        "std": 0.0,
        "min": 0,
        "max": 0,
        "median": 0.0,
    }
    assert result == expected


def test_compute_statistics_single_value():
    """Test compute_statistics with single value (edge case for NaN std)."""
    single_series = pd.Series([42.5])
    result = compute_statistics(single_series)

    expected = {
        "count": 1,
        "mean": 42.5,
        "std": 0.0,  # Standard deviation is 0 for single value
        "min": 42.5,
        "max": 42.5,
        "median": 42.5,
    }
    assert result == expected


def test_compute_statistics_multiple_values():
    """Test compute_statistics with multiple values."""
    series = pd.Series([1, 2, 3, 4, 5])
    result = compute_statistics(series)

    expected = {
        "count": 5,
        "mean": 3.0,
        "std": 1.58,
        "min": 1,
        "max": 5,
        "median": 3.0,
    }
    assert result == expected


def test_compute_statistics_multiple_values_with_precision():
    """Test compute_statistics with multiple values and custom decimal precision."""
    series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    result = compute_statistics(series, decimal_precision=1)

    expected = {
        "count": 5,
        "mean": 3.3,
        "std": 1.7,
        "min": 1.1,
        "max": 5.5,
        "median": 3.3,
    }
    assert result == expected


def test_compute_statistics_boolean_series():
    """Test compute_statistics handles boolean series correctly."""
    series = pd.Series([True, False, True, True, False])
    result = compute_statistics(series)

    # Boolean series should be converted to int (True=1, False=0)
    expected = {
        "count": 5,
        "mean": 0.6,  # 3/5 = 0.6
        "std": 0.55,  # std of [1, 0, 1, 1, 0]
        "min": 0.0,
        "max": 1.0,
        "median": 1.0,
    }
    assert result == expected


# =============================================================================
# Distribution Analysis Tests
# =============================================================================


def _sklearn_available():
    """Check if sklearn is available for distribution tests."""
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def unimodal_data():
    """Create unimodal (normal) distribution data."""
    import numpy as np

    np.random.seed(42)
    return pd.Series(np.random.normal(100, 10, 200))


@pytest.fixture
def bimodal_data():
    """Create bimodal distribution data."""
    import numpy as np

    np.random.seed(42)
    # Two distinct modes: ~50 and ~500
    short = np.random.normal(50, 5, 100)
    long = np.random.normal(500, 30, 100)
    return pd.Series(np.concatenate([short, long]))


@pytest.fixture
def trimodal_data():
    """Create trimodal distribution data."""
    import numpy as np

    np.random.seed(42)
    # Three distinct modes
    mode1 = np.random.normal(20, 3, 80)
    mode2 = np.random.normal(100, 10, 80)
    mode3 = np.random.normal(500, 40, 80)
    return pd.Series(np.concatenate([mode1, mode2, mode3]))


@pytest.mark.skipif(not _sklearn_available(), reason="sklearn not installed")
class TestDistributionDetection:
    """Tests for distribution type detection using GMM."""

    def test_detect_unimodal_normal(self, unimodal_data):
        """Test that a normal distribution is detected as unimodal."""
        result = detect_distribution_type(unimodal_data)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.num_modes == 1
        assert len(result.mode_statistics) == 1

        # Mode statistics should match the distribution parameters (~100, ~10)
        mode = result.mode_statistics[0]
        assert 90 < mode.mean < 110  # Close to 100
        assert 5 < mode.std < 15  # Close to 10
        assert mode.weight == pytest.approx(1.0, abs=0.01)

    def test_detect_bimodal_distribution(self, bimodal_data):
        """Test that two modes are correctly identified."""
        result = detect_distribution_type(bimodal_data)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.distribution_type == DistributionType.BIMODAL
        assert result.num_modes == 2
        assert len(result.mode_statistics) == 2

        # Check mode means are close to expected values (50 and 500)
        means = sorted([ms.mean for ms in result.mode_statistics])
        assert 40 < means[0] < 60  # First mode around 50
        assert 450 < means[1] < 550  # Second mode around 500

        # Weights should be roughly equal (~0.5 each)
        weights = [ms.weight for ms in result.mode_statistics]
        for w in weights:
            assert 0.4 < w < 0.6

        # High confidence for well-separated modes
        assert result.confidence > 0.8

    def test_detect_trimodal_distribution(self, trimodal_data):
        """Test that three modes are correctly identified."""
        result = detect_distribution_type(trimodal_data)

        assert isinstance(result, DistributionAnalysisResult)
        assert result.distribution_type == DistributionType.MULTIMODAL
        assert result.num_modes >= 3  # May detect 3 or more modes
        assert len(result.mode_statistics) >= 3

    def test_insufficient_data(self):
        """Test graceful handling of small datasets."""
        # Empty series
        empty = pd.Series([], dtype=float)
        result = detect_distribution_type(empty)
        assert result.distribution_type == DistributionType.INSUFFICIENT_DATA
        assert result.num_modes == 0

        # Single value
        single = pd.Series([42.0])
        result = detect_distribution_type(single)
        assert result.distribution_type == DistributionType.INSUFFICIENT_DATA

        # Small dataset (below min_samples threshold)
        small = pd.Series([1, 2, 3, 4, 5])
        result = detect_distribution_type(small, min_samples=50)
        # Should fallback to unimodal
        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.num_modes == 1

    def test_low_cv_skips_gmm(self):
        """Test that low coefficient of variation skips GMM analysis."""
        import numpy as np

        # Very tight distribution (low CV)
        np.random.seed(42)
        tight = pd.Series(np.random.normal(100, 1, 200))  # CV = 0.01

        result = detect_distribution_type(tight, cv_threshold=0.5)

        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.num_modes == 1
        # Should have high confidence (bypassed GMM)
        assert result.confidence == 1.0


@pytest.mark.skipif(not _sklearn_available(), reason="sklearn not installed")
class TestMultimodalOutlierDetection:
    """Tests for outlier detection in multimodal distributions."""

    def test_unimodal_outlier_detection(self, unimodal_data):
        """Test standard outlier detection for unimodal distributions."""
        dist_result = detect_distribution_type(unimodal_data)

        # Add some outliers
        import numpy as np

        data_with_outliers = pd.Series(
            np.append(unimodal_data.values, [200, 0, 250])  # Clear outliers
        )
        dist_result = detect_distribution_type(data_with_outliers)

        outlier_mask, details = compute_multimodal_outliers(
            data_with_outliers, dist_result, std_threshold=3.0
        )

        assert details["distribution_type"] == "unimodal"
        assert details["num_outliers"] >= 2  # At least 200 and 250 should be flagged
        assert details["num_outliers"] < 10  # But not too many

    def test_no_false_positives_between_modes(self, bimodal_data):
        """Test that samples near mode means are NOT flagged as outliers."""
        dist_result = detect_distribution_type(bimodal_data)

        outlier_mask, details = compute_multimodal_outliers(
            bimodal_data, dist_result, std_threshold=3.0
        )

        # In a clean bimodal distribution, there should be very few outliers
        # (only true outliers within each mode, not samples between modes)
        assert details["num_outliers"] < len(bimodal_data) * 0.02  # Less than 2%

        # Specifically check that values at mode centers aren't outliers
        import numpy as np

        center_mask = (
            ((bimodal_data > 45) & (bimodal_data < 55)) |  # Near mode 1
            ((bimodal_data > 470) & (bimodal_data < 530))  # Near mode 2
        )
        # None of these center values should be outliers
        assert not np.any(outlier_mask[center_mask.values])

    def test_bimodal_outlier_detection(self, bimodal_data):
        """Test outlier detection within each mode of bimodal distribution."""
        import numpy as np

        # Add clear outliers to each mode
        outliers = np.array([10, 600])  # Far from both modes
        data_with_outliers = pd.Series(
            np.append(bimodal_data.values, outliers)
        )

        dist_result = detect_distribution_type(data_with_outliers)
        outlier_mask, details = compute_multimodal_outliers(
            data_with_outliers, dist_result, std_threshold=3.0
        )

        # The added outliers should be detected
        # The outlier at 10 should be flagged (far from mode ~50)
        # The outlier at 600 should be flagged (far from mode ~500)
        assert details["num_outliers"] >= 2
        assert details["distribution_type"] in ("bimodal", "multimodal")


@pytest.mark.skipif(not _sklearn_available(), reason="sklearn not installed")
class TestDistributionStatistics:
    """Tests for compute_statistics_with_distribution function."""

    def test_unimodal_statistics(self, unimodal_data):
        """Test statistics with distribution info for unimodal data."""
        result = compute_statistics_with_distribution(unimodal_data)

        # Should have all base statistics
        assert "count" in result
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert "median" in result

        # Should have distribution info
        assert result["distribution_type"] == "unimodal"
        assert result["num_modes"] == 1
        assert "mode_statistics" in result
        assert len(result["mode_statistics"]) == 1

    def test_bimodal_statistics(self, bimodal_data):
        """Test statistics with distribution info for bimodal data."""
        result = compute_statistics_with_distribution(bimodal_data)

        # Should have all base statistics
        assert "count" in result
        assert "mean" in result

        # Should have distribution info
        assert result["distribution_type"] == "bimodal"
        assert result["num_modes"] == 2
        assert "mode_statistics" in result
        assert len(result["mode_statistics"]) == 2

        # Each mode should have all required fields
        for mode in result["mode_statistics"]:
            assert "mode_id" in mode
            assert "mean" in mode
            assert "std" in mode
            assert "count" in mode
            assert "weight" in mode

    def test_empty_series_statistics(self):
        """Test statistics with distribution info for empty series."""
        empty = pd.Series([], dtype=float)
        result = compute_statistics_with_distribution(empty)

        # Should have base statistics
        assert result["count"] == 0
        assert result["mean"] == 0.0

        # Should NOT have distribution info (too small)
        assert "distribution_type" not in result


class TestModeStatisticsDataclass:
    """Tests for ModeStatistics dataclass."""

    def test_mode_statistics_creation(self):
        """Test creating ModeStatistics instance."""
        mode = ModeStatistics(
            mode_id=0,
            mean=100.0,
            std=10.0,
            count=150,
            weight=0.75,
        )

        assert mode.mode_id == 0
        assert mode.mean == 100.0
        assert mode.std == 10.0
        assert mode.count == 150
        assert mode.weight == 0.75


class TestDistributionAnalysisResultDataclass:
    """Tests for DistributionAnalysisResult dataclass."""

    def test_distribution_result_creation(self):
        """Test creating DistributionAnalysisResult instance."""
        mode_stats = [
            ModeStatistics(mode_id=0, mean=50.0, std=5.0, count=100, weight=0.5),
            ModeStatistics(mode_id=1, mean=500.0, std=30.0, count=100, weight=0.5),
        ]

        result = DistributionAnalysisResult(
            distribution_type=DistributionType.BIMODAL,
            num_modes=2,
            global_statistics={"count": 200, "mean": 275.0, "std": 225.0},
            mode_statistics=mode_stats,
            mode_assignments=[0] * 100 + [1] * 100,
            confidence=0.95,
        )

        assert result.distribution_type == DistributionType.BIMODAL
        assert result.num_modes == 2
        assert len(result.mode_statistics) == 2
        assert len(result.mode_assignments) == 200
        assert result.confidence == 0.95
