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

from oumi.core.configs.analyze_config import (
    AnalyzeConfig,
    DatasetSource,
)
from oumi.core.datasets import BaseMapDataset
from oumi.datasets import TextSftJsonLinesDataset, VLJsonlinesDataset
from oumi.utils.analysis_utils import (
    build_tokenizer_from_config,
    compute_statistics,
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


def test_load_dataset_from_config_missing_dataset_name():
    """Test error handling when dataset_name is not provided."""
    with pytest.raises(
        ValueError,
        match="Either 'dataset_name' or 'dataset_path' must be provided when "
        "dataset_source=DatasetSource.CONFIG",
    ):
        AnalyzeConfig(
            dataset_source=DatasetSource.CONFIG,  # Required field
            dataset_name=None,
            dataset_path=None,
            split="train",
        )


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


def test_build_tokenizer_from_config_success():
    """Test successful tokenizer building from config."""
    tokenizer_config = {
        "model_name": "gpt2",
        "tokenizer_kwargs": {"padding_side": "left"},
        "trust_remote_code": False,
    }

    tokenizer = build_tokenizer_from_config(tokenizer_config)

    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")


def test_build_tokenizer_from_config_none():
    """Test tokenizer building with None config."""
    tokenizer = build_tokenizer_from_config(None)

    assert tokenizer is None


def test_build_tokenizer_from_config_missing_model_name():
    """Test error handling when model_name is missing from config."""
    tokenizer_config = {
        "tokenizer_kwargs": {"padding_side": "left"},
        "trust_remote_code": False,
    }

    with pytest.raises(
        ValueError, match="tokenizer_config must contain 'model_name' field"
    ):
        build_tokenizer_from_config(tokenizer_config)


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
    """Test loading custom dataset in conversation format."""
    config = AnalyzeConfig(
        dataset_path=temp_conversation_file,
        dataset_format="oumi",
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
    """Test loading custom dataset in alpaca format."""
    config = AnalyzeConfig(
        dataset_path=temp_alpaca_file,
        dataset_format="alpaca",
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

    config = AnalyzeConfig(
        dataset_path=temp_vision_language_file,
        dataset_format="oumi",
        processor_name="openai/clip-vit-base-patch32",  # Processor provided
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
        dataset_format="oumi",
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
        dataset_format="oumi",  # Required for custom datasets
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
            dataset_format="oumi",  # Required for custom datasets
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
