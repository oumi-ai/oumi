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

from unittest.mock import Mock, patch

import datasets
import pytest

from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.analysis_utils import (
    build_tokenizer_from_config,
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
def mock_hf_loader():
    """Fixture to patch HuggingFace loading."""
    with patch(
        "oumi.utils.analysis_utils._load_dataset_from_huggingface_hub"
    ) as mock_loader:
        yield mock_loader


@pytest.fixture
def mock_hf_loading():
    """Fixture to patch HuggingFace loading at the module level."""
    with patch(
        "oumi.utils.analysis_utils._load_dataset_from_huggingface_hub"
    ) as mock_loader:
        # Set up a default mock that raises an error to prevent real loading
        mock_loader.side_effect = RuntimeError("Mock HuggingFace loading")
        yield mock_loader


def test_load_dataset_from_config_success(
    mock_dataset_class_and_instance, mock_registry, mock_hf_loading
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
    with pytest.raises(ValueError, match="'dataset_name' must be provided"):
        AnalyzeConfig(
            dataset_name=None,
            split="train",
        )


def test_load_dataset_from_config_dataset_not_registered(mock_registry):
    """Test error handling when dataset is not found in registry."""
    config = AnalyzeConfig(
        dataset_name="nonexistent_dataset",
        split="train",
    )

    mock_registry.get_dataset.return_value = None

    # Mock the HuggingFace dataset loading
    with patch(
        "oumi.utils.analysis_utils._load_dataset_from_huggingface_hub"
    ) as mock_hf_loader:
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_hf_loader.return_value = mock_dataset

        result = load_dataset_from_config(config)

        # Verify that HuggingFace loading was attempted
        mock_hf_loader.assert_called_once_with(config)
        assert result == mock_dataset


def test_load_dataset_from_config_huggingface_loading(mock_registry):
    """Test HuggingFace dataset loading when dataset is not in registry."""
    config = AnalyzeConfig(
        dataset_name="test_hf_dataset",
        split="train",
        trust_remote_code=True,
    )

    mock_registry.get_dataset.return_value = None

    # Mock the HuggingFace dataset loading
    with patch(
        "oumi.utils.analysis_utils._load_dataset_from_huggingface_hub"
    ) as mock_hf_loader:
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_hf_loader.return_value = mock_dataset

        result = load_dataset_from_config(config)

        # Verify that HuggingFace loading was attempted
        mock_hf_loader.assert_called_once_with(config)
        assert result == mock_dataset


def test_load_dataset_from_huggingface_hub_success():
    """Test successful HuggingFace Hub dataset loading."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        trust_remote_code=True,
    )

    # Mock the datasets library and registry
    with (
        patch("datasets.load_dataset") as mock_load_dataset,
        patch("oumi.utils.analysis_utils.REGISTRY") as mock_registry,
    ):
        # Mock the HuggingFace dataset as a proper Dataset type
        class MockDataset(datasets.Dataset):
            def __init__(self):
                pass

            def __getitem__(self, idx):
                return {"messages": [{"role": "user", "content": "test"}]}

        mock_hf_dataset = MockDataset()
        mock_load_dataset.return_value = mock_hf_dataset

        # Mock the registry to return a dataset class
        mock_dataset_class = Mock()
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_dataset_class.return_value = mock_dataset
        mock_registry.get_dataset.return_value = mock_dataset_class

        from oumi.utils.analysis_utils import _load_dataset_from_huggingface_hub

        result = _load_dataset_from_huggingface_hub(config)

        # Verify the dataset was loaded correctly
        mock_load_dataset.assert_called_once_with(
            path="test_dataset", split="train", trust_remote_code=True
        )
        assert result == mock_dataset


def test_load_dataset_from_huggingface_hub_with_subset():
    """Test HuggingFace Hub dataset loading with subset."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        subset="subset_name",
        split="train",
        trust_remote_code=True,
    )

    # Mock the datasets library and registry
    with (
        patch("datasets.load_dataset") as mock_load_dataset,
        patch("oumi.utils.analysis_utils.REGISTRY") as mock_registry,
    ):

        class MockDataset(datasets.Dataset):
            def __init__(self):
                pass

            def __getitem__(self, idx):
                return {"messages": [{"role": "user", "content": "test"}]}

        mock_hf_dataset = MockDataset()
        mock_load_dataset.return_value = mock_hf_dataset

        # Mock the registry to return a dataset class
        mock_dataset_class = Mock()
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_dataset_class.return_value = mock_dataset
        mock_registry.get_dataset.return_value = mock_dataset_class

        from oumi.utils.analysis_utils import _load_dataset_from_huggingface_hub

        result = _load_dataset_from_huggingface_hub(config)

        # Verify the dataset was loaded with subset
        mock_load_dataset.assert_called_once_with(
            path="test_dataset",
            name="subset_name",
            split="train",
            trust_remote_code=True,
        )
        assert result == mock_dataset


def test_load_dataset_from_config_for_non_basemapdataset(mock_registry, mock_hf_loader):
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


def test_load_dataset_from_config_registry_exception(mock_registry, mock_hf_loader):
    """Test error handling when registry.get_dataset raises an exception."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_registry.get_dataset.side_effect = Exception("Registry error")

    with pytest.raises(Exception, match="Registry error"):
        load_dataset_from_config(config)


def test_load_dataset_from_config_with_processor_parameters(
    mock_dataset_class_and_instance, mock_registry, mock_hf_loading
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


def test_load_dataset_from_huggingface_hub_import_error():
    """Test error handling when datasets library is not available."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    # Mock import error
    with patch(
        "datasets.load_dataset", side_effect=ImportError("No module named 'datasets'")
    ):
        from oumi.utils.analysis_utils import _load_dataset_from_huggingface_hub

        with pytest.raises(ImportError, match="The 'datasets' library is required"):
            _load_dataset_from_huggingface_hub(config)


def test_detect_dataset_field_mapping():
    """Test the dataset field mapping detection function."""
    from oumi.utils.analysis_utils import _detect_dataset_field_mapping

    # Test with standard vision-language fields
    sample = {
        "image": "path/to/image.jpg",
        "question": "What is in this image?",
        "answer": "A cat",
    }

    result = _detect_dataset_field_mapping(sample)
    assert result["image_column"] == "image"
    assert result["question_column"] == "question"
    assert result["answer_column"] == "answer"

    # Test with alternative field names
    sample2 = {
        "img": "path/to/image.jpg",
        "prompt": "Describe this image",
        "output": "A beautiful landscape",
    }

    result2 = _detect_dataset_field_mapping(sample2)
    assert result2["image_column"] == "img"
    assert result2["question_column"] == "prompt"
    assert result2["answer_column"] == "output"


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


def test_load_dataset_from_config_huggingface_loading(mock_registry):
    """Test HuggingFace dataset loading when dataset is not in registry."""
    config = AnalyzeConfig(
        dataset_name="test_hf_dataset",
        split="train",
        trust_remote_code=True,
    )

    mock_registry.get_dataset.return_value = None

    # Mock the HuggingFace dataset loading
    with patch(
        "oumi.utils.analysis_utils._load_dataset_from_huggingface_hub"
    ) as mock_hf_loader:
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_hf_loader.return_value = mock_dataset

        result = load_dataset_from_config(config)

        # Verify that HuggingFace loading was attempted
        mock_hf_loader.assert_called_once_with(config)
        assert result == mock_dataset


def test_load_dataset_from_huggingface_hub_success():
    """Test successful HuggingFace Hub dataset loading."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        trust_remote_code=True,
    )

    # Mock the datasets library and registry
    with (
        patch("datasets.load_dataset") as mock_load_dataset,
        patch("oumi.utils.analysis_utils.REGISTRY") as mock_registry,
    ):
        # Mock the HuggingFace dataset as a proper Dataset type
        class MockDataset(datasets.Dataset):
            def __init__(self):
                pass

            def __getitem__(self, idx):
                return {"messages": [{"role": "user", "content": "test"}]}

        mock_hf_dataset = MockDataset()
        mock_load_dataset.return_value = mock_hf_dataset

        # Mock the registry to return a dataset class
        mock_dataset_class = Mock()
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_dataset_class.return_value = mock_dataset
        mock_registry.get_dataset.return_value = mock_dataset_class

        from oumi.utils.analysis_utils import _load_dataset_from_huggingface_hub

        result = _load_dataset_from_huggingface_hub(config)

        # Verify the dataset was loaded correctly
        mock_load_dataset.assert_called_once_with(
            path="test_dataset", split="train", trust_remote_code=True
        )
        assert result == mock_dataset


def test_load_dataset_from_huggingface_hub_with_subset():
    """Test HuggingFace Hub dataset loading with subset."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        subset="subset_name",
        split="train",
        trust_remote_code=True,
    )

    # Mock the datasets library and registry
    with (
        patch("datasets.load_dataset") as mock_load_dataset,
        patch("oumi.utils.analysis_utils.REGISTRY") as mock_registry,
    ):

        class MockDataset(datasets.Dataset):
            def __init__(self):
                pass

            def __getitem__(self, idx):
                return {"messages": [{"role": "user", "content": "test"}]}

        mock_hf_dataset = MockDataset()
        mock_load_dataset.return_value = mock_hf_dataset

        # Mock the registry to return a dataset class
        mock_dataset_class = Mock()
        mock_dataset = Mock(spec=BaseMapDataset)
        mock_dataset_class.return_value = mock_dataset
        mock_registry.get_dataset.return_value = mock_dataset_class

        from oumi.utils.analysis_utils import _load_dataset_from_huggingface_hub

        result = _load_dataset_from_huggingface_hub(config)

        # Verify the dataset was loaded with subset
        mock_load_dataset.assert_called_once_with(
            path="test_dataset",
            name="subset_name",
            split="train",
            trust_remote_code=True,
        )
        assert result == mock_dataset


def test_load_dataset_from_huggingface_hub_import_error():
    """Test error handling when datasets library is not available."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    # Mock import error
    with patch(
        "datasets.load_dataset", side_effect=ImportError("No module named 'datasets'")
    ):
        from oumi.utils.analysis_utils import _load_dataset_from_huggingface_hub

        with pytest.raises(ImportError, match="The 'datasets' library is required"):
            _load_dataset_from_huggingface_hub(config)


def test_detect_dataset_field_mapping():
    """Test the dataset field mapping detection function."""
    from oumi.utils.analysis_utils import _detect_dataset_field_mapping

    # Test with standard vision-language fields
    sample = {
        "image": "path/to/image.jpg",
        "question": "What is in this image?",
        "answer": "A cat",
    }

    result = _detect_dataset_field_mapping(sample)
    assert result["image_column"] == "image"
    assert result["question_column"] == "question"
    assert result["answer_column"] == "answer"

    # Test with alternative field names
    sample2 = {
        "img": "path/to/image.jpg",
        "prompt": "Describe this image",
        "output": "A beautiful landscape",
    }

    result2 = _detect_dataset_field_mapping(sample2)
    assert result2["image_column"] == "img"
    assert result2["question_column"] == "prompt"
    assert result2["answer_column"] == "output"
