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

import pytest

from oumi.core.configs import DatasetAnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.analysis_utils import load_dataset_from_config


class TestLoadDatasetFromConfig:
    """Test the load_dataset_from_config function."""

    def _mock_dataset_class_and_instance(self):
        mock_dataset_class = Mock()
        mock_dataset_instance = Mock(spec=BaseMapDataset)
        mock_dataset_class.return_value = mock_dataset_instance
        return mock_dataset_class, mock_dataset_instance

    def test_load_dataset_from_config_success(self):
        """
        Test successful dataset loading.
        """
        config = DatasetAnalyzeConfig(
            dataset_name="test_dataset",
            split="train",
        )

        mock_dataset_class, mock_dataset_instance = (
            self._mock_dataset_class_and_instance()
        )

        with patch("oumi.core.registry.REGISTRY") as mock_registry:
            mock_registry.get_dataset.return_value = mock_dataset_class

            result = load_dataset_from_config(config)

            assert result == mock_dataset_instance
            assert mock_registry.get_dataset.called

    def test_load_dataset_from_config_missing_dataset_name(self):
        """Test error handling when dataset_name is not provided."""
        with pytest.raises(ValueError, match="'dataset_name' must be provided"):
            DatasetAnalyzeConfig(
                dataset_name=None,
                split="train",
            )

    def test_load_dataset_from_config_dataset_not_registered(self):
        """
        Test error handling when dataset is not found in registry.
        """
        config = DatasetAnalyzeConfig(
            dataset_name="nonexistent_dataset",
            split="train",
        )

        with patch("oumi.core.registry.REGISTRY") as mock_registry:
            mock_registry.get_dataset.return_value = None

            with pytest.raises(
                NotImplementedError,
                match=(
                    "Dataset 'nonexistent_dataset' is not registered in the REGISTRY. "
                    "Loading from HuggingFace Hub is not yet implemented."
                ),
            ):
                load_dataset_from_config(config)

    def test_load_dataset_from_config_for_non_basemapdataset(self):
        """
        Test error handling when dataset class doesn't inherit from
        BaseMapDataset.
        """
        config = DatasetAnalyzeConfig(
            dataset_name="test_dataset",
            split="train",
        )

        mock_dataset_class = Mock()
        mock_dataset_instance = Mock()  # Not a BaseMapDataset
        mock_dataset_class.return_value = mock_dataset_instance

        with patch("oumi.core.registry.REGISTRY") as mock_registry:
            mock_registry.get_dataset.return_value = mock_dataset_class

            with pytest.raises(
                NotImplementedError,
                match=(
                    "Dataset type .* is not supported for analysis. "
                    "Please use a dataset that inherits from BaseMapDataset."
                ),
            ):
                load_dataset_from_config(config)

    def test_load_dataset_from_config_registry_exception(self):
        """Test error handling when registry.get_dataset raises an exception."""
        config = DatasetAnalyzeConfig(
            dataset_name="test_dataset",
            split="train",
        )

        with patch("oumi.core.registry.REGISTRY") as mock_registry:
            mock_registry.get_dataset.side_effect = Exception("Registry error")

            with pytest.raises(Exception, match="Registry error"):
                load_dataset_from_config(config)
