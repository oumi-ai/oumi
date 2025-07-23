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

from typing import Any, Optional
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from oumi.core.configs import AnalyzeConfig
from oumi.core.datasets import BaseMapDataset
from oumi.utils.analysis_utils import load_dataset_from_config


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
    with patch("oumi.core.registry.REGISTRY") as mock_registry:
        yield mock_registry


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
        match="Dataset type .* is not supported for analysis",
    ):
        load_dataset_from_config(config)


def test_load_dataset_from_config_registry_exception(mock_registry):
    """Test error handling when registry raises an exception."""
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
    )

    mock_registry.get_dataset.side_effect = Exception("Registry error")

    with pytest.raises(Exception, match="Registry error"):
        load_dataset_from_config(config)


# Mock classes for testing (used by other test files)
class MockMessage:
    """Mock message for testing."""

    def __init__(self, content: str, role: str, message_id: Optional[str] = None):
        self.content = content
        self.role = Mock()
        self.role.value = role
        self.id = message_id

    def compute_flattened_text_content(self) -> str:
        """Mock flattened text content for multimodal messages."""
        return f"flattened_{self.content}"


class MockConversation:
    """Mock conversation for testing."""

    def __init__(self, conversation_id: Optional[str], messages: list[MockMessage]):
        self.conversation_id = conversation_id
        self.messages = messages


class MockDataset(BaseMapDataset):
    """Mock dataset for testing."""

    def __init__(self, conversations: list[MockConversation]):
        self.conversations = conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def conversation(self, idx: int) -> MockConversation:
        return self.conversations[idx]

    def transform(self, sample: pd.Series) -> dict:
        return sample.to_dict()


class MockAnalyzer:
    """Mock analyzer for testing."""

    def __init__(
        self,
        analyzer_id: str,
        should_fail: bool = False,
        custom_metrics: Optional[dict[str, Any]] = None,
    ):
        self.analyzer_id = analyzer_id
        self.should_fail = should_fail
        self.custom_metrics = custom_metrics
        self.analyze_calls = []

    def analyze_message(
        self, text_content: str, message_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock analysis that returns basic metrics."""
        self.analyze_calls.append((text_content, message_metadata))

        if self.should_fail:
            raise ValueError(f"Analyzer {self.analyzer_id} failed")

        if self.custom_metrics:
            return self.custom_metrics

        return {
            "char_count": len(text_content),
            "word_count": len(text_content.split()),
            "analyzer_id": self.analyzer_id,
        }
