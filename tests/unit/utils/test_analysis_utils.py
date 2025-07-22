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
from oumi.utils.analysis_utils import (
    compute_sample_level_analysis,
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


# Tests for compute_sample_level_analysis function


class MockMessage:
    """Mock message for testing."""

    def __init__(self, content: str, role: str, message_id: Optional[str] = None):
        self.content = content
        self.role = Mock(value=role)
        self.id = message_id

    def compute_flattened_text_content(self) -> str:
        """Mock method for multimodal content."""
        return f"flattened_{self.content}"


class MockConversation:
    """Mock conversation for testing."""

    def __init__(self, conversation_id: Optional[str], messages: list[MockMessage]):
        self.conversation_id = conversation_id
        self.messages = messages


class MockDataset(BaseMapDataset):
    """Mock dataset for testing."""

    def __init__(self, conversations: list[MockConversation]):
        self._conversations = conversations
        # Call parent constructor with required parameters
        super().__init__(dataset_name="test_dataset")

    def __len__(self) -> int:
        return len(self._conversations)

    def conversation(self, idx: int) -> MockConversation:
        return self._conversations[idx]

    def transform(self, sample: pd.Series) -> dict:
        """Required implementation of abstract method."""
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
        self.analyze_calls = []
        self.custom_metrics = custom_metrics or {}

    def analyze_message(
        self, text_content: str, message_metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock analysis method."""
        self.analyze_calls.append((text_content, message_metadata))

        if self.should_fail:
            raise ValueError(f"Mock analyzer {self.analyzer_id} failed")

        # Return custom metrics if provided, otherwise default metrics
        if self.custom_metrics:
            return {**self.custom_metrics, "analyzer_id": self.analyzer_id}
        else:
            return {
                "char_count": len(text_content),
                "word_count": len(text_content.split()),
                "analyzer_id": self.analyzer_id,
            }


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
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


@pytest.fixture
def mock_analyzers():
    """Create mock analyzers for testing."""
    return {
        "analyzer1": MockAnalyzer("analyzer1"),
        "analyzer2": MockAnalyzer("analyzer2"),
    }


@pytest.fixture
def basic_config():
    """Create a basic analysis configuration."""
    return AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        analyzers=[],
    )


def test_compute_sample_level_analysis_basic(
    sample_conversations, mock_analyzers, basic_config
):
    """Test basic functionality of compute_sample_level_analysis."""
    dataset = MockDataset(sample_conversations)

    results = compute_sample_level_analysis(dataset, basic_config, mock_analyzers)

    # Check basic structure
    assert results["dataset_name"] == "test_dataset"
    assert results["total_conversations"] == 2
    assert results["conversations_analyzed"] == 2
    assert results["total_messages"] == 4

    # Check messages structure
    messages = results["messages"]
    assert len(messages) == 4

    # Check first message
    first_message = messages[0]
    assert first_message["conversation_id"] == "conv_1"
    assert first_message["conversation_index"] == 0
    assert first_message["message_index"] == 0
    assert first_message["message_id"] == "msg_1_0"
    assert first_message["role"] == "user"
    assert first_message["text_content"] == "Hello, how are you?"

    # Check analyzer metrics are prefixed
    assert "analyzer1_char_count" in first_message
    assert "analyzer1_word_count" in first_message
    assert "analyzer1_analyzer_id" in first_message
    assert "analyzer2_char_count" in first_message
    assert "analyzer2_word_count" in first_message
    assert "analyzer2_analyzer_id" in first_message


def test_compute_sample_level_analysis_with_sample_limit(
    sample_conversations, mock_analyzers
):
    """Test analysis with sample count limit."""
    dataset = MockDataset(sample_conversations)
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=1,  # Only analyze first conversation
        analyzers=[],
    )

    results = compute_sample_level_analysis(dataset, config, mock_analyzers)

    assert results["total_conversations"] == 2
    assert results["conversations_analyzed"] == 1
    assert results["total_messages"] == 2  # Only 2 messages from first conversation

    messages = results["messages"]
    assert len(messages) == 2
    assert all(msg["conversation_index"] == 0 for msg in messages)


def test_compute_sample_level_analysis_sample_count_none(
    sample_conversations, mock_analyzers
):
    """Test analysis with sample_count=None (analyze all conversations)."""
    dataset = MockDataset(sample_conversations)
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=None,  # Analyze all conversations
        analyzers=[],
    )

    results = compute_sample_level_analysis(dataset, config, mock_analyzers)

    assert results["total_conversations"] == 2
    assert results["conversations_analyzed"] == 2
    assert results["total_messages"] == 4


def test_compute_sample_level_analysis_sample_count_zero(
    sample_conversations, mock_analyzers
):
    """Test analysis with sample_count=0 (analyze no conversations)."""
    dataset = MockDataset(sample_conversations)
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=0,  # Should analyze no conversations
        analyzers=[],
    )

    results = compute_sample_level_analysis(dataset, config, mock_analyzers)

    assert results["total_conversations"] == 2
    assert results["conversations_analyzed"] == 0
    assert results["total_messages"] == 0
    assert results["messages"] == []


def test_compute_sample_level_analysis_sample_count_exceeds_total(
    sample_conversations, mock_analyzers
):
    """Test analysis when sample_count exceeds total conversations."""
    dataset = MockDataset(sample_conversations)
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        sample_count=10,  # More than total conversations
        analyzers=[],
    )

    results = compute_sample_level_analysis(dataset, config, mock_analyzers)

    assert results["total_conversations"] == 2
    assert results["conversations_analyzed"] == 2  # Should not exceed total
    assert results["total_messages"] == 4


def test_compute_sample_level_analysis_multimodal_content():
    """Test analysis with multimodal content (non-string content)."""
    # Create message with non-string content
    mock_message = MockMessage("test content", "user")
    # Override content to be a dict for multimodal testing
    mock_message.content = {"text": "Hello world", "image": "image_data"}  # type: ignore

    conversation = MockConversation("conv_1", [mock_message])
    dataset = MockDataset([conversation])

    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        analyzers=[],
    )

    analyzers = {"length": MockAnalyzer("length")}

    results = compute_sample_level_analysis(dataset, config, analyzers)

    assert len(results["messages"]) == 1
    message = results["messages"][0]
    assert (
        message["text_content"]
        == "flattened_{'text': 'Hello world', 'image': 'image_data'}"
    )  # Uses compute_flattened_text_content


def test_compute_sample_level_analysis_missing_conversation_id():
    """Test analysis when conversation_id is None."""
    conversation = MockConversation(None, [MockMessage("Hello", "user")])
    dataset = MockDataset([conversation])

    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        analyzers=[],
    )

    analyzers = {"length": MockAnalyzer("length")}

    results = compute_sample_level_analysis(dataset, config, analyzers)

    assert len(results["messages"]) == 1
    message = results["messages"][0]
    assert message["conversation_id"] == "conv_0"  # Should use fallback


def test_compute_sample_level_analysis_missing_message_id():
    """Test analysis when message_id is None."""
    message = MockMessage("Hello", "user")
    message.id = None
    conversation = MockConversation("conv_1", [message])
    dataset = MockDataset([conversation])

    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        analyzers=[],
    )

    analyzers = {"length": MockAnalyzer("length")}

    results = compute_sample_level_analysis(dataset, config, analyzers)

    assert len(results["messages"]) == 1
    message_data = results["messages"][0]
    assert message_data["message_id"] == "msg_0_0"  # Should use fallback


def test_compute_sample_level_analysis_analyzer_failure(sample_conversations):
    """Test analysis when an analyzer fails."""
    dataset = MockDataset(sample_conversations)
    config = AnalyzeConfig(
        dataset_name="test_dataset",
        split="train",
        analyzers=[],
    )

    # Create analyzers where one fails
    analyzers = {
        "working_analyzer": MockAnalyzer("working_analyzer", should_fail=False),
        "failing_analyzer": MockAnalyzer("failing_analyzer", should_fail=True),
    }

    results = compute_sample_level_analysis(dataset, config, analyzers)

    # Should still process all messages
    assert results["total_messages"] == 4

    # Check that working analyzer metrics are present
    first_message = results["messages"][0]
    assert "working_analyzer_char_count" in first_message
    assert "working_analyzer_word_count" in first_message
    assert "working_analyzer_analyzer_id" in first_message

    # Check that failing analyzer metrics are not present
    assert "failing_analyzer_char_count" not in first_message
    assert "failing_analyzer_word_count" not in first_message
    assert "failing_analyzer_analyzer_id" not in first_message


def test_compute_sample_level_analysis_empty_dataset(mock_analyzers, basic_config):
    """Test analysis with empty dataset."""
    dataset = MockDataset([])

    results = compute_sample_level_analysis(dataset, basic_config, mock_analyzers)

    assert results["dataset_name"] == "test_dataset"
    assert results["total_conversations"] == 0
    assert results["conversations_analyzed"] == 0
    assert results["total_messages"] == 0
    assert results["messages"] == []


def test_compute_sample_level_analysis_empty_conversation(mock_analyzers, basic_config):
    """Test analysis with conversation containing no messages."""
    conversation = MockConversation("conv_1", [])
    dataset = MockDataset([conversation])

    results = compute_sample_level_analysis(dataset, basic_config, mock_analyzers)

    assert results["total_conversations"] == 1
    assert results["conversations_analyzed"] == 1
    assert results["total_messages"] == 0
    assert results["messages"] == []


def test_compute_sample_level_analysis_analyzer_calls(
    sample_conversations, mock_analyzers, basic_config
):
    """Test that analyzers are called with correct parameters."""
    dataset = MockDataset(sample_conversations)

    compute_sample_level_analysis(dataset, basic_config, mock_analyzers)

    # Check that analyzers were called for each message
    analyzer1 = mock_analyzers["analyzer1"]
    analyzer2 = mock_analyzers["analyzer2"]

    assert len(analyzer1.analyze_calls) == 4
    assert len(analyzer2.analyze_calls) == 4

    # Check first call parameters
    text_content, metadata = analyzer1.analyze_calls[0]
    assert text_content == "Hello, how are you?"
    assert metadata["conversation_id"] == "conv_1"
    assert metadata["conversation_index"] == 0
    assert metadata["message_index"] == 0
    assert metadata["role"] == "user"


def test_compute_sample_level_analysis_metric_prefixing(
    sample_conversations, basic_config
):
    """Test that analyzer metrics are properly prefixed to avoid conflicts."""
    dataset = MockDataset(sample_conversations)

    # Create analyzers that return the same metric names
    analyzers = {
        "analyzer1": MockAnalyzer("analyzer1"),
        "analyzer2": MockAnalyzer("analyzer2"),
    }

    results = compute_sample_level_analysis(dataset, basic_config, analyzers)

    first_message = results["messages"][0]

    # Check that metrics are prefixed with analyzer ID
    assert "analyzer1_char_count" in first_message
    assert "analyzer1_word_count" in first_message
    assert "analyzer1_analyzer_id" in first_message
    assert "analyzer2_char_count" in first_message
    assert "analyzer2_word_count" in first_message
    assert "analyzer2_analyzer_id" in first_message

    # Check that values are different (different analyzer IDs)
    assert first_message["analyzer1_analyzer_id"] == "analyzer1"
    assert first_message["analyzer2_analyzer_id"] == "analyzer2"


def test_compute_sample_level_analysis_no_analyzers(sample_conversations, basic_config):
    """Test analysis with no analyzers configured."""
    dataset = MockDataset(sample_conversations)

    results = compute_sample_level_analysis(dataset, basic_config, {})

    assert results["total_messages"] == 4

    # Messages should only contain basic information, no analyzer metrics
    first_message = results["messages"][0]
    basic_fields = {
        "conversation_id",
        "conversation_index",
        "message_index",
        "message_id",
        "role",
        "text_content",
    }

    for field in basic_fields:
        assert field in first_message

    # Should not have any analyzer-prefixed fields
    analyzer_fields = [
        field
        for field in first_message.keys()
        if "_" in field
        and field.split("_")[0]
        in ["analyzer1", "analyzer2", "working_analyzer", "failing_analyzer"]
    ]
    assert len(analyzer_fields) == 0
