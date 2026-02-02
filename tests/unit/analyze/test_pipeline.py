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

"""Tests for AnalysisPipeline."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from oumi.analyze.base import (
    ConversationAnalyzer,
    DatasetAnalyzer,
    MessageAnalyzer,
    PreferenceAnalyzer,
)
from oumi.analyze.pipeline import AnalysisPipeline
from oumi.core.types.conversation import Conversation, Message, Role

# -----------------------------------------------------------------------------
# Test Result Models
# -----------------------------------------------------------------------------


class SimpleMetrics(BaseModel):
    """Simple metrics for testing."""

    value: int
    name: str


class MessageMetrics(BaseModel):
    """Message-level metrics for testing."""

    char_count: int
    role: str


class DatasetMetrics(BaseModel):
    """Dataset-level metrics for testing."""

    total_conversations: int
    total_messages: int


class PreferenceMetrics(BaseModel):
    """Preference metrics for testing."""

    chosen_longer: bool


# -----------------------------------------------------------------------------
# Test Analyzers
# -----------------------------------------------------------------------------


class SimpleConversationAnalyzer(ConversationAnalyzer[SimpleMetrics]):
    """Simple conversation analyzer for testing."""

    def __init__(self, multiplier: int = 1):
        self.multiplier = multiplier

    def analyze(self, conversation: Conversation) -> SimpleMetrics:
        return SimpleMetrics(
            value=len(conversation.messages) * self.multiplier,
            name=conversation.conversation_id or "unknown",
        )


class SimpleMessageAnalyzer(MessageAnalyzer[MessageMetrics]):
    """Simple message analyzer for testing."""

    def analyze(self, message: Message) -> MessageMetrics:
        content = message.content if isinstance(message.content, str) else ""
        return MessageMetrics(
            char_count=len(content),
            role=message.role.value,
        )


class SimpleDatasetAnalyzer(DatasetAnalyzer[DatasetMetrics]):
    """Simple dataset analyzer for testing."""

    def analyze(self, conversations: list[Conversation]) -> DatasetMetrics:
        total_messages = sum(len(c.messages) for c in conversations)
        return DatasetMetrics(
            total_conversations=len(conversations),
            total_messages=total_messages,
        )


class SimplePreferenceAnalyzer(PreferenceAnalyzer[PreferenceMetrics]):
    """Simple preference analyzer for testing."""

    def analyze(
        self, chosen: Conversation, rejected: Conversation
    ) -> PreferenceMetrics:
        return PreferenceMetrics(
            chosen_longer=len(chosen.messages) > len(rejected.messages)
        )


class DerivedConversationAnalyzer(ConversationAnalyzer[SimpleMetrics]):
    """Derived analyzer that depends on SimpleConversationAnalyzer."""

    depends_on = ["SimpleConversationAnalyzer"]

    def __init__(self):
        self._dependency_results: dict[str, Any] = {}

    def set_dependencies(self, results: dict[str, Any]) -> None:
        self._dependency_results = results

    def analyze(self, conversation: Conversation) -> SimpleMetrics:
        base_results = self._dependency_results.get("SimpleConversationAnalyzer", [])
        base_value = base_results[0].value if base_results else 0
        return SimpleMetrics(
            value=base_value * 2,
            name=f"derived_{conversation.conversation_id or 'unknown'}",
        )


class ChainedAnalyzer(ConversationAnalyzer[SimpleMetrics]):
    """Analyzer that depends on DerivedConversationAnalyzer (chained dependency)."""

    depends_on = ["DerivedConversationAnalyzer"]

    def __init__(self):
        self._dependency_results: dict[str, Any] = {}

    def set_dependencies(self, results: dict[str, Any]) -> None:
        self._dependency_results = results

    def analyze(self, conversation: Conversation) -> SimpleMetrics:
        base_results = self._dependency_results.get("DerivedConversationAnalyzer", [])
        base_value = base_results[0].value if base_results else 0
        return SimpleMetrics(
            value=base_value + 100,
            name=f"chained_{conversation.conversation_id or 'unknown'}",
        )


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_conversations() -> list[Conversation]:
    """Create sample conversations for testing."""
    return [
        Conversation(
            conversation_id="conv1",
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
            ],
        ),
        Conversation(
            conversation_id="conv2",
            messages=[
                Message(role=Role.USER, content="How are you?"),
                Message(role=Role.ASSISTANT, content="I'm doing well, thanks!"),
                Message(role=Role.USER, content="Great!"),
            ],
        ),
    ]


@pytest.fixture
def sample_preference_pairs(
    sample_conversations: list[Conversation],
) -> list[tuple[Conversation, Conversation]]:
    """Create sample preference pairs for testing."""
    return [(sample_conversations[1], sample_conversations[0])]


# -----------------------------------------------------------------------------
# Tests: Initialization
# -----------------------------------------------------------------------------


def test_init_empty():
    """Test initialization with no analyzers."""
    pipeline = AnalysisPipeline(analyzers=[])
    assert pipeline.analyzers == []
    assert pipeline._conversation_analyzers == []
    assert pipeline._message_analyzers == []
    assert pipeline._dataset_analyzers == []
    assert pipeline._preference_analyzers == []


def test_init_categorizes_analyzers():
    """Test that analyzers are properly categorized by type."""
    conv_analyzer = SimpleConversationAnalyzer()
    msg_analyzer = SimpleMessageAnalyzer()
    dataset_analyzer = SimpleDatasetAnalyzer()
    pref_analyzer = SimplePreferenceAnalyzer()

    pipeline = AnalysisPipeline(
        analyzers=[conv_analyzer, msg_analyzer, dataset_analyzer, pref_analyzer]
    )

    assert conv_analyzer in pipeline._conversation_analyzers
    assert msg_analyzer in pipeline._message_analyzers
    assert dataset_analyzer in pipeline._dataset_analyzers
    assert pref_analyzer in pipeline._preference_analyzers


def test_init_with_cache_dir(tmp_path: Path):
    """Test initialization with cache directory."""
    pipeline = AnalysisPipeline(analyzers=[], cache_dir=tmp_path)
    assert pipeline.cache_dir == tmp_path


def test_init_with_string_cache_dir(tmp_path: Path):
    """Test initialization with string cache directory."""
    pipeline = AnalysisPipeline(analyzers=[], cache_dir=str(tmp_path))
    assert pipeline.cache_dir == tmp_path


# -----------------------------------------------------------------------------
# Tests: Run
# -----------------------------------------------------------------------------


def test_run_conversation_analyzer(sample_conversations: list[Conversation]):
    """Test running a conversation analyzer."""
    analyzer = SimpleConversationAnalyzer()
    pipeline = AnalysisPipeline(analyzers=[analyzer])

    results = pipeline.run(sample_conversations)

    assert "SimpleConversationAnalyzer" in results
    conv_results = results["SimpleConversationAnalyzer"]
    assert isinstance(conv_results, list)
    assert len(conv_results) == 2
    assert conv_results[0].value == 2  # 2 messages in conv1
    assert conv_results[1].value == 3  # 3 messages in conv2


def test_run_message_analyzer(sample_conversations: list[Conversation]):
    """Test running a message analyzer."""
    analyzer = SimpleMessageAnalyzer()
    pipeline = AnalysisPipeline(analyzers=[analyzer])

    results = pipeline.run(sample_conversations)

    assert "SimpleMessageAnalyzer" in results
    msg_results = results["SimpleMessageAnalyzer"]
    assert isinstance(msg_results, list)
    # Total messages: 2 + 3 = 5
    assert len(msg_results) == 5


def test_run_dataset_analyzer(sample_conversations: list[Conversation]):
    """Test running a dataset analyzer."""
    analyzer = SimpleDatasetAnalyzer()
    pipeline = AnalysisPipeline(analyzers=[analyzer])

    results = pipeline.run(sample_conversations)

    assert "SimpleDatasetAnalyzer" in results
    dataset_result = results["SimpleDatasetAnalyzer"]
    assert isinstance(dataset_result, DatasetMetrics)
    assert dataset_result.total_conversations == 2
    assert dataset_result.total_messages == 5


def test_run_multiple_analyzers(sample_conversations: list[Conversation]):
    """Test running multiple analyzers of different types."""
    pipeline = AnalysisPipeline(
        analyzers=[
            SimpleConversationAnalyzer(),
            SimpleMessageAnalyzer(),
            SimpleDatasetAnalyzer(),
        ]
    )

    results = pipeline.run(sample_conversations)

    assert len(results) == 3
    assert "SimpleConversationAnalyzer" in results
    assert "SimpleMessageAnalyzer" in results
    assert "SimpleDatasetAnalyzer" in results


def test_run_stores_conversations(sample_conversations: list[Conversation]):
    """Test that run() stores the conversations."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])
    pipeline.run(sample_conversations)

    assert pipeline.conversations == sample_conversations


def test_run_builds_message_index(sample_conversations: list[Conversation]):
    """Test that run() builds the message-to-conversation index."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])
    pipeline.run(sample_conversations)

    # conv1 has 2 messages (indices 0, 1), conv2 has 3 messages (indices 2, 3, 4)
    assert pipeline.message_to_conversation_idx == [0, 0, 1, 1, 1]


def test_run_empty_conversations():
    """Test running on empty conversation list."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])
    results = pipeline.run([])

    assert "SimpleConversationAnalyzer" in results
    assert results["SimpleConversationAnalyzer"] == []


# -----------------------------------------------------------------------------
# Tests: Dependency Handling
# -----------------------------------------------------------------------------


def test_derived_analyzer_receives_dependencies(
    sample_conversations: list[Conversation],
):
    """Test that derived analyzers receive dependency results."""
    base_analyzer = SimpleConversationAnalyzer()
    derived_analyzer = DerivedConversationAnalyzer()

    pipeline = AnalysisPipeline(analyzers=[derived_analyzer, base_analyzer])
    results = pipeline.run(sample_conversations)

    assert "DerivedConversationAnalyzer" in results
    derived_results = results["DerivedConversationAnalyzer"]
    assert len(derived_results) == 2


def test_chained_dependencies(sample_conversations: list[Conversation]):
    """Test chained dependencies (A -> B -> C)."""
    base = SimpleConversationAnalyzer()
    derived = DerivedConversationAnalyzer()
    chained = ChainedAnalyzer()

    # Add in reverse order to test topological sort
    pipeline = AnalysisPipeline(analyzers=[chained, base, derived])
    results = pipeline.run(sample_conversations)

    assert "SimpleConversationAnalyzer" in results
    assert "DerivedConversationAnalyzer" in results
    assert "ChainedAnalyzer" in results


def test_circular_dependency_raises_error(sample_conversations: list[Conversation]):
    """Test that circular dependencies raise an error."""

    class CircularA(ConversationAnalyzer[SimpleMetrics]):
        depends_on = ["CircularB"]

        def analyze(self, conversation: Conversation) -> SimpleMetrics:
            return SimpleMetrics(value=1, name="a")

    class CircularB(ConversationAnalyzer[SimpleMetrics]):
        depends_on = ["CircularA"]

        def analyze(self, conversation: Conversation) -> SimpleMetrics:
            return SimpleMetrics(value=1, name="b")

    pipeline = AnalysisPipeline(analyzers=[CircularA(), CircularB()])

    with pytest.raises(ValueError, match="Circular dependency"):
        pipeline.run(sample_conversations)


# -----------------------------------------------------------------------------
# Tests: Preference Analyzers
# -----------------------------------------------------------------------------


def test_run_preference(
    sample_preference_pairs: list[tuple[Conversation, Conversation]],
):
    """Test running preference analyzers."""
    analyzer = SimplePreferenceAnalyzer()
    pipeline = AnalysisPipeline(analyzers=[analyzer])

    results = pipeline.run_preference(sample_preference_pairs)

    assert "SimplePreferenceAnalyzer" in results
    pref_results = results["SimplePreferenceAnalyzer"]
    assert len(pref_results) == 1
    assert pref_results[0].chosen_longer is True  # conv2 (3 msgs) > conv1 (2 msgs)


def test_preference_not_run_by_run(sample_conversations: list[Conversation]):
    """Test that preference analyzers are not run by run()."""
    conv_analyzer = SimpleConversationAnalyzer()
    pref_analyzer = SimplePreferenceAnalyzer()

    pipeline = AnalysisPipeline(analyzers=[conv_analyzer, pref_analyzer])
    results = pipeline.run(sample_conversations)

    assert "SimpleConversationAnalyzer" in results
    assert "SimplePreferenceAnalyzer" not in results


# -----------------------------------------------------------------------------
# Tests: Caching
# -----------------------------------------------------------------------------


def test_save_cache(sample_conversations: list[Conversation], tmp_path: Path):
    """Test that results are saved to cache."""
    pipeline = AnalysisPipeline(
        analyzers=[SimpleConversationAnalyzer()],
        cache_dir=tmp_path,
    )
    pipeline.run(sample_conversations)

    cache_file = tmp_path / "analysis_results.json"
    assert cache_file.exists()


def test_load_cache(sample_conversations: list[Conversation], tmp_path: Path):
    """Test loading results from cache."""
    # First, run and save
    pipeline1 = AnalysisPipeline(
        analyzers=[SimpleConversationAnalyzer()],
        cache_dir=tmp_path,
    )
    pipeline1.run(sample_conversations)

    # Then, load in new pipeline
    pipeline2 = AnalysisPipeline(
        analyzers=[SimpleConversationAnalyzer()],
        cache_dir=tmp_path,
    )
    loaded = pipeline2.load_cache()

    assert loaded is True
    assert "SimpleConversationAnalyzer" in pipeline2.results


def test_load_cache_no_cache_dir():
    """Test load_cache returns False when no cache_dir."""
    pipeline = AnalysisPipeline(analyzers=[])
    assert pipeline.load_cache() is False


def test_load_cache_missing_file(tmp_path: Path):
    """Test load_cache returns False when file doesn't exist."""
    pipeline = AnalysisPipeline(analyzers=[], cache_dir=tmp_path)
    assert pipeline.load_cache() is False


# -----------------------------------------------------------------------------
# Tests: Utility Methods
# -----------------------------------------------------------------------------


def test_get_analyzer():
    """Test getting an analyzer by name."""
    analyzer = SimpleConversationAnalyzer()
    pipeline = AnalysisPipeline(analyzers=[analyzer])

    found = pipeline.get_analyzer("SimpleConversationAnalyzer")
    assert found is analyzer


def test_get_analyzer_not_found():
    """Test get_analyzer returns None for unknown name."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])
    assert pipeline.get_analyzer("NonExistent") is None


def test_get_analyzer_with_custom_id():
    """Test get_analyzer with custom analyzer_id."""
    analyzer = SimpleConversationAnalyzer()
    analyzer.analyzer_id = "custom_name"  # type: ignore[attr-defined]

    pipeline = AnalysisPipeline(analyzers=[analyzer])

    assert pipeline.get_analyzer("custom_name") is analyzer
    assert pipeline.get_analyzer("SimpleConversationAnalyzer") is None


def test_to_dataframe(sample_conversations: list[Conversation]):
    """Test converting results to DataFrame."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])
    pipeline.run(sample_conversations)

    result_df = pipeline.to_dataframe()

    assert len(result_df) == 2
    assert "conversation_id" in result_df.columns
    assert "simpleconversation__value" in result_df.columns


def test_to_dataframe_raises_without_results():
    """Test to_dataframe raises error when no results."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])

    with pytest.raises(RuntimeError, match="No results available"):
        pipeline.to_dataframe()


def test_results_property(sample_conversations: list[Conversation]):
    """Test results property."""
    pipeline = AnalysisPipeline(analyzers=[SimpleConversationAnalyzer()])
    pipeline.run(sample_conversations)

    assert pipeline.results == pipeline._results


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


def test_analyzer_with_custom_id_in_results(sample_conversations: list[Conversation]):
    """Test that custom analyzer_id is used in results keys."""
    analyzer = SimpleConversationAnalyzer()
    analyzer.analyzer_id = "my_analyzer"  # type: ignore[attr-defined]

    pipeline = AnalysisPipeline(analyzers=[analyzer])
    results = pipeline.run(sample_conversations)

    assert "my_analyzer" in results
    assert "SimpleConversationAnalyzer" not in results


def test_multiple_analyzers_same_type(sample_conversations: list[Conversation]):
    """Test multiple analyzers of the same type with different IDs."""
    analyzer1 = SimpleConversationAnalyzer(multiplier=1)
    analyzer1.analyzer_id = "conv_analyzer_1"  # type: ignore[attr-defined]

    analyzer2 = SimpleConversationAnalyzer(multiplier=2)
    analyzer2.analyzer_id = "conv_analyzer_2"  # type: ignore[attr-defined]

    pipeline = AnalysisPipeline(analyzers=[analyzer1, analyzer2])
    results = pipeline.run(sample_conversations)

    assert "conv_analyzer_1" in results
    assert "conv_analyzer_2" in results
    # First conversation has 2 messages
    assert results["conv_analyzer_1"][0].value == 2  # 2 * 1
    assert results["conv_analyzer_2"][0].value == 4  # 2 * 2
