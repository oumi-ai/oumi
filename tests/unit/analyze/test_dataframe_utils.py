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

"""Tests for DataFrame conversion utilities."""

import pytest
from pydantic import BaseModel

from oumi.analyze.utils.dataframe import (
    _add_result_to_row,
    _get_column_prefix,
    results_to_dict,
    to_analysis_dataframe,
    to_message_dataframe,
)
from oumi.core.types.conversation import Conversation, Message, Role

# -----------------------------------------------------------------------------
# Test Result Models
# -----------------------------------------------------------------------------


class SimpleMetrics(BaseModel):
    """Simple metrics for testing."""

    score: int
    name: str


class MetricsWithList(BaseModel):
    """Metrics containing a list field."""

    values: list[int]
    total: int


class MetricsWithNested(BaseModel):
    """Metrics with nested structure."""

    stats: dict[str, int]
    label: str


class DatasetMetrics(BaseModel):
    """Dataset-level metrics for testing."""

    total_count: int
    avg_score: float


class MessageMetrics(BaseModel):
    """Message-level metrics for testing."""

    char_count: int
    word_count: int


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
def conversation_without_id() -> list[Conversation]:
    """Create conversations without explicit IDs."""
    return [
        Conversation(
            messages=[
                Message(role=Role.USER, content="Test"),
            ],
        ),
    ]


# -----------------------------------------------------------------------------
# Tests: _get_column_prefix
# -----------------------------------------------------------------------------


def test_get_column_prefix_removes_analyzer_suffix():
    """Test that 'Analyzer' suffix is removed."""
    assert _get_column_prefix("LengthAnalyzer") == "length"
    assert _get_column_prefix("QualityAnalyzer") == "quality"
    assert _get_column_prefix("DataQualityAnalyzer") == "dataquality"


def test_get_column_prefix_removes_metrics_suffix():
    """Test that 'Metrics' suffix is removed."""
    assert _get_column_prefix("LengthMetrics") == "length"
    assert _get_column_prefix("TurnStatsMetrics") == "turnstats"


def test_get_column_prefix_no_suffix():
    """Test names without common suffixes."""
    assert _get_column_prefix("CustomName") == "customname"
    assert _get_column_prefix("MyChecker") == "mychecker"


def test_get_column_prefix_lowercase():
    """Test that result is always lowercase."""
    assert _get_column_prefix("UPPERCASE") == "uppercase"
    assert _get_column_prefix("MixedCase") == "mixedcase"


# -----------------------------------------------------------------------------
# Tests: _add_result_to_row
# -----------------------------------------------------------------------------


def test_add_result_to_row_scalar_fields():
    """Test adding scalar fields to a row."""
    row: dict = {}
    result = SimpleMetrics(score=85, name="test")

    _add_result_to_row(row, result, "simple")

    assert row["simple__score"] == 85
    assert row["simple__name"] == "test"


def test_add_result_to_row_list_field():
    """Test adding list fields to a row."""
    row: dict = {}
    result = MetricsWithList(values=[1, 2, 3], total=6)

    _add_result_to_row(row, result, "metrics")

    assert row["metrics__values"] == [1, 2, 3]
    assert row["metrics__total"] == 6


def test_add_result_to_row_nested_dict():
    """Test that nested dicts are flattened."""
    row: dict = {}
    result = MetricsWithNested(stats={"min": 0, "max": 100}, label="test")

    _add_result_to_row(row, result, "nested")

    assert row["nested__stats__min"] == 0
    assert row["nested__stats__max"] == 100
    assert row["nested__label"] == "test"


def test_add_result_to_row_preserves_existing():
    """Test that existing row data is preserved."""
    row: dict = {"existing_key": "existing_value"}
    result = SimpleMetrics(score=50, name="added")

    _add_result_to_row(row, result, "new")

    assert row["existing_key"] == "existing_value"
    assert row["new__score"] == 50


def test_add_result_to_row_handles_dict():
    """Test that raw dicts (from cache) are handled correctly."""
    row: dict = {}
    # Simulate cached result loaded from JSON (raw dict, not Pydantic model)
    cached_result = {"score": 75, "name": "from_cache"}

    _add_result_to_row(row, cached_result, "cached")

    assert row["cached__score"] == 75
    assert row["cached__name"] == "from_cache"


# -----------------------------------------------------------------------------
# Tests: to_analysis_dataframe
# -----------------------------------------------------------------------------


def test_to_analysis_dataframe_basic(sample_conversations: list[Conversation]):
    """Test basic DataFrame creation."""
    results = {
        "SimpleAnalyzer": [
            SimpleMetrics(score=80, name="first"),
            SimpleMetrics(score=90, name="second"),
        ]
    }

    df = to_analysis_dataframe(sample_conversations, results)

    assert len(df) == 2
    assert "conversation_id" in df.columns
    assert "conversation_index" in df.columns
    assert "num_messages" in df.columns
    assert "simple__score" in df.columns
    assert "simple__name" in df.columns


def test_to_analysis_dataframe_conversation_metadata(
    sample_conversations: list[Conversation],
):
    """Test that conversation metadata is correct."""
    df = to_analysis_dataframe(sample_conversations, {})

    assert df.iloc[0]["conversation_id"] == "conv1"
    assert df.iloc[0]["conversation_index"] == 0
    assert df.iloc[0]["num_messages"] == 2

    assert df.iloc[1]["conversation_id"] == "conv2"
    assert df.iloc[1]["conversation_index"] == 1
    assert df.iloc[1]["num_messages"] == 3


def test_to_analysis_dataframe_fallback_conversation_id(
    conversation_without_id: list[Conversation],
):
    """Test fallback ID when conversation has no ID."""
    df = to_analysis_dataframe(conversation_without_id, {})

    assert df.iloc[0]["conversation_id"] == "conv_0"


def test_to_analysis_dataframe_dataset_level_result(
    sample_conversations: list[Conversation],
):
    """Test that dataset-level results are repeated for each row."""
    results = {
        "DatasetAnalyzer": DatasetMetrics(total_count=100, avg_score=75.5),
    }

    df = to_analysis_dataframe(sample_conversations, results)

    # Dataset-level result should appear in both rows
    assert df.iloc[0]["dataset__total_count"] == 100
    assert df.iloc[0]["dataset__avg_score"] == 75.5
    assert df.iloc[1]["dataset__total_count"] == 100
    assert df.iloc[1]["dataset__avg_score"] == 75.5


def test_to_analysis_dataframe_multiple_analyzers(
    sample_conversations: list[Conversation],
):
    """Test multiple analyzers in results."""
    results = {
        "SimpleAnalyzer": [
            SimpleMetrics(score=80, name="a"),
            SimpleMetrics(score=90, name="b"),
        ],
        "ListAnalyzer": [
            MetricsWithList(values=[1, 2], total=3),
            MetricsWithList(values=[4, 5], total=9),
        ],
    }

    df = to_analysis_dataframe(sample_conversations, results)

    assert "simple__score" in df.columns
    assert "list__values" in df.columns


def test_to_analysis_dataframe_empty_conversations():
    """Test with empty conversation list."""
    df = to_analysis_dataframe([], {})

    assert len(df) == 0


def test_to_analysis_dataframe_empty_results(
    sample_conversations: list[Conversation],
):
    """Test with empty results dictionary."""
    df = to_analysis_dataframe(sample_conversations, {})

    assert len(df) == 2
    assert "conversation_id" in df.columns
    # Only metadata columns, no analyzer columns
    assert len(df.columns) == 3


# -----------------------------------------------------------------------------
# Tests: to_message_dataframe
# -----------------------------------------------------------------------------


def test_to_message_dataframe_basic(sample_conversations: list[Conversation]):
    """Test basic message-level DataFrame creation."""
    # 5 total messages: 2 in conv1 + 3 in conv2
    results = {
        "MessageAnalyzer": [
            MessageMetrics(char_count=5, word_count=1),  # "Hello"
            MessageMetrics(char_count=10, word_count=2),  # "Hi there!"
            MessageMetrics(char_count=12, word_count=3),  # "How are you?"
            MessageMetrics(char_count=22, word_count=4),  # "I'm doing well, thanks!"
            MessageMetrics(char_count=6, word_count=1),  # "Great!"
        ],
    }

    df = to_message_dataframe(sample_conversations, results)

    assert len(df) == 5
    assert "conversation_id" in df.columns
    assert "message_index" in df.columns
    assert "role" in df.columns
    assert "text_content" in df.columns
    assert "message__char_count" in df.columns


def test_to_message_dataframe_message_metadata(
    sample_conversations: list[Conversation],
):
    """Test that message metadata is correct."""
    df = to_message_dataframe(sample_conversations, {})

    # First message of first conversation
    assert df.iloc[0]["conversation_id"] == "conv1"
    assert df.iloc[0]["conversation_index"] == 0
    assert df.iloc[0]["message_index"] == 0
    assert df.iloc[0]["role"] == "user"
    assert df.iloc[0]["text_content"] == "Hello"

    # Second message of first conversation
    assert df.iloc[1]["message_index"] == 1
    assert df.iloc[1]["role"] == "assistant"

    # First message of second conversation
    assert df.iloc[2]["conversation_id"] == "conv2"
    assert df.iloc[2]["conversation_index"] == 1
    assert df.iloc[2]["message_index"] == 0


def test_to_message_dataframe_empty_conversations():
    """Test with empty conversation list."""
    df = to_message_dataframe([], {})

    assert len(df) == 0


# -----------------------------------------------------------------------------
# Tests: results_to_dict
# -----------------------------------------------------------------------------


def test_results_to_dict_list_results():
    """Test converting list results to dict."""
    results = {
        "SimpleAnalyzer": [
            SimpleMetrics(score=80, name="a"),
            SimpleMetrics(score=90, name="b"),
        ],
    }

    output = results_to_dict(results)

    assert "SimpleAnalyzer" in output
    assert isinstance(output["SimpleAnalyzer"], list)
    assert len(output["SimpleAnalyzer"]) == 2
    assert output["SimpleAnalyzer"][0]["score"] == 80
    assert output["SimpleAnalyzer"][1]["score"] == 90


def test_results_to_dict_single_result():
    """Test converting single (dataset-level) result to dict."""
    results = {
        "DatasetAnalyzer": DatasetMetrics(total_count=100, avg_score=75.5),
    }

    output = results_to_dict(results)

    assert "DatasetAnalyzer" in output
    assert isinstance(output["DatasetAnalyzer"], dict)
    assert output["DatasetAnalyzer"]["total_count"] == 100
    assert output["DatasetAnalyzer"]["avg_score"] == 75.5


def test_results_to_dict_mixed_results():
    """Test converting mixed list and single results."""
    results = {
        "SimpleAnalyzer": [
            SimpleMetrics(score=80, name="a"),
        ],
        "DatasetAnalyzer": DatasetMetrics(total_count=50, avg_score=60.0),
    }

    output = results_to_dict(results)

    assert isinstance(output["SimpleAnalyzer"], list)
    assert isinstance(output["DatasetAnalyzer"], dict)


def test_results_to_dict_empty():
    """Test with empty results."""
    output = results_to_dict({})

    assert output == {}
