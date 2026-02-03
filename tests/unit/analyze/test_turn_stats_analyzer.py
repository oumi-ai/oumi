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

"""Tests for TurnStatsAnalyzer."""

import pytest

from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer, TurnStatsMetrics
from oumi.core.types.conversation import Conversation, Message, Role

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simple_conversation() -> Conversation:
    """Create a simple two-message conversation."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there, how can I help?"),
        ]
    )


@pytest.fixture
def conversation_with_system() -> Conversation:
    """Create a conversation with a system message."""
    return Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What is Python?"),
            Message(
                role=Role.ASSISTANT,
                content="Python is a high-level programming language.",
            ),
        ]
    )


@pytest.fixture
def multi_turn_conversation() -> Conversation:
    """Create a multi-turn conversation."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hi"),
            Message(role=Role.ASSISTANT, content="Hello! How can I help you today?"),
            Message(role=Role.USER, content="What's the weather like?"),
            Message(
                role=Role.ASSISTANT,
                content="I don't have access to weather data, sorry.",
            ),
            Message(role=Role.USER, content="That's okay, thanks anyway!"),
            Message(role=Role.ASSISTANT, content="You're welcome!"),
        ]
    )


@pytest.fixture
def empty_conversation() -> Conversation:
    """Create an empty conversation."""
    return Conversation(messages=[])


@pytest.fixture
def user_only_conversation() -> Conversation:
    """Create a conversation with only user messages."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello?"),
            Message(role=Role.USER, content="Anyone there?"),
        ]
    )


# -----------------------------------------------------------------------------
# TurnStatsMetrics Tests
# -----------------------------------------------------------------------------


def test_turn_stats_metrics_creation():
    """Test that TurnStatsMetrics can be created with required fields."""
    metrics = TurnStatsMetrics(
        num_turns=4,
        num_user_turns=2,
        num_assistant_turns=2,
        has_system_message=False,
        avg_user_chars=10.0,
        avg_assistant_chars=20.0,
        response_ratio=2.0,
        first_turn_role="user",
        last_turn_role="assistant",
    )
    assert metrics.num_turns == 4
    assert metrics.num_user_turns == 2
    assert metrics.num_assistant_turns == 2
    assert metrics.has_system_message is False
    assert metrics.avg_user_chars == 10.0
    assert metrics.avg_assistant_chars == 20.0
    assert metrics.response_ratio == 2.0
    assert metrics.first_turn_role == "user"
    assert metrics.last_turn_role == "assistant"


def test_turn_stats_metrics_defaults():
    """Test TurnStatsMetrics default values."""
    metrics = TurnStatsMetrics(
        num_turns=2,
        num_user_turns=1,
        num_assistant_turns=1,
        has_system_message=False,
        avg_user_chars=5.0,
        avg_assistant_chars=10.0,
        response_ratio=2.0,
        first_turn_role="user",
        last_turn_role="assistant",
    )
    assert metrics.total_user_chars == 0
    assert metrics.total_assistant_chars == 0
    assert metrics.assistant_turn_ratio == 0.0


# -----------------------------------------------------------------------------
# TurnStatsAnalyzer Initialization Tests
# -----------------------------------------------------------------------------


def test_analyzer_default_initialization():
    """Test TurnStatsAnalyzer initializes with default settings."""
    analyzer = TurnStatsAnalyzer()
    assert analyzer.include_system_in_counts is False


def test_analyzer_include_system():
    """Test TurnStatsAnalyzer with system messages included in counts."""
    analyzer = TurnStatsAnalyzer(include_system_in_counts=True)
    assert analyzer.include_system_in_counts is True


# -----------------------------------------------------------------------------
# TurnStatsAnalyzer.analyze() Tests
# -----------------------------------------------------------------------------


def test_analyze_simple_conversation(simple_conversation):
    """Test analyzing a simple conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert isinstance(result, TurnStatsMetrics)
    assert result.num_turns == 2
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 1
    assert result.has_system_message is False
    assert result.first_turn_role == "user"
    assert result.last_turn_role == "assistant"


def test_analyze_response_ratio(simple_conversation):
    """Test that response ratio is computed correctly."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    # "Hello" = 5 chars, "Hi there, how can I help?" = 25 chars
    expected_ratio = 25.0 / 5.0
    assert result.response_ratio == expected_ratio


def test_analyze_conversation_with_system(conversation_with_system):
    """Test analyzing a conversation with a system message."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    # System messages excluded from turn count by default
    assert result.num_turns == 2
    assert result.has_system_message is True
    assert result.first_turn_role == "system"


def test_analyze_conversation_with_system_included(conversation_with_system):
    """Test analyzing with system messages included in count."""
    analyzer = TurnStatsAnalyzer(include_system_in_counts=True)
    result = analyzer.analyze(conversation_with_system)

    # System messages included in turn count
    assert result.num_turns == 3


def test_analyze_multi_turn_conversation(multi_turn_conversation):
    """Test analyzing a multi-turn conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(multi_turn_conversation)

    assert result.num_turns == 6
    assert result.num_user_turns == 3
    assert result.num_assistant_turns == 3
    assert result.first_turn_role == "user"
    assert result.last_turn_role == "assistant"


def test_analyze_assistant_turn_ratio(multi_turn_conversation):
    """Test that assistant turn ratio is computed correctly."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(multi_turn_conversation)

    # 3 assistant turns / 6 total turns = 0.5
    assert result.assistant_turn_ratio == 0.5


def test_analyze_empty_conversation(empty_conversation):
    """Test analyzing an empty conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(empty_conversation)

    assert result.num_turns == 0
    assert result.num_user_turns == 0
    assert result.num_assistant_turns == 0
    assert result.has_system_message is False
    assert result.avg_user_chars == 0.0
    assert result.avg_assistant_chars == 0.0
    assert result.response_ratio == 0.0
    assert result.first_turn_role is None
    assert result.last_turn_role is None


def test_analyze_user_only_conversation(user_only_conversation):
    """Test analyzing a conversation with only user messages."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(user_only_conversation)

    assert result.num_turns == 2
    assert result.num_user_turns == 2
    assert result.num_assistant_turns == 0
    assert result.avg_assistant_chars == 0.0
    assert result.response_ratio == 0.0  # No assistant, so ratio is 0
    assert result.assistant_turn_ratio == 0.0


def test_analyze_character_totals(simple_conversation):
    """Test that character totals are computed correctly."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    # "Hello" = 5 chars
    assert result.total_user_chars == 5
    # "Hi there, how can I help?" = 25 chars
    assert result.total_assistant_chars == 25


def test_analyze_character_averages(multi_turn_conversation):
    """Test that character averages are computed correctly."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(multi_turn_conversation)

    # User messages: "Hi" (2), "What's the weather like?" (24),
    # "That's okay, thanks anyway!" (27)
    expected_user_avg = (2 + 24 + 27) / 3
    assert result.avg_user_chars == expected_user_avg


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_registry():
    """Test that TurnStatsAnalyzer is registered in the core registry."""
    from oumi.core.registry import REGISTRY, RegistryType

    analyzer_class = REGISTRY.get(
        name="typed_turn_stats", type=RegistryType.SAMPLE_ANALYZER
    )
    assert analyzer_class is TurnStatsAnalyzer


# -----------------------------------------------------------------------------
# Analyzer Metadata Tests
# -----------------------------------------------------------------------------


def test_get_result_schema():
    """Test that result schema can be retrieved."""
    schema = TurnStatsAnalyzer.get_result_schema()
    assert "properties" in schema
    assert "num_turns" in schema["properties"]
    assert "response_ratio" in schema["properties"]


def test_get_metric_names():
    """Test that metric names can be retrieved."""
    names = TurnStatsAnalyzer.get_metric_names()
    assert "num_turns" in names
    assert "num_user_turns" in names
    assert "num_assistant_turns" in names
    assert "response_ratio" in names
    assert "first_turn_role" in names


def test_get_metric_descriptions():
    """Test that metric descriptions can be retrieved."""
    descriptions = TurnStatsAnalyzer.get_metric_descriptions()
    assert "num_turns" in descriptions
    assert len(descriptions["num_turns"]) > 0


def test_get_scope():
    """Test that analyzer scope is conversation."""
    assert TurnStatsAnalyzer.get_scope() == "conversation"


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------


def test_single_message_conversation():
    """Test analyzing a conversation with a single message."""
    conversation = Conversation(messages=[Message(role=Role.USER, content="Hello")])
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conversation)

    assert result.num_turns == 1
    assert result.num_user_turns == 1
    assert result.first_turn_role == "user"
    assert result.last_turn_role == "user"


def test_system_only_conversation():
    """Test analyzing a conversation with only a system message."""
    conversation = Conversation(
        messages=[Message(role=Role.SYSTEM, content="System prompt")]
    )
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conversation)

    assert result.num_turns == 0  # System excluded by default
    assert result.has_system_message is True
    assert result.first_turn_role == "system"
    assert result.last_turn_role == "system"


def test_system_only_conversation_included():
    """Test system-only conversation with system included in count."""
    conversation = Conversation(
        messages=[Message(role=Role.SYSTEM, content="System prompt")]
    )
    analyzer = TurnStatsAnalyzer(include_system_in_counts=True)
    result = analyzer.analyze(conversation)

    assert result.num_turns == 1
