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

from oumi.analyze.analyzers.turn_stats import (
    TurnStatsAnalyzer,
    TurnStatsMetrics,
)
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
            Message(role=Role.ASSISTANT, content="Hi there!"),
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
            Message(role=Role.ASSISTANT, content="Hello! How can I help?"),
            Message(role=Role.USER, content="What is AI?"),
            Message(
                role=Role.ASSISTANT,
                content="AI stands for Artificial Intelligence.",
            ),
        ]
    )


@pytest.fixture
def empty_conversation() -> Conversation:
    """Create an empty conversation."""
    return Conversation(messages=[])


# -----------------------------------------------------------------------------
# TurnStatsMetrics Model Tests
# -----------------------------------------------------------------------------


def test_metrics_creation():
    """Test that TurnStatsMetrics can be created with required fields."""
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
    assert metrics.num_turns == 2
    assert metrics.num_user_turns == 1
    assert metrics.response_ratio == 2.0
    assert metrics.total_user_chars == 0
    assert metrics.total_assistant_chars == 0
    assert metrics.assistant_turn_ratio == 0.0


def test_metrics_with_all_fields():
    """Test TurnStatsMetrics with all fields populated."""
    metrics = TurnStatsMetrics(
        num_turns=4,
        num_user_turns=2,
        num_assistant_turns=2,
        has_system_message=True,
        avg_user_chars=50.0,
        avg_assistant_chars=150.0,
        total_user_chars=100,
        total_assistant_chars=300,
        response_ratio=3.0,
        assistant_turn_ratio=0.5,
        first_turn_role="system",
        last_turn_role="assistant",
    )
    assert metrics.total_user_chars == 100
    assert metrics.total_assistant_chars == 300
    assert metrics.assistant_turn_ratio == 0.5


# -----------------------------------------------------------------------------
# TurnStatsAnalyzer Initialization Tests
# -----------------------------------------------------------------------------


def test_default_initialization():
    """Test TurnStatsAnalyzer with default parameters."""
    analyzer = TurnStatsAnalyzer()
    assert analyzer.include_system_in_counts is False


def test_include_system_in_counts():
    """Test TurnStatsAnalyzer with include_system_in_counts."""
    analyzer = TurnStatsAnalyzer(include_system_in_counts=True)
    assert analyzer.include_system_in_counts is True


# -----------------------------------------------------------------------------
# Basic Analysis Tests
# -----------------------------------------------------------------------------


def test_simple_conversation(simple_conversation):
    """Test analyzing a simple two-message conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert isinstance(result, TurnStatsMetrics)
    assert result.num_turns == 2
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 1
    assert result.has_system_message is False
    assert result.first_turn_role == "user"
    assert result.last_turn_role == "assistant"


def test_multi_turn_conversation(multi_turn_conversation):
    """Test analyzing a multi-turn conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(multi_turn_conversation)

    assert result.num_turns == 4
    assert result.num_user_turns == 2
    assert result.num_assistant_turns == 2
    assert result.assistant_turn_ratio == 0.5


def test_empty_conversation(empty_conversation):
    """Test analyzing an empty conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(empty_conversation)

    assert result.num_turns == 0
    assert result.num_user_turns == 0
    assert result.num_assistant_turns == 0
    assert result.avg_user_chars == 0.0
    assert result.avg_assistant_chars == 0.0
    assert result.response_ratio == 0.0
    assert result.first_turn_role == ""
    assert result.last_turn_role == ""


# -----------------------------------------------------------------------------
# System Message Handling Tests
# -----------------------------------------------------------------------------


def test_system_message_excluded_from_counts(conversation_with_system):
    """Test that system messages are excluded from turn counts by default."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    assert result.has_system_message is True
    # Default: system messages excluded from num_turns
    assert result.num_turns == 2
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 1


def test_system_message_included_in_counts(conversation_with_system):
    """Test that system messages are included when configured."""
    analyzer = TurnStatsAnalyzer(include_system_in_counts=True)
    result = analyzer.analyze(conversation_with_system)

    assert result.has_system_message is True
    # With include_system: all messages counted
    assert result.num_turns == 3
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 1
    assert result.first_turn_role == "system"


# -----------------------------------------------------------------------------
# Character Length and Ratio Tests
# -----------------------------------------------------------------------------


def test_character_length_stats(simple_conversation):
    """Test character length statistics."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    # "Hello" = 5 chars for user
    assert result.avg_user_chars == 5.0
    assert result.total_user_chars == 5
    # "Hi there!" = 9 chars for assistant
    assert result.avg_assistant_chars == 9.0
    assert result.total_assistant_chars == 9


def test_response_ratio(simple_conversation):
    """Test response ratio calculation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    # Response ratio = avg_assistant_chars / avg_user_chars = 9 / 5
    assert result.response_ratio == pytest.approx(9.0 / 5.0)


def test_response_ratio_zero_user_chars():
    """Test response ratio when user messages have zero length."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content=""),
            Message(role=Role.ASSISTANT, content="Hello!"),
        ]
    )
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conv)

    # avg_user is 0, so response_ratio should be 0
    assert result.response_ratio == 0.0


def test_multi_turn_averages(multi_turn_conversation):
    """Test character averaging across multiple turns."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(multi_turn_conversation)

    # User: "Hi" (2) + "What is AI?" (11) = 13 total, avg = 6.5
    assert result.total_user_chars == 13
    assert result.avg_user_chars == 6.5
    assert result.num_user_turns == 2


# -----------------------------------------------------------------------------
# Turn Ratio Tests
# -----------------------------------------------------------------------------


def test_assistant_turn_ratio_balanced(multi_turn_conversation):
    """Test assistant turn ratio in a balanced conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(multi_turn_conversation)

    assert result.assistant_turn_ratio == 0.5


def test_assistant_turn_ratio_unbalanced():
    """Test assistant turn ratio in an unbalanced conversation."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.USER, content="Are you there?"),
            Message(role=Role.ASSISTANT, content="Yes!"),
        ]
    )
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conv)

    # 1 assistant / 3 total non-system turns
    assert result.assistant_turn_ratio == pytest.approx(1.0 / 3.0)


# -----------------------------------------------------------------------------
# First/Last Turn Role Tests
# -----------------------------------------------------------------------------


def test_first_last_turn_role(simple_conversation):
    """Test first and last turn role tracking."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.first_turn_role == "user"
    assert result.last_turn_role == "assistant"


def test_first_turn_is_system(conversation_with_system):
    """Test first turn role when system message is present."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    assert result.first_turn_role == "system"
    assert result.last_turn_role == "assistant"


def test_single_message():
    """Test conversation with a single message."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
        ]
    )
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conv)

    assert result.num_turns == 1
    assert result.first_turn_role == "user"
    assert result.last_turn_role == "user"
    assert result.assistant_turn_ratio == 0.0


# -----------------------------------------------------------------------------
# Registry and Metadata Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_registry():
    """Test that TurnStatsAnalyzer is registered in the CLI registry."""
    from oumi.analyze.cli import ANALYZER_REGISTRY

    assert "turn_stats" in ANALYZER_REGISTRY
    assert ANALYZER_REGISTRY["turn_stats"] is TurnStatsAnalyzer


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
    assert "response_ratio" in names


def test_analyzer_is_conversation_analyzer():
    """Test that TurnStatsAnalyzer is a ConversationAnalyzer."""
    from oumi.analyze.base import ConversationAnalyzer

    assert issubclass(TurnStatsAnalyzer, ConversationAnalyzer)
