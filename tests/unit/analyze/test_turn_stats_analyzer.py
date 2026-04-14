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
        first_turn_role="user",
        last_turn_role="assistant",
    )
    assert metrics.num_turns == 2
    assert metrics.num_user_turns == 1
    assert metrics.num_assistant_turns == 1
    assert metrics.has_system_message is False
    assert metrics.num_tool_turns == 0  # default
    assert metrics.first_turn_role == "user"
    assert metrics.last_turn_role == "assistant"


def test_metrics_with_all_fields():
    """Test TurnStatsMetrics with all fields populated."""
    metrics = TurnStatsMetrics(
        num_turns=5,
        num_user_turns=2,
        num_assistant_turns=2,
        num_tool_turns=1,
        has_system_message=True,
        first_turn_role="system",
        last_turn_role="assistant",
    )
    assert metrics.num_tool_turns == 1
    assert metrics.has_system_message is True
    assert metrics.first_turn_role == "system"


def test_metrics_none_roles():
    """Test TurnStatsMetrics with None role values (empty conversation)."""
    metrics = TurnStatsMetrics(
        num_turns=0,
        num_user_turns=0,
        num_assistant_turns=0,
        has_system_message=False,
    )
    assert metrics.first_turn_role is None
    assert metrics.last_turn_role is None


# -----------------------------------------------------------------------------
# TurnStatsAnalyzer Initialization Tests
# -----------------------------------------------------------------------------


def test_default_initialization():
    """Test TurnStatsAnalyzer initializes without errors."""
    analyzer = TurnStatsAnalyzer()
    assert analyzer is not None


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


def test_empty_conversation(empty_conversation):
    """Test analyzing an empty conversation."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(empty_conversation)

    assert result.num_turns == 0
    assert result.num_user_turns == 0
    assert result.num_assistant_turns == 0
    assert result.has_system_message is False
    assert result.first_turn_role is None
    assert result.last_turn_role is None


# -----------------------------------------------------------------------------
# System Message Tests
# -----------------------------------------------------------------------------


def test_system_message_detected(conversation_with_system):
    """Test that system messages are detected."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    assert result.has_system_message is True
    assert result.num_turns == 3  # all messages counted
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 1


def test_no_system_message(simple_conversation):
    """Test conversation without system message."""
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_system_message is False


# -----------------------------------------------------------------------------
# Tool Turn Tests
# -----------------------------------------------------------------------------


def test_tool_turns_counted():
    """Test that tool turns are counted."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Use a tool."),
            Message(role=Role.ASSISTANT, content="Let me check."),
            Message(role=Role.TOOL, content='{"result": "done"}'),
            Message(role=Role.ASSISTANT, content="Here's the result."),
        ]
    )
    analyzer = TurnStatsAnalyzer()
    result = analyzer.analyze(conv)

    assert result.num_turns == 4
    assert result.num_tool_turns == 1
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 2


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
    assert result.num_user_turns == 1
    assert result.num_assistant_turns == 0
    assert result.first_turn_role == "user"
    assert result.last_turn_role == "user"


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
    assert "num_user_turns" in schema["properties"]


def test_get_metric_names():
    """Test that metric names can be retrieved."""
    names = TurnStatsAnalyzer.get_metric_names()
    assert "num_turns" in names
    assert "num_user_turns" in names
    assert "num_assistant_turns" in names
    assert "has_system_message" in names


def test_get_config_schema():
    """Test that config schema can be retrieved."""
    schema = TurnStatsAnalyzer.get_config_schema()
    assert isinstance(schema, dict)


def test_analyzer_is_conversation_analyzer():
    """Test that TurnStatsAnalyzer is a ConversationAnalyzer."""
    from oumi.analyze.base import ConversationAnalyzer

    assert issubclass(TurnStatsAnalyzer, ConversationAnalyzer)
