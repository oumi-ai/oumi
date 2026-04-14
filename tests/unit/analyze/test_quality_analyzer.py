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

"""Tests for DataQualityAnalyzer."""

import pytest

from oumi.analyze.analyzers.quality import (
    DataQualityAnalyzer,
    DataQualityMetrics,
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
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I'm doing well, thanks!"),
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
def non_alternating_conversation() -> Conversation:
    """Create a conversation with non-alternating turns."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.USER, content="Are you there?"),
            Message(role=Role.ASSISTANT, content="Yes, I'm here!"),
        ]
    )


@pytest.fixture
def conversation_with_empty_turns() -> Conversation:
    """Create a conversation with empty messages."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content=""),
            Message(role=Role.USER, content="  "),
            Message(role=Role.ASSISTANT, content="Sorry about that!"),
        ]
    )


@pytest.fixture
def empty_conversation() -> Conversation:
    """Create an empty conversation."""
    return Conversation(messages=[])


# -----------------------------------------------------------------------------
# DataQualityMetrics Model Tests
# -----------------------------------------------------------------------------


def test_metrics_creation():
    """Test that DataQualityMetrics can be created with required fields."""
    metrics = DataQualityMetrics(
        has_non_alternating_turns=False,
        has_empty_turns=False,
        empty_turn_count=0,
        has_invalid_values=False,
        invalid_value_patterns=[],
    )
    assert metrics.has_non_alternating_turns is False
    assert metrics.has_empty_turns is False
    assert metrics.empty_turn_count == 0
    assert metrics.has_invalid_values is False
    assert metrics.invalid_value_patterns == []


def test_metrics_with_issues():
    """Test DataQualityMetrics with quality issues detected."""
    metrics = DataQualityMetrics(
        has_non_alternating_turns=True,
        has_empty_turns=True,
        empty_turn_count=2,
        has_invalid_values=True,
        invalid_value_patterns=["NaN", "null"],
    )
    assert metrics.has_non_alternating_turns is True
    assert metrics.has_empty_turns is True
    assert metrics.empty_turn_count == 2
    assert metrics.has_invalid_values is True
    assert metrics.invalid_value_patterns == ["NaN", "null"]


# -----------------------------------------------------------------------------
# DataQualityAnalyzer Initialization Tests
# -----------------------------------------------------------------------------


def test_default_initialization():
    """Test DataQualityAnalyzer initializes without errors."""
    analyzer = DataQualityAnalyzer()
    assert analyzer is not None


# -----------------------------------------------------------------------------
# Turn Pattern Tests
# -----------------------------------------------------------------------------


def test_alternating_turns(simple_conversation):
    """Test detection of proper alternating turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_non_alternating_turns is False


def test_non_alternating_turns(non_alternating_conversation):
    """Test detection of non-alternating turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(non_alternating_conversation)

    assert result.has_non_alternating_turns is True


def test_system_message_ignored_in_alternation(conversation_with_system):
    """Test that system messages are excluded from alternation check."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    assert result.has_non_alternating_turns is False


def test_consecutive_assistant_messages():
    """Test detection of consecutive assistant messages."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi!"),
            Message(role=Role.ASSISTANT, content="How can I help?"),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_non_alternating_turns is True


def test_single_message_no_alternation_issue():
    """Test that a single message doesn't flag alternation issues."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_non_alternating_turns is False


# -----------------------------------------------------------------------------
# Empty Content Tests
# -----------------------------------------------------------------------------


def test_no_empty_turns(simple_conversation):
    """Test conversation with no empty turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0


def test_empty_turns_detected(conversation_with_empty_turns):
    """Test detection of empty turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_empty_turns)

    assert result.has_empty_turns is True
    assert result.empty_turn_count == 2


def test_whitespace_only_is_empty():
    """Test that whitespace-only content counts as empty."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="   \t\n  "),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_empty_turns is True
    assert result.empty_turn_count == 1


# -----------------------------------------------------------------------------
# Invalid Values Tests
# -----------------------------------------------------------------------------


def test_no_invalid_values(simple_conversation):
    """Test conversation with no invalid values."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_invalid_values is False
    assert result.invalid_value_patterns == []


def test_nan_detected():
    """Test detection of NaN values in content."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="The value is NaN here."),
            Message(role=Role.ASSISTANT, content="That's invalid."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_invalid_values is True
    assert "NaN" in result.invalid_value_patterns


def test_null_detected():
    """Test detection of null values in content."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="The result was null."),
            Message(role=Role.ASSISTANT, content="Let me check."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_invalid_values is True
    assert "null" in result.invalid_value_patterns


def test_none_detected():
    """Test detection of None values in content."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="It returned None instead."),
            Message(role=Role.ASSISTANT, content="That's a bug."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_invalid_values is True
    assert "None" in result.invalid_value_patterns


def test_undefined_detected():
    """Test detection of undefined values in content."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="The variable is undefined."),
            Message(role=Role.ASSISTANT, content="Check your code."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_invalid_values is True
    assert "undefined" in result.invalid_value_patterns


def test_multiple_invalid_patterns():
    """Test detection of multiple invalid value patterns."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Values are NaN and null here."),
            Message(role=Role.ASSISTANT, content="Also None is bad."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_invalid_values is True
    assert len(result.invalid_value_patterns) >= 3
    assert "NaN" in result.invalid_value_patterns
    assert "null" in result.invalid_value_patterns
    assert "None" in result.invalid_value_patterns


def test_invalid_value_word_boundary():
    """Test that invalid value detection respects word boundaries."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="I like bananas."),
            Message(role=Role.ASSISTANT, content="Me too!"),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    # "banana" contains "nan" but shouldn't match due to word boundary
    assert result.has_invalid_values is False


def test_patterns_are_sorted():
    """Test that invalid value patterns are returned sorted."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="null and NaN"),
            Message(role=Role.ASSISTANT, content="Also undefined."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.invalid_value_patterns == sorted(result.invalid_value_patterns)


# -----------------------------------------------------------------------------
# Empty Conversation Tests
# -----------------------------------------------------------------------------


def test_empty_conversation(empty_conversation):
    """Test analysis of an empty conversation."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(empty_conversation)

    assert isinstance(result, DataQualityMetrics)
    assert result.has_non_alternating_turns is False
    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0
    assert result.has_invalid_values is False
    assert result.invalid_value_patterns == []


# -----------------------------------------------------------------------------
# Combined Scenario Tests
# -----------------------------------------------------------------------------


def test_all_issues_present():
    """Test conversation with all quality issues."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.USER, content=""),
            Message(role=Role.ASSISTANT, content="Value is NaN."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_non_alternating_turns is True
    assert result.has_empty_turns is True
    assert result.empty_turn_count == 1
    assert result.has_invalid_values is True
    assert "NaN" in result.invalid_value_patterns


def test_clean_conversation(simple_conversation):
    """Test that a clean conversation has no quality issues."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_non_alternating_turns is False
    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0
    assert result.has_invalid_values is False
    assert result.invalid_value_patterns == []


# -----------------------------------------------------------------------------
# Registry and Metadata Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_registry():
    """Test that DataQualityAnalyzer is registered in the CLI registry."""
    from oumi.analyze.cli import ANALYZER_REGISTRY

    assert "quality" in ANALYZER_REGISTRY
    assert ANALYZER_REGISTRY["quality"] is DataQualityAnalyzer


def test_get_result_schema():
    """Test that result schema can be retrieved."""
    schema = DataQualityAnalyzer.get_result_schema()
    assert "properties" in schema
    assert "has_non_alternating_turns" in schema["properties"]
    assert "has_empty_turns" in schema["properties"]
    assert "has_invalid_values" in schema["properties"]


def test_get_metric_names():
    """Test that metric names can be retrieved."""
    names = DataQualityAnalyzer.get_metric_names()
    assert "has_non_alternating_turns" in names
    assert "has_empty_turns" in names
    assert "empty_turn_count" in names
    assert "has_invalid_values" in names
    assert "invalid_value_patterns" in names


def test_analyzer_is_conversation_analyzer():
    """Test that DataQualityAnalyzer is a ConversationAnalyzer."""
    from oumi.analyze.base import ConversationAnalyzer

    assert issubclass(DataQualityAnalyzer, ConversationAnalyzer)


def test_get_config_schema():
    """Test that config schema can be retrieved."""
    schema = DataQualityAnalyzer.get_config_schema()
    assert isinstance(schema, dict)
