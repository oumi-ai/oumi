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
def conversation_with_refusal() -> Conversation:
    """Create a conversation with a policy refusal."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Do something bad."),
            Message(
                role=Role.ASSISTANT,
                content="I cannot assist with that request.",
            ),
        ]
    )


@pytest.fixture
def conversation_with_think_tags() -> Conversation:
    """Create a conversation with balanced think tags."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="What is 2 + 2?"),
            Message(
                role=Role.ASSISTANT,
                content="<think>Simple arithmetic.</think>The answer is 4.",
            ),
        ]
    )


@pytest.fixture
def conversation_with_unbalanced_tags() -> Conversation:
    """Create a conversation with unbalanced think tags."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="What is 2 + 2?"),
            Message(
                role=Role.ASSISTANT,
                content="<think>Let me think about this...",
            ),
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
        has_alternating_turns=True,
        turn_sequence="user,assistant",
        has_empty_turns=False,
        has_invalid_values=False,
        estimated_tokens=10,
        fits_4k_context=True,
        fits_8k_context=True,
        appears_truncated=False,
        has_policy_refusal=False,
        has_think_tags=False,
        has_unbalanced_tags=False,
        passes_basic_quality=True,
    )
    assert metrics.has_alternating_turns is True
    assert metrics.passes_basic_quality is True
    assert metrics.num_consecutive_same_role == 0
    assert metrics.empty_turn_count == 0
    assert metrics.empty_turn_indices == []
    assert metrics.quality_issues == []


def test_metrics_with_all_fields():
    """Test DataQualityMetrics with all fields populated."""
    metrics = DataQualityMetrics(
        has_alternating_turns=False,
        turn_sequence="user,user,assistant",
        num_consecutive_same_role=2,
        has_empty_turns=True,
        empty_turn_count=1,
        empty_turn_indices=[1],
        has_invalid_values=True,
        invalid_value_patterns=["NaN"],
        estimated_tokens=5000,
        fits_4k_context=False,
        fits_8k_context=True,
        appears_truncated=True,
        ends_mid_sentence=True,
        truncation_reason="Ends with comma",
        has_policy_refusal=True,
        refusal_count=1,
        refusal_phrases=["i cannot"],
        has_think_tags=True,
        has_unbalanced_tags=True,
        unmatched_tags=["<think>"],
        passes_basic_quality=False,
        quality_issues=["Non-alternating turns detected"],
    )
    assert metrics.num_consecutive_same_role == 2
    assert metrics.empty_turn_indices == [1]
    assert metrics.invalid_value_patterns == ["NaN"]
    assert metrics.truncation_reason == "Ends with comma"
    assert len(metrics.quality_issues) == 1


# -----------------------------------------------------------------------------
# DataQualityAnalyzer Initialization Tests
# -----------------------------------------------------------------------------


def test_default_initialization():
    """Test DataQualityAnalyzer with default parameters."""
    analyzer = DataQualityAnalyzer()
    assert analyzer.check_turn_pattern is True
    assert analyzer.check_empty_content is True
    assert analyzer.check_invalid_values is True
    assert analyzer.check_truncation is True
    assert analyzer.check_refusals is True
    assert analyzer.check_tags is True
    assert analyzer.context_4k_threshold == 4096
    assert analyzer.context_8k_threshold == 8192
    assert analyzer.tokens_per_word == 1.3


def test_custom_initialization():
    """Test DataQualityAnalyzer with custom parameters."""
    analyzer = DataQualityAnalyzer(
        check_turn_pattern=False,
        check_refusals=False,
        context_4k_threshold=2048,
        tokens_per_word=1.5,
    )
    assert analyzer.check_turn_pattern is False
    assert analyzer.check_refusals is False
    assert analyzer.context_4k_threshold == 2048
    assert analyzer.tokens_per_word == 1.5


# -----------------------------------------------------------------------------
# Turn Pattern Tests
# -----------------------------------------------------------------------------


def test_alternating_turns(simple_conversation):
    """Test detection of proper alternating turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_alternating_turns is True
    assert result.num_consecutive_same_role == 0
    assert result.turn_sequence == "user,assistant"


def test_non_alternating_turns(non_alternating_conversation):
    """Test detection of non-alternating turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(non_alternating_conversation)

    assert result.has_alternating_turns is False
    assert result.num_consecutive_same_role == 2
    assert "Non-alternating turns detected" in result.quality_issues


def test_system_message_ignored_in_alternation(conversation_with_system):
    """Test that system messages are excluded from alternation check."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    assert result.has_alternating_turns is True
    assert "system" in result.turn_sequence


def test_turn_pattern_check_disabled():
    """Test that disabling turn pattern check skips it."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.USER, content="Again"),
        ]
    )
    analyzer = DataQualityAnalyzer(check_turn_pattern=False)
    result = analyzer.analyze(conv)

    assert result.has_alternating_turns is True
    assert result.turn_sequence == ""


# -----------------------------------------------------------------------------
# Empty Content Tests
# -----------------------------------------------------------------------------


def test_no_empty_turns(simple_conversation):
    """Test conversation with no empty turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0
    assert result.empty_turn_indices == []


def test_empty_turns_detected(conversation_with_empty_turns):
    """Test detection of empty turns."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_empty_turns)

    assert result.has_empty_turns is True
    assert result.empty_turn_count == 2
    assert 1 in result.empty_turn_indices
    assert 2 in result.empty_turn_indices


def test_empty_content_check_disabled():
    """Test that disabling empty content check skips it."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content=""),
        ]
    )
    analyzer = DataQualityAnalyzer(check_empty_content=False)
    result = analyzer.analyze(conv)

    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0


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


def test_invalid_values_check_disabled():
    """Test that disabling invalid values check skips it."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Value is NaN"),
            Message(role=Role.ASSISTANT, content="OK."),
        ]
    )
    analyzer = DataQualityAnalyzer(check_invalid_values=False)
    result = analyzer.analyze(conv)

    assert result.has_invalid_values is False


# -----------------------------------------------------------------------------
# Context Length Tests
# -----------------------------------------------------------------------------


def test_fits_context_windows(simple_conversation):
    """Test that a short conversation fits context windows."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.estimated_tokens > 0
    assert result.fits_4k_context is True
    assert result.fits_8k_context is True


def test_exceeds_4k_context():
    """Test detection of conversation exceeding 4K context."""
    # Create a long conversation that exceeds 4K tokens (~3200 words)
    long_text = "word " * 3500
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content=long_text),
            Message(role=Role.ASSISTANT, content="OK."),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.fits_4k_context is False
    assert any("4K" in issue for issue in result.quality_issues)


def test_custom_context_thresholds():
    """Test with custom context thresholds."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="word " * 100),
            Message(role=Role.ASSISTANT, content="OK."),
        ]
    )
    analyzer = DataQualityAnalyzer(context_4k_threshold=50)
    result = analyzer.analyze(conv)

    assert result.fits_4k_context is False


# -----------------------------------------------------------------------------
# Truncation Tests
# -----------------------------------------------------------------------------


def test_no_truncation(simple_conversation):
    """Test conversation that doesn't appear truncated."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.appears_truncated is False


def test_truncation_ends_with_comma():
    """Test detection of truncation when message ends with comma."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="List some things."),
            Message(
                role=Role.ASSISTANT,
                content="Here are some things: apples, oranges,",
            ),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.appears_truncated is True
    assert "truncated" in result.quality_issues[0].lower() or any(
        "truncated" in issue.lower() for issue in result.quality_issues
    )


def test_truncation_ends_with_incomplete_word():
    """Test detection of truncation ending with an incomplete phrase."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Tell me more."),
            Message(
                role=Role.ASSISTANT,
                content="The main reasons are quality and",
            ),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.appears_truncated is True


def test_truncation_check_disabled():
    """Test that disabling truncation check skips it."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Things like,"),
        ]
    )
    analyzer = DataQualityAnalyzer(check_truncation=False)
    result = analyzer.analyze(conv)

    assert result.appears_truncated is False


# -----------------------------------------------------------------------------
# Refusal Detection Tests
# -----------------------------------------------------------------------------


def test_no_refusal(simple_conversation):
    """Test conversation with no policy refusal."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_policy_refusal is False
    assert result.refusal_count == 0
    assert result.refusal_phrases == []


def test_refusal_detected(conversation_with_refusal):
    """Test detection of policy refusal."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_refusal)

    assert result.has_policy_refusal is True
    assert result.refusal_count == 1
    assert len(result.refusal_phrases) > 0


def test_refusal_only_in_assistant():
    """Test that refusal patterns in user messages are ignored."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="I cannot do this."),
            Message(role=Role.ASSISTANT, content="Sure, I can help you!"),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_policy_refusal is False


def test_refusal_check_disabled():
    """Test that disabling refusal check skips it."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Do something bad."),
            Message(
                role=Role.ASSISTANT,
                content="I cannot assist with that.",
            ),
        ]
    )
    analyzer = DataQualityAnalyzer(check_refusals=False)
    result = analyzer.analyze(conv)

    assert result.has_policy_refusal is False


# -----------------------------------------------------------------------------
# Think Tag Tests
# -----------------------------------------------------------------------------


def test_no_think_tags(simple_conversation):
    """Test conversation with no think tags."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.has_think_tags is False
    assert result.has_unbalanced_tags is False


def test_balanced_think_tags(conversation_with_think_tags):
    """Test detection of balanced think tags."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_think_tags)

    assert result.has_think_tags is True
    assert result.has_unbalanced_tags is False
    assert result.unmatched_tags == []


def test_unbalanced_think_tags(conversation_with_unbalanced_tags):
    """Test detection of unbalanced think tags."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conversation_with_unbalanced_tags)

    assert result.has_think_tags is True
    assert result.has_unbalanced_tags is True
    assert len(result.unmatched_tags) > 0


def test_unbalanced_code_blocks():
    """Test detection of unbalanced code blocks."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Show me code."),
            Message(
                role=Role.ASSISTANT,
                content="```python\nprint('hello')\n",
            ),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.has_unbalanced_tags is True
    assert "```" in result.unmatched_tags


def test_tags_check_disabled():
    """Test that disabling tag check skips it."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="<think>Unclosed."),
        ]
    )
    analyzer = DataQualityAnalyzer(check_tags=False)
    result = analyzer.analyze(conv)

    assert result.has_think_tags is False
    assert result.has_unbalanced_tags is False


# -----------------------------------------------------------------------------
# Overall Quality Tests
# -----------------------------------------------------------------------------


def test_passes_basic_quality(simple_conversation):
    """Test that a good conversation passes basic quality."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.passes_basic_quality is True
    assert result.quality_issues == []


def test_fails_basic_quality_multiple_issues():
    """Test that multiple quality issues are reported."""
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.USER, content=""),
            Message(
                role=Role.ASSISTANT,
                content="I cannot help with",
            ),
        ]
    )
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(conv)

    assert result.passes_basic_quality is False
    assert len(result.quality_issues) >= 2


def test_empty_conversation(empty_conversation):
    """Test analysis of an empty conversation."""
    analyzer = DataQualityAnalyzer()
    result = analyzer.analyze(empty_conversation)

    assert isinstance(result, DataQualityMetrics)
    assert result.estimated_tokens == 0
    assert result.has_empty_turns is False


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
    assert "has_alternating_turns" in schema["properties"]
    assert "passes_basic_quality" in schema["properties"]


def test_get_metric_names():
    """Test that metric names can be retrieved."""
    names = DataQualityAnalyzer.get_metric_names()
    assert "has_alternating_turns" in names
    assert "has_empty_turns" in names
    assert "passes_basic_quality" in names


def test_analyzer_is_conversation_analyzer():
    """Test that DataQualityAnalyzer is a ConversationAnalyzer."""
    from oumi.analyze.base import ConversationAnalyzer

    assert issubclass(DataQualityAnalyzer, ConversationAnalyzer)
