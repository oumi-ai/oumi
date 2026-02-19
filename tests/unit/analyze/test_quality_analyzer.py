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

from oumi.analyze.analyzers.quality import DataQualityAnalyzer, DataQualityMetrics
from oumi.core.types.conversation import ContentItem, Conversation, Message, Role, Type

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def analyzer() -> DataQualityAnalyzer:
    """Create a DataQualityAnalyzer instance."""
    return DataQualityAnalyzer()


@pytest.fixture
def alternating_conversation() -> Conversation:
    """Create a properly alternating user/assistant conversation."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
            Message(role=Role.USER, content="How are you?"),
            Message(role=Role.ASSISTANT, content="I'm doing well, thanks."),
        ]
    )


@pytest.fixture
def non_alternating_conversation() -> Conversation:
    """Create a conversation with consecutive assistant messages."""
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
            Message(role=Role.ASSISTANT, content="Let me add more."),
        ]
    )


@pytest.fixture
def empty_conversation() -> Conversation:
    """Create a conversation with no messages."""
    return Conversation(messages=[])


# -----------------------------------------------------------------------------
# DataQualityMetrics Tests
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


# -----------------------------------------------------------------------------
# Alternating Turns Tests
# -----------------------------------------------------------------------------


def test_alternating_conversation_passes(analyzer, alternating_conversation):
    """Properly alternating conversation is not flagged."""
    result = analyzer.analyze(alternating_conversation)
    assert result.has_non_alternating_turns is False


def test_consecutive_assistant_turns_flagged(analyzer, non_alternating_conversation):
    """Consecutive assistant messages are flagged as non-alternating."""
    result = analyzer.analyze(non_alternating_conversation)
    assert result.has_non_alternating_turns is True


def test_consecutive_user_turns_flagged(analyzer):
    """Consecutive user messages are flagged as non-alternating."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.USER, content="Anyone there?"),
            Message(role=Role.ASSISTANT, content="Hi!"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_non_alternating_turns is True


def test_system_message_ignored_in_alternation_check(analyzer):
    """System messages are excluded from the alternation check."""
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi!"),
            Message(role=Role.USER, content="Thanks"),
            Message(role=Role.ASSISTANT, content="Sure!"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_non_alternating_turns is False


def test_tool_turn_breaks_alternation(analyzer):
    """Tool turns count in the alternation sequence."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="What's the weather?"),
            Message(role=Role.ASSISTANT, content="Let me check."),
            Message(role=Role.TOOL, content='{"temp": 72}'),
            Message(role=Role.ASSISTANT, content="It's 72 degrees."),
        ]
    )
    # assistant -> tool -> assistant: tool breaks it so assistant appears twice
    # non-consecutively relative to itself, but tool != assistant so no consecutive pair
    result = analyzer.analyze(conversation)
    assert result.has_non_alternating_turns is False


def test_empty_conversation_not_flagged(analyzer, empty_conversation):
    """Empty conversation is not flagged as non-alternating."""
    result = analyzer.analyze(empty_conversation)
    assert result.has_non_alternating_turns is False


def test_single_message_not_flagged(analyzer):
    """Single-message conversation is not flagged as non-alternating."""
    conversation = Conversation(messages=[Message(role=Role.USER, content="Hello")])
    result = analyzer.analyze(conversation)
    assert result.has_non_alternating_turns is False


def test_system_only_conversation_not_flagged(analyzer):
    """Conversation with only a system message is not flagged."""
    conversation = Conversation(
        messages=[Message(role=Role.SYSTEM, content="You are helpful.")]
    )
    result = analyzer.analyze(conversation)
    assert result.has_non_alternating_turns is False


# -----------------------------------------------------------------------------
# Empty Turns Tests
# -----------------------------------------------------------------------------


def test_no_empty_turns(analyzer, alternating_conversation):
    """Conversation with no empty messages is not flagged."""
    result = analyzer.analyze(alternating_conversation)
    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0


def test_empty_string_detected(analyzer):
    """Empty string content is detected as an empty turn."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content=""),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_empty_turns is True
    assert result.empty_turn_count == 1


def test_whitespace_only_detected(analyzer):
    """Whitespace-only content is detected as an empty turn."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="  \t\n  "),
            Message(role=Role.ASSISTANT, content="Hello"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_empty_turns is True
    assert result.empty_turn_count == 1


def test_multiple_empty_turns_counted(analyzer):
    """All empty turns are counted."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content=""),
            Message(role=Role.ASSISTANT, content=""),
            Message(role=Role.USER, content="real message"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_empty_turns is True
    assert result.empty_turn_count == 2


def test_empty_conversation_no_empty_turns(analyzer, empty_conversation):
    """Empty conversation has no empty turns."""
    result = analyzer.analyze(empty_conversation)
    assert result.has_empty_turns is False
    assert result.empty_turn_count == 0


# -----------------------------------------------------------------------------
# Invalid Serialized Values Tests
# -----------------------------------------------------------------------------


def test_no_invalid_values(analyzer, alternating_conversation):
    """Conversation with normal content has no invalid values."""
    result = analyzer.analyze(alternating_conversation)
    assert result.has_invalid_values is False
    assert result.invalid_value_patterns == []


@pytest.mark.parametrize(
    "content, expected_pattern",
    [
        ("The value is NaN", "NaN"),
        ("result: nan", "nan"),
        ("data: null", "null"),
        ("field: None", "None"),
        ("x: undefined", "undefined"),
    ],
)
def test_invalid_value_patterns_detected(analyzer, content, expected_pattern):
    """Each invalid value pattern is independently detected."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="What's the value?"),
            Message(role=Role.ASSISTANT, content=content),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_invalid_values is True
    assert expected_pattern in result.invalid_value_patterns


def test_multiple_patterns_all_detected(analyzer):
    """Multiple invalid value patterns in the same message are all reported."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="values: NaN, null, undefined"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_invalid_values is True
    assert "NaN" in result.invalid_value_patterns
    assert "null" in result.invalid_value_patterns
    assert "undefined" in result.invalid_value_patterns


def test_invalid_patterns_returned_sorted(analyzer):
    """Invalid value patterns list is returned sorted."""
    conversation = Conversation(
        messages=[Message(role=Role.USER, content="NaN null None undefined nan")]
    )
    result = analyzer.analyze(conversation)
    assert result.invalid_value_patterns == sorted(result.invalid_value_patterns)


def test_word_boundary_not_partial_match(analyzer):
    """Patterns are matched at word boundaries — substrings do not trigger."""
    conversation = Conversation(
        messages=[
            # "channel" contains "nan" but should not match
            # "nullable" contains "null" but should not match
            Message(role=Role.USER, content="channel nullable notNaN"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_invalid_values is False


def test_invalid_value_in_any_message_detected(analyzer):
    """Invalid values anywhere in the conversation are detected."""
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="The value was NaN"),
            Message(role=Role.ASSISTANT, content="I see."),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_invalid_values is True
    assert "NaN" in result.invalid_value_patterns


# -----------------------------------------------------------------------------
# Multimodal ContentItem Tests
# -----------------------------------------------------------------------------


def test_empty_content_item_list_detected_as_empty(analyzer):
    """A message with an empty ContentItem list is detected as an empty turn."""
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content=[]),
            Message(role=Role.ASSISTANT, content="Hi"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_empty_turns is True
    assert result.empty_turn_count == 1


def test_content_item_text_scanned_for_invalid_values(analyzer):
    """Text inside ContentItem objects is scanned for invalid value patterns."""
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[ContentItem(type=Type.TEXT, content="value is NaN")],
            ),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.has_invalid_values is True
    assert "NaN" in result.invalid_value_patterns


def test_content_item_none_content_no_false_positive(analyzer):
    """A ContentItem with content=None does not trigger a 'None' pattern match."""
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[ContentItem(type=Type.IMAGE_BINARY, binary=b"\x89PNG")],
            ),
        ]
    )
    result = analyzer.analyze(conversation)
    assert "None" not in result.invalid_value_patterns


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_registry():
    """DataQualityAnalyzer is registered under the 'quality' key."""
    from oumi.core.registry import REGISTRY, RegistryType

    analyzer_class = REGISTRY.get(name="quality", type=RegistryType.SAMPLE_ANALYZER)
    assert analyzer_class is DataQualityAnalyzer


# -----------------------------------------------------------------------------
# Analyzer Metadata Tests
# -----------------------------------------------------------------------------


def test_get_result_schema():
    """Result schema includes all metric fields."""
    schema = DataQualityAnalyzer.get_result_schema()
    assert "properties" in schema
    assert "has_non_alternating_turns" in schema["properties"]
    assert "has_empty_turns" in schema["properties"]
    assert "empty_turn_count" in schema["properties"]
    assert "has_invalid_values" in schema["properties"]
    assert "invalid_value_patterns" in schema["properties"]


def test_get_metric_names():
    """All expected metric names are present."""
    names = DataQualityAnalyzer.get_metric_names()
    assert "has_non_alternating_turns" in names
    assert "has_empty_turns" in names
    assert "empty_turn_count" in names
    assert "has_invalid_values" in names
    assert "invalid_value_patterns" in names


def test_get_config_schema():
    """Config schema is empty (no configuration options)."""
    schema = DataQualityAnalyzer.get_config_schema()
    assert schema == {"properties": {}}


def test_get_scope():
    """Analyzer scope is conversation-level."""
    assert DataQualityAnalyzer.get_scope() == "conversation"
