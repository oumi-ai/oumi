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

"""Tests for LengthAnalyzer."""

import pytest

from oumi.analyze.analyzers.length import LengthAnalyzer, LengthMetrics
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
def empty_conversation() -> Conversation:
    """Create an empty conversation."""
    return Conversation(messages=[])


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer that splits on whitespace."""

    class MockTokenizer:
        def encode(self, text: str) -> list[int]:
            # Simple whitespace tokenization for testing
            return list(range(len(text.split())))

    return MockTokenizer()


# -----------------------------------------------------------------------------
# LengthMetrics Tests
# -----------------------------------------------------------------------------


def test_length_metrics_creation():
    """Test that LengthMetrics can be created with required fields."""
    metrics = LengthMetrics(
        total_tokens=10,
        avg_tokens_per_message=5.0,
        message_token_counts=[4, 6],
        num_messages=2,
    )
    assert metrics.total_tokens == 10
    assert metrics.avg_tokens_per_message == 5.0
    assert metrics.message_token_counts == [4, 6]
    assert metrics.num_messages == 2
    # Role stats default to 0
    assert metrics.user_total_tokens == 0
    assert metrics.assistant_total_tokens == 0
    assert metrics.system_total_tokens == 0


def test_length_metrics_with_role_stats():
    """Test LengthMetrics with role-specific statistics."""
    metrics = LengthMetrics(
        total_tokens=15,
        avg_tokens_per_message=5.0,
        message_token_counts=[5, 5, 5],
        num_messages=3,
        user_total_tokens=5,
        assistant_total_tokens=5,
        system_total_tokens=5,
    )
    assert metrics.user_total_tokens == 5
    assert metrics.assistant_total_tokens == 5
    assert metrics.system_total_tokens == 5


# -----------------------------------------------------------------------------
# LengthAnalyzer Initialization Tests
# -----------------------------------------------------------------------------


def test_analyzer_default_initialization():
    """Test LengthAnalyzer initializes with default settings."""
    analyzer = LengthAnalyzer()
    assert analyzer.tokenizer is None
    assert analyzer.tiktoken_encoding == "cl100k_base"


def test_analyzer_custom_encoding():
    """Test LengthAnalyzer with custom tiktoken encoding."""
    analyzer = LengthAnalyzer(tiktoken_encoding="p50k_base")
    assert analyzer.tiktoken_encoding == "p50k_base"


def test_analyzer_with_custom_tokenizer(mock_tokenizer):
    """Test LengthAnalyzer with a custom tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)
    assert analyzer.tokenizer is mock_tokenizer
    assert analyzer._tiktoken_encoder is None


# -----------------------------------------------------------------------------
# LengthAnalyzer.analyze() Tests
# -----------------------------------------------------------------------------


def test_analyze_simple_conversation(simple_conversation):
    """Test analyzing a simple conversation."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert isinstance(result, LengthMetrics)
    assert result.num_messages == 2
    assert len(result.message_token_counts) == 2
    assert result.total_tokens == sum(result.message_token_counts)
    assert result.avg_tokens_per_message == result.total_tokens / 2


def test_analyze_role_stats(simple_conversation):
    """Test that role stats are always computed."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze(simple_conversation)

    assert result.user_total_tokens > 0
    assert result.assistant_total_tokens > 0
    # System should be 0 since there's no system message
    assert result.system_total_tokens == 0


def test_analyze_conversation_with_system(conversation_with_system):
    """Test analyzing a conversation with a system message."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze(conversation_with_system)

    assert result.num_messages == 3
    assert result.system_total_tokens > 0


def test_analyze_empty_conversation(empty_conversation):
    """Test analyzing an empty conversation."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze(empty_conversation)

    assert result.num_messages == 0
    assert result.total_tokens == 0
    assert result.avg_tokens_per_message == 0.0
    assert result.message_token_counts == []


def test_analyze_with_custom_tokenizer(simple_conversation, mock_tokenizer):
    """Test analyzing with a custom tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)
    result = analyzer.analyze(simple_conversation)

    # Mock tokenizer splits on whitespace
    # "Hello" -> 1 token, "Hi there!" -> 2 tokens
    assert result.message_token_counts == [1, 2]
    assert result.total_tokens == 3


# -----------------------------------------------------------------------------
# LengthAnalyzer.analyze_text() Tests
# -----------------------------------------------------------------------------


def test_analyze_text_simple():
    """Test analyzing a simple text string."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze_text("Hello, world!")

    assert result.num_messages == 1
    assert len(result.message_token_counts) == 1
    assert result.total_tokens == result.message_token_counts[0]
    assert result.avg_tokens_per_message == float(result.total_tokens)


def test_analyze_text_with_custom_tokenizer(mock_tokenizer):
    """Test analyze_text with a custom tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)
    result = analyzer.analyze_text("one two three four")

    # Mock tokenizer splits on whitespace -> 4 tokens
    assert result.total_tokens == 4


def test_analyze_text_empty():
    """Test analyzing empty text."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze_text("")

    assert result.total_tokens == 0
    assert result.num_messages == 1


def test_analyze_text_role_stats_are_zero():
    """Test that analyze_text returns zero for role stats (no conversation context)."""
    analyzer = LengthAnalyzer()
    result = analyzer.analyze_text("Some text")

    # No conversation context, so role stats default to 0
    assert result.user_total_tokens == 0
    assert result.assistant_total_tokens == 0
    assert result.system_total_tokens == 0


# -----------------------------------------------------------------------------
# Token Counting Tests
# -----------------------------------------------------------------------------


def test_count_tokens_with_tiktoken():
    """Test token counting with tiktoken encoder."""
    analyzer = LengthAnalyzer()
    if analyzer._tiktoken_encoder is not None:
        count = analyzer._count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)


def test_count_tokens_with_custom_tokenizer(mock_tokenizer):
    """Test token counting with custom tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)
    count = analyzer._count_tokens("one two three")
    assert count == 3


def test_count_tokens_empty_string():
    """Test token counting with empty string."""
    analyzer = LengthAnalyzer()
    count = analyzer._count_tokens("")
    assert count == 0


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_registry():
    """Test that LengthAnalyzer is registered in the core registry."""
    from oumi.core.registry import REGISTRY, RegistryType

    analyzer_class = REGISTRY.get(name="typed_length", type=RegistryType.SAMPLE_ANALYZER)
    assert analyzer_class is LengthAnalyzer


# -----------------------------------------------------------------------------
# Analyzer Metadata Tests
# -----------------------------------------------------------------------------


def test_get_result_schema():
    """Test that result schema can be retrieved."""
    schema = LengthAnalyzer.get_result_schema()
    assert "properties" in schema
    assert "total_tokens" in schema["properties"]
    assert "avg_tokens_per_message" in schema["properties"]


def test_get_metric_names():
    """Test that metric names can be retrieved."""
    names = LengthAnalyzer.get_metric_names()
    assert "total_tokens" in names
    assert "avg_tokens_per_message" in names
    assert "message_token_counts" in names
    assert "num_messages" in names


def test_get_metric_descriptions():
    """Test that metric descriptions can be retrieved."""
    descriptions = LengthAnalyzer.get_metric_descriptions()
    assert "total_tokens" in descriptions
    assert len(descriptions["total_tokens"]) > 0


def test_get_scope():
    """Test that analyzer scope is conversation."""
    assert LengthAnalyzer.get_scope() == "conversation"
