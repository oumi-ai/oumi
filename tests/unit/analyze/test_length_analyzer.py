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

from oumi.analyze.analyzers.length import (
    LengthAnalyzer,
    LengthMetrics,
    Tokenizer,
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


@pytest.fixture
def tiktoken_tokenizer():
    """Get a tiktoken tokenizer via from_config."""
    return LengthAnalyzer.from_config({"tokenizer_name": "cl100k_base"}).tokenizer


# -----------------------------------------------------------------------------
# Tokenizer Protocol Tests
# -----------------------------------------------------------------------------


def test_tokenizer_protocol(mock_tokenizer):
    """Test that mock tokenizer satisfies the Tokenizer protocol."""
    assert isinstance(mock_tokenizer, Tokenizer)


def test_default_tokenizer():
    """Test that from_config with tiktoken encoding returns a valid tokenizer."""
    analyzer = LengthAnalyzer.from_config({"tokenizer_name": "cl100k_base"})
    assert analyzer.tokenizer is not None
    tokens = analyzer.tokenizer.encode("Hello, world!")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_default_tokenizer_custom_encoding():
    """Test from_config with a non-default tiktoken encoding."""
    analyzer = LengthAnalyzer.from_config({"tokenizer_name": "p50k_base"})
    assert analyzer.tokenizer is not None
    tokens = analyzer.tokenizer.encode("Hello")
    assert len(tokens) > 0


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
    # Role stats default to None when not provided
    assert metrics.user_total_tokens is None
    assert metrics.assistant_total_tokens is None
    assert metrics.system_total_tokens is None


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
    """Test LengthAnalyzer initializes with no tokenizer by default."""
    analyzer = LengthAnalyzer()
    assert analyzer.tokenizer is None


def test_analyzer_with_tokenizer(tiktoken_tokenizer):
    """Test LengthAnalyzer with a tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    assert analyzer.tokenizer is tiktoken_tokenizer


def test_analyzer_with_custom_tokenizer(mock_tokenizer):
    """Test LengthAnalyzer with a custom tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)
    assert analyzer.tokenizer is mock_tokenizer


# -----------------------------------------------------------------------------
# LengthAnalyzer.analyze() Tests
# -----------------------------------------------------------------------------


def test_analyze_simple_conversation(simple_conversation, tiktoken_tokenizer):
    """Test analyzing a simple conversation."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze(simple_conversation)

    assert isinstance(result, LengthMetrics)
    assert result.num_messages == 2
    assert len(result.message_token_counts) == 2
    assert result.total_tokens == sum(result.message_token_counts)
    assert result.avg_tokens_per_message == result.total_tokens / 2


def test_analyze_role_stats(simple_conversation, tiktoken_tokenizer):
    """Test that role stats are computed when compute_role_stats is enabled."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze(simple_conversation)

    assert result.user_total_tokens is not None
    assert result.user_total_tokens > 0
    assert result.assistant_total_tokens is not None
    assert result.assistant_total_tokens > 0
    # System should be 0 since there are no system messages
    assert result.system_total_tokens == 0


def test_analyze_conversation_with_system(conversation_with_system, tiktoken_tokenizer):
    """Test analyzing a conversation with a system message."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze(conversation_with_system)

    assert result.num_messages == 3
    assert result.system_total_tokens is not None
    assert result.system_total_tokens > 0


def test_analyze_empty_conversation(empty_conversation, tiktoken_tokenizer):
    """Test analyzing an empty conversation."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
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


def test_analyze_without_explicit_tokenizer_uses_tiktoken_fallback(
    simple_conversation,
):
    """Test that analyze falls back to tiktoken when no tokenizer is provided."""
    analyzer = LengthAnalyzer()  # No explicit tokenizer, uses tiktoken fallback
    result = analyzer.analyze(simple_conversation)
    # Should still count tokens via tiktoken fallback
    assert result.total_tokens > 0


# -----------------------------------------------------------------------------
# LengthAnalyzer.analyze_text() Tests
# -----------------------------------------------------------------------------


def test_analyze_text_simple(tiktoken_tokenizer):
    """Test analyzing a simple text string."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
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


def test_analyze_text_empty(tiktoken_tokenizer):
    """Test analyzing empty text."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze_text("")

    assert result.total_tokens == 0
    assert result.num_messages == 1


def test_analyze_text_role_stats_are_none(tiktoken_tokenizer):
    """Test that analyze_text returns None for role stats (no conversation context)."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze_text("Some text")

    # No conversation context, so role stats are None
    assert result.user_total_tokens is None
    assert result.assistant_total_tokens is None
    assert result.system_total_tokens is None


# -----------------------------------------------------------------------------
# Token Counting Tests
# -----------------------------------------------------------------------------


def test_count_tokens_with_tiktoken(tiktoken_tokenizer):
    """Test token counting with tiktoken encoder."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    count = analyzer._count_tokens("Hello, world!")
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_with_custom_tokenizer(mock_tokenizer):
    """Test token counting with custom tokenizer."""
    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)
    count = analyzer._count_tokens("one two three")
    assert count == 3


def test_count_tokens_empty_string(tiktoken_tokenizer):
    """Test token counting with empty string."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    count = analyzer._count_tokens("")
    assert count == 0


def test_count_tokens_with_tiktoken_fallback():
    """Test that _count_tokens uses tiktoken fallback when no tokenizer given."""
    analyzer = LengthAnalyzer()
    # Falls back to tiktoken, so should count tokens
    assert analyzer._count_tokens("test text") > 0


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_cli_registry():
    """Test that LengthAnalyzer is registered in the CLI analyzer registry."""
    from oumi.analyze.cli import ANALYZER_REGISTRY

    assert "length" in ANALYZER_REGISTRY
    assert ANALYZER_REGISTRY["length"] is LengthAnalyzer
    assert "LengthAnalyzer" in ANALYZER_REGISTRY
    assert ANALYZER_REGISTRY["LengthAnalyzer"] is LengthAnalyzer


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


def test_analyzer_is_conversation_analyzer():
    """Test that LengthAnalyzer is a ConversationAnalyzer."""
    from oumi.analyze.base import ConversationAnalyzer

    assert issubclass(LengthAnalyzer, ConversationAnalyzer)


def test_from_config_tiktoken():
    """Test creating analyzer from config with tiktoken."""
    analyzer = LengthAnalyzer.from_config({"tokenizer_name": "cl100k_base"})
    assert analyzer.tokenizer is not None

    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello world"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.total_tokens > 0


def test_from_config_huggingface():
    """Test creating analyzer from config with HuggingFace tokenizer.

    Uses openai-community/gpt2 (the HF model ID) to exercise the HuggingFace
    path in from_config(). "gpt2" alone is intentionally not in TIKTOKEN_ENCODINGS
    since it is also a valid HF model ID.
    """
    from transformers import PreTrainedTokenizerBase

    analyzer = LengthAnalyzer.from_config({"tokenizer_name": "openai-community/gpt2"})
    assert analyzer.tokenizer is not None
    # Verify we got a HuggingFace tokenizer, not a tiktoken encoding
    assert isinstance(analyzer.tokenizer, PreTrainedTokenizerBase)

    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello world"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.total_tokens > 0


def test_from_config_default():
    """Test from_config with default tokenizer."""
    analyzer = LengthAnalyzer.from_config({})
    assert analyzer.tokenizer is not None

    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Test"),
        ]
    )
    result = analyzer.analyze(conversation)
    assert result.total_tokens > 0


def test_create_analyzer_from_config_uses_from_config():
    """Test that create_analyzer_from_config uses from_config method."""
    from oumi.analyze.cli import create_analyzer_from_config

    analyzer = create_analyzer_from_config("length", {"tokenizer_name": "cl100k_base"})
    assert analyzer is not None
    assert isinstance(analyzer, LengthAnalyzer)
    assert analyzer.tokenizer is not None
