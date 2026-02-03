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
    default_tokenizer,
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
    """Get the default tiktoken tokenizer."""
    return default_tokenizer()


# -----------------------------------------------------------------------------
# Tokenizer Protocol Tests
# -----------------------------------------------------------------------------


def test_tokenizer_protocol(mock_tokenizer):
    """Test that mock tokenizer satisfies the Tokenizer protocol."""
    assert isinstance(mock_tokenizer, Tokenizer)


def test_default_tokenizer():
    """Test that default_tokenizer returns a valid tokenizer."""
    tokenizer = default_tokenizer()
    assert hasattr(tokenizer, "encode")
    tokens = tokenizer.encode("Hello, world!")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_default_tokenizer_custom_encoding():
    """Test default_tokenizer with custom encoding."""
    tokenizer = default_tokenizer("p50k_base")
    tokens = tokenizer.encode("Hello")
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
    # Role stats default to 0
    assert metrics.user_total_tokens == 0
    assert metrics.assistant_total_tokens == 0
    assert metrics.system_total_tokens == 0
    assert metrics.tool_total_tokens == 0


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
    """Test that role stats are always computed."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze(simple_conversation)

    assert result.user_total_tokens > 0
    assert result.assistant_total_tokens > 0
    # System and tool should be 0 since there are no such messages
    assert result.system_total_tokens == 0
    assert result.tool_total_tokens == 0


def test_analyze_conversation_with_system(conversation_with_system, tiktoken_tokenizer):
    """Test analyzing a conversation with a system message."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze(conversation_with_system)

    assert result.num_messages == 3
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


def test_analyze_raises_without_tokenizer(simple_conversation):
    """Test that analyze raises RuntimeError without a tokenizer."""
    analyzer = LengthAnalyzer()  # No tokenizer

    with pytest.raises(RuntimeError, match="No tokenizer configured"):
        analyzer.analyze(simple_conversation)


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


def test_analyze_text_role_stats_are_zero(tiktoken_tokenizer):
    """Test that analyze_text returns zero for role stats (no conversation context)."""
    analyzer = LengthAnalyzer(tokenizer=tiktoken_tokenizer)
    result = analyzer.analyze_text("Some text")

    # No conversation context, so role stats default to 0
    assert result.user_total_tokens == 0
    assert result.assistant_total_tokens == 0
    assert result.system_total_tokens == 0
    assert result.tool_total_tokens == 0


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


def test_count_tokens_raises_when_no_tokenizer():
    """Test that RuntimeError is raised when no tokenizer is configured."""
    analyzer = LengthAnalyzer()

    with pytest.raises(RuntimeError, match="No tokenizer configured"):
        analyzer._count_tokens("test text")


# -----------------------------------------------------------------------------
# Pipeline Tokenizer Injection Tests
# -----------------------------------------------------------------------------


def test_pipeline_injects_tokenizer(simple_conversation):
    """Test that AnalysisPipeline injects tokenizer into analyzers."""
    from oumi.analyze.pipeline import AnalysisPipeline

    analyzer = LengthAnalyzer()  # No tokenizer
    assert analyzer.tokenizer is None

    pipeline = AnalysisPipeline(analyzers=[analyzer])

    # Pipeline should have injected the tokenizer
    assert analyzer.tokenizer is not None

    # Should now work
    results = pipeline.run([simple_conversation])
    assert "LengthAnalyzer" in results


def test_pipeline_respects_custom_tokenizer(simple_conversation, mock_tokenizer):
    """Test that AnalysisPipeline doesn't override existing tokenizers."""
    from oumi.analyze.pipeline import AnalysisPipeline

    analyzer = LengthAnalyzer(tokenizer=mock_tokenizer)

    pipeline = AnalysisPipeline(analyzers=[analyzer])

    # Should keep the custom tokenizer
    assert analyzer.tokenizer is mock_tokenizer

    results = pipeline.run([simple_conversation])
    # Mock tokenizer: "Hello" -> 1, "Hi there!" -> 2
    length_results = results["LengthAnalyzer"]
    assert isinstance(length_results, list)
    assert isinstance(length_results[0], LengthMetrics)
    assert length_results[0].total_tokens == 3


def test_pipeline_custom_tokenizer_for_all(simple_conversation, mock_tokenizer):
    """Test that pipeline can provide a custom tokenizer for all analyzers."""
    from oumi.analyze.pipeline import AnalysisPipeline

    analyzer = LengthAnalyzer()  # No tokenizer

    _ = AnalysisPipeline(
        analyzers=[analyzer],
        tokenizer=mock_tokenizer,
    )

    # Should use the pipeline's tokenizer
    assert analyzer.tokenizer is mock_tokenizer


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------


def test_analyzer_registered_in_registry():
    """Test that LengthAnalyzer is registered in the core registry."""
    from oumi.core.registry import REGISTRY, RegistryType

    analyzer_class = REGISTRY.get(
        name="typed_length", type=RegistryType.SAMPLE_ANALYZER
    )
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
