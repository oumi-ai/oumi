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

"""Tests for the LengthAnalyzer."""

from unittest.mock import Mock

import pytest

from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.analysis_utils import conversation_to_dataframes


def _single_message_conversation(text):
    return Conversation(messages=[Message(role=Role.USER, content=text)])


def _count_analysis_columns(df):
    """Count the number of analysis columns in a DataFrame."""
    analysis_suffixes = [
        "_length_token_count",
    ]
    return len(
        [
            col
            for col in df.columns
            if any(col.endswith(suffix) for suffix in analysis_suffixes)
        ]
    )


def test_token_count():
    """Test token count functionality."""
    # Test token count with tokenizer only (default: includes special tokens)
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5, 2]  # 7 tokens

    analyzer = LengthAnalyzer(
        token_count=True,
        tokenizer=mock_tokenizer,
    )
    conv = _single_message_conversation("Hello, world!")
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)

    result_df, _ = analyzer.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )
    assert result_df.iloc[0]["text_content_length_token_count"] == 7
    # analyze calls tokenizer once per field
    assert mock_tokenizer.encode.call_count == 1
    # Check that it was called with the message text
    mock_tokenizer.encode.assert_any_call("Hello, world!", add_special_tokens=True)

    # Test without special tokens (explicitly set to False)
    mock_tokenizer_no_special = Mock()
    mock_tokenizer_no_special.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens

    analyzer_no_special = LengthAnalyzer(
        token_count=True,
        tokenizer=mock_tokenizer_no_special,
        include_special_tokens=False,
    )
    conv = _single_message_conversation("Hello, world!")
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
    result_df, _ = analyzer_no_special.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )
    assert result_df.iloc[0]["text_content_length_token_count"] == 5
    # Check that it was called without special tokens
    mock_tokenizer_no_special.encode.assert_any_call(
        "Hello, world!", add_special_tokens=False
    )

    # Test without tokenizer and without tiktoken_encoding (should raise ValueError)
    with pytest.raises(ValueError, match="Either tokenizer or tiktoken_encoding"):
        LengthAnalyzer(
            token_count=True,
            tokenizer=None,
            tiktoken_encoding=None,  # Disable tiktoken
        )

    # Test with tokenizer but token_count=False (should not call tokenizer)
    mock_tokenizer_unused = Mock()
    analyzer_unused = LengthAnalyzer(
        token_count=False,  # Token count disabled
        tokenizer=mock_tokenizer_unused,
    )
    conv = _single_message_conversation("Hello, world!")
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
    result_df, _ = analyzer_unused.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )
    # Should not call tokenizer since token_count=False
    mock_tokenizer_unused.encode.assert_not_called()


def test_conversation_level_token_count():
    """Test that conversation-level token count is computed correctly with tokenizer."""
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    # 6 tokens for each message; 10 tokens for conversation
    mock_tokenizer.encode.side_effect = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        list(range(10)),
    ]

    # Create analyzer without dataset
    analyzer = LengthAnalyzer(
        token_count=True,
        tokenizer=mock_tokenizer,
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I am doing well, thank you!"),
        ]
    )

    # Analyze the conversation
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
    result_df, _ = analyzer.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )

    # Check that field-level token count is computed for each message
    assert "text_content_length_token_count" in result_df.columns
    # Each message should have 6 tokens
    assert result_df.iloc[0]["text_content_length_token_count"] == 6
    assert result_df.iloc[1]["text_content_length_token_count"] == 6

    # Verify that encode was used for field-level token count
    # Two message encodes (one per row)
    assert mock_tokenizer.encode.call_count == 2


def test_conversation_level_token_count_without_dataset():
    """Test that conversation-level token count is computed without a dataset using
    tokenizer chat template directly."""
    mock_tokenizer = Mock()
    mock_tokenizer.apply_chat_template.return_value = "<prompt>"
    mock_tokenizer.encode.side_effect = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5],
        list(range(8)),
    ]

    # Create analyzer WITHOUT dataset
    analyzer = LengthAnalyzer(
        token_count=True,
        tokenizer=mock_tokenizer,
        # No dataset parameter
    )

    # Create a conversation with multiple messages
    conv = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, how are you?"),
            Message(role=Role.ASSISTANT, content="I am doing well, thank you!"),
        ]
    )

    # Analyze the conversation
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
    result_df, _ = analyzer.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )

    # Check that field-level token count is computed for each message
    assert result_df.iloc[0]["text_content_length_token_count"] == 6
    assert result_df.iloc[1]["text_content_length_token_count"] == 6
    # Two message encodes (one per row)
    assert mock_tokenizer.encode.call_count == 2


def test_tiktoken_token_count():
    """Test token count functionality with tiktoken."""
    pytest.importorskip("tiktoken")

    # Test default tiktoken encoding (o200k_base)
    analyzer = LengthAnalyzer(
        token_count=True,
        # Uses default tiktoken_encoding="o200k_base"
    )
    conv = _single_message_conversation("Hello, world!")
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)

    result_df = analyzer.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )
    # o200k_base encodes "Hello, world!" as tokens
    assert "text_content_length_token_count" in result_df.columns
    assert result_df.iloc[0]["text_content_length_token_count"] > 0

    # Test with explicit tiktoken encoding
    analyzer_cl100k = LengthAnalyzer(
        token_count=True,
        tiktoken_encoding="cl100k_base",
    )
    conv = _single_message_conversation("Hello, world!")
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)

    result_df = analyzer_cl100k.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )
    assert "text_content_length_token_count" in result_df.columns
    assert result_df.iloc[0]["text_content_length_token_count"] > 0


def test_tiktoken_defaults():
    """Test that defaults use tiktoken with o200k_base encoding."""
    pytest.importorskip("tiktoken")

    # Test default instantiation (should use tiktoken o200k_base)
    analyzer = LengthAnalyzer()
    conv = _single_message_conversation("Hello, world!")
    _, test_df = conversation_to_dataframes(conv, "test_conv", 0)

    result_df = analyzer.analyze_sample(
        test_df, schema={"text_content": {"content_type": "text"}}
    )

    # With new defaults: only token_count should be present
    assert "text_content_length_token_count" in result_df.columns
    assert result_df.iloc[0]["text_content_length_token_count"] > 0
