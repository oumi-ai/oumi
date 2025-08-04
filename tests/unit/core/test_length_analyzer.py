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

from oumi.core.analyze.length_analyzer import LengthAnalyzer
from oumi.core.types.conversation import Conversation, Message, Role


def _single_message_conversation(text):
    return Conversation(messages=[Message(role=Role.USER, content=text)])


def test_char_count():
    """Test character count functionality."""
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=False, token_count=False
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.compute_metrics(conv)
    assert result.messages[0].analyzer_metrics["char_count"] == 13
    # Only char_count should be present
    assert len(result.messages[0].analyzer_metrics) == 1


def test_word_count():
    """Test word count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=True, sentence_count=False, token_count=False
    )
    conv = _single_message_conversation("Hello world! This is a test.")
    result = analyzer.compute_metrics(conv)
    assert result.messages[0].analyzer_metrics["word_count"] == 6
    # Only word_count should be present
    assert len(result.messages[0].analyzer_metrics) == 1


def test_sentence_count():
    """Test sentence count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=False, sentence_count=True, token_count=False
    )
    conv = _single_message_conversation("Hello world! This is a test. How are you?")
    result = analyzer.compute_metrics(conv)
    assert result.messages[0].analyzer_metrics["sentence_count"] == 3
    # Only sentence_count should be present
    assert len(result.messages[0].analyzer_metrics) == 1


def test_analyzer_instantiation():
    """Test analyzer can be instantiated with different parameter combinations."""
    # Test with defaults
    analyzer = LengthAnalyzer()
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.compute_metrics(conv)
    assert result.messages[0].analyzer_metrics["char_count"] == 13
    assert result.messages[0].analyzer_metrics["word_count"] == 2
    assert result.messages[0].analyzer_metrics["sentence_count"] == 1
    assert "token_count" not in result.messages[0].analyzer_metrics

    # Test with custom parameters
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=True, token_count=False
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.compute_metrics(conv)
    assert result.messages[0].analyzer_metrics["char_count"] == 13
    assert "word_count" not in result.messages[0].analyzer_metrics
    assert result.messages[0].analyzer_metrics["sentence_count"] == 1
    assert "token_count" not in result.messages[0].analyzer_metrics

    # Test with partial parameters (some defaults, some overridden)
    analyzer = LengthAnalyzer(char_count=False, word_count=True)
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.compute_metrics(conv)
    assert "char_count" not in result.messages[0].analyzer_metrics
    assert result.messages[0].analyzer_metrics["word_count"] == 2
    assert result.messages[0].analyzer_metrics["sentence_count"] == 1  # Default True
    assert "token_count" not in result.messages[0].analyzer_metrics  # Default False


def test_token_count():
    """Test token count functionality."""
    # Test token count with tokenizer (default: includes special tokens)
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [0, 1, 2, 3, 4, 5, 2]  # 7 tokens
    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer.compute_metrics(conv, tokenizer=mock_tokenizer)
    assert result.messages[0].analyzer_metrics["token_count"] == 7
    # compute_metrics calls tokenizer twice: once for message, once for conversation
    assert mock_tokenizer.encode.call_count == 2
    # Check that it was called with the message text
    mock_tokenizer.encode.assert_any_call("Hello, world!", add_special_tokens=True)

    # Test without special tokens (explicitly set to False)
    mock_tokenizer_no_special = Mock()
    mock_tokenizer_no_special.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
    analyzer_no_special = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer_no_special,
        include_special_tokens=False,
    )
    conv = _single_message_conversation("Hello, world!")
    result = analyzer_no_special.compute_metrics(
        conv, tokenizer=mock_tokenizer_no_special
    )
    assert result.messages[0].analyzer_metrics["token_count"] == 5
    # compute_metrics calls tokenizer twice: once for message, once for conversation
    assert mock_tokenizer_no_special.encode.call_count == 2
    # Check that it was called with the message text
    mock_tokenizer_no_special.encode.assert_any_call(
        "Hello, world!", add_special_tokens=False
    )
