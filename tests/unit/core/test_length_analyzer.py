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


def test_char_count():
    """Test character count functionality."""
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=False, token_count=False
    )
    result = analyzer.analyze_message("Hello, world!")
    assert result["char_count"] == 13
    assert len(result) == 1  # Only char_count should be present


def test_word_count():
    """Test word count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=True, sentence_count=False, token_count=False
    )
    result = analyzer.analyze_message("Hello world! This is a test.")
    assert result["word_count"] == 6
    assert len(result) == 1  # Only word_count should be present


def test_sentence_count():
    """Test sentence count functionality."""
    analyzer = LengthAnalyzer(
        char_count=False, word_count=False, sentence_count=True, token_count=False
    )
    result = analyzer.analyze_message("Hello world! This is a test. How are you?")
    assert result["sentence_count"] == 3
    assert len(result) == 1  # Only sentence_count should be present


def test_analyzer_instantiation():
    """Test analyzer can be instantiated with different parameter combinations."""
    # Test with defaults
    analyzer = LengthAnalyzer()
    result = analyzer.analyze_message("Hello, world!")
    assert result["char_count"] == 13
    assert result["word_count"] == 2
    assert result["sentence_count"] == 1
    assert "token_count" not in result

    # Test with custom parameters
    analyzer = LengthAnalyzer(
        char_count=True, word_count=False, sentence_count=True, token_count=False
    )
    result = analyzer.analyze_message("Hello, world!")
    assert result["char_count"] == 13
    assert "word_count" not in result
    assert result["sentence_count"] == 1
    assert "token_count" not in result

    # Test with partial parameters (some defaults, some overridden)
    analyzer = LengthAnalyzer(char_count=False, word_count=True)
    result = analyzer.analyze_message("Hello, world!")
    assert "char_count" not in result
    assert result["word_count"] == 2
    assert result["sentence_count"] == 1  # Default True
    assert "token_count" not in result  # Default False


def test_token_count():
    """Test token count functionality."""

    # Test token count with tokenizer (default: includes special tokens)
    mock_tokenizer = Mock()
    mock_tokenizer.encode.return_value = [
        0,
        1,
        2,
        3,
        4,
        5,
        2,
    ]  # 7 tokens with special tokens

    analyzer = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer,
    )
    result = analyzer.analyze_message("Hello, world!", tokenizer=mock_tokenizer)

    assert result["token_count"] == 7
    mock_tokenizer.encode.assert_called_once_with(
        "Hello, world!", add_special_tokens=True
    )

    # Test without special tokens (explicitly set to False)
    mock_tokenizer_no_special = Mock()
    mock_tokenizer_no_special.encode.return_value = [
        1,
        2,
        3,
        4,
        5,
    ]  # 5 tokens without special tokens

    analyzer_no_special = LengthAnalyzer(
        char_count=False,
        word_count=False,
        sentence_count=False,
        token_count=True,
        tokenizer=mock_tokenizer_no_special,
        include_special_tokens=False,
    )
    result = analyzer_no_special.analyze_message(
        "Hello, world!", tokenizer=mock_tokenizer_no_special
    )

    assert result["token_count"] == 5
    mock_tokenizer_no_special.encode.assert_called_once_with(
        "Hello, world!", add_special_tokens=False
    )

    # Test without tokenizer (should fail during initialization)
    result = analyzer.analyze_message("Hello, world!")
    assert result["token_count"] == 7  # uses instance tokenizer with default (True)
