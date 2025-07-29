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

from oumi.core.analyze.length_analyzer import LengthAnalyzer


def test_analyze_message_char_count():
    """Test character count analysis."""
    analyzer = LengthAnalyzer(
        {"char_count": True, "word_count": False, "sentence_count": False}
    )
    text = "Hello, world!"
    result = analyzer.analyze_message(text)
    assert result["char_count"] == 13


def test_analyze_message_word_count():
    """Test word count analysis."""
    analyzer = LengthAnalyzer(
        {"char_count": False, "word_count": True, "sentence_count": False}
    )
    text = "Hello world! This is a test."
    result = analyzer.analyze_message(text)
    assert result["word_count"] == 6


def test_analyze_message_sentence_count():
    """Test sentence count analysis."""
    analyzer = LengthAnalyzer(
        {"char_count": False, "word_count": False, "sentence_count": True}
    )
    text = "Hello world! This is a test. How are you?"
    result = analyzer.analyze_message(text)
    assert result["sentence_count"] == 3


def test_analyze_message_empty_text():
    """Test analysis with empty text."""
    analyzer = LengthAnalyzer(
        {
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
        }
    )
    result = analyzer.analyze_message("")

    assert result["char_count"] == 0
    assert result["word_count"] == 0
    assert result["sentence_count"] == 0


def test_analyze_message_whitespace_only():
    """Test analysis with whitespace-only text."""
    analyzer = LengthAnalyzer(
        {
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
        }
    )
    result = analyzer.analyze_message("   \n\t   ")

    assert result["char_count"] == 7
    assert result["word_count"] == 0
    assert result["sentence_count"] == 0
