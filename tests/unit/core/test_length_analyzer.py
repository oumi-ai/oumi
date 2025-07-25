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


class TestLengthAnalyzer:
    """Test cases for LengthAnalyzer."""

    def test_length_analyzer_initialization(self):
        """Test LengthAnalyzer initialization with different configs."""
        # Test with default config
        analyzer = LengthAnalyzer({})
        assert analyzer.char_count is True
        assert analyzer.word_count is True
        assert analyzer.sentence_count is True
        assert analyzer.token_count is False

        # Test with custom config
        config = {
            "char_count": False,
            "word_count": True,
            "sentence_count": False,
            "token_count": True,
        }
        analyzer = LengthAnalyzer(config)
        assert analyzer.char_count is False
        assert analyzer.word_count is True
        assert analyzer.sentence_count is False
        assert analyzer.token_count is True

    def test_analyze_message_char_count(self):
        """Test character count analysis."""
        analyzer = LengthAnalyzer(
            {"char_count": True, "word_count": False, "sentence_count": False}
        )
        text = "Hello, world!"
        result = analyzer.analyze_message(text)
        assert result["char_count"] == 13

    def test_analyze_message_word_count(self):
        """Test word count analysis."""
        analyzer = LengthAnalyzer(
            {"char_count": False, "word_count": True, "sentence_count": False}
        )
        text = "Hello world! This is a test."
        result = analyzer.analyze_message(text)
        assert result["word_count"] == 6

    def test_analyze_message_sentence_count(self):
        """Test sentence count analysis."""
        analyzer = LengthAnalyzer(
            {"char_count": False, "word_count": False, "sentence_count": True}
        )
        text = "Hello world! This is a test. How are you?"
        result = analyzer.analyze_message(text)
        assert result["sentence_count"] == 3

    def test_analyze_message_all_metrics(self):
        """Test all metrics together."""
        analyzer = LengthAnalyzer(
            {
                "char_count": True,
                "word_count": True,
                "sentence_count": True,
                "token_count": True,
            }
        )
        text = "Hello world! This is a test. How are you?"
        result = analyzer.analyze_message(text)

        assert result["char_count"] == 39
        assert result["word_count"] == 8
        assert result["sentence_count"] == 3
        assert result["token_count"] == 8  # Falls back to word count

    def test_analyze_message_empty_text(self):
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

    def test_analyze_message_whitespace_only(self):
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
