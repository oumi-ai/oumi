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

"""Tests for the DiversityAnalyzer."""

import math

import pytest

from oumi.core.analyze.diversity_analyzer import DiversityAnalyzer
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.analysis_utils import conversation_to_dataframes


def _single_message_conversation(text):
    return Conversation(messages=[Message(role=Role.USER, content=text)])


def _count_analysis_columns(df, analyzer_id="diversity"):
    """Count the number of analysis columns in a DataFrame."""
    analysis_suffixes = [
        f"_{analyzer_id}_unique_words_ratio",
        f"_{analyzer_id}_type_token_ratio",
        f"_{analyzer_id}_vocabulary_richness",
        f"_{analyzer_id}_hapax_legomena_ratio",
    ]
    return len(
        [
            col
            for col in df.columns
            if any(col.endswith(suffix) for suffix in analysis_suffixes)
        ]
    )


class TestDiversityAnalyzerBasicMetrics:
    """Tests for basic diversity metric calculations."""

    def test_unique_words_ratio_high_diversity(self):
        """Test unique words ratio with all unique words."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=True,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
        )
        # 5 unique words out of 5 total = 1.0
        conv = _single_message_conversation("one two three four five")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 1.0
        assert _count_analysis_columns(result_df) == 1

    def test_unique_words_ratio_low_diversity(self):
        """Test unique words ratio with repeated words."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=True,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
        )
        # 1 unique word out of 5 total = 0.2
        conv = _single_message_conversation("the the the the the")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 0.2
        assert _count_analysis_columns(result_df) == 1

    def test_type_token_ratio(self):
        """Test type-token ratio calculation."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=False,
            type_token_ratio=True,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
        )
        # 3 unique words out of 6 total = 0.5
        conv = _single_message_conversation("cat dog cat bird dog cat")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_type_token_ratio"] == 0.5
        assert _count_analysis_columns(result_df) == 1

    def test_vocabulary_richness(self):
        """Test vocabulary richness (Root TTR) calculation."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=False,
            type_token_ratio=False,
            vocabulary_richness=True,
            hapax_legomena_ratio=False,
        )
        # 4 unique words out of 4 total -> 4 / sqrt(4) = 4 / 2 = 2.0
        conv = _single_message_conversation("one two three four")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_vocabulary_richness"] == 2.0
        assert _count_analysis_columns(result_df) == 1

    def test_vocabulary_richness_longer_text(self):
        """Test vocabulary richness with longer text to show length normalization."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=False,
            type_token_ratio=False,
            vocabulary_richness=True,
            hapax_legomena_ratio=False,
        )
        # 9 unique words out of 9 total -> 9 / sqrt(9) = 9 / 3 = 3.0
        conv = _single_message_conversation("one two three four five six seven eight nine")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_vocabulary_richness"] == 3.0

    def test_hapax_legomena_ratio(self):
        """Test hapax legomena ratio calculation."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=False,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=True,
        )
        # Words: cat(3), dog(2), bird(1) -> 3 unique, 1 hapax -> 1/3 = 0.333...
        conv = _single_message_conversation("cat dog cat bird dog cat")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        expected = 1 / 3
        assert (
            abs(
                result_df.iloc[0]["text_content_diversity_hapax_legomena_ratio"]
                - expected
            )
            < 0.0001
        )
        assert _count_analysis_columns(result_df) == 1

    def test_hapax_legomena_all_unique(self):
        """Test hapax legomena ratio when all words appear once."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=False,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=True,
        )
        # All 5 words are unique and appear once -> 5/5 = 1.0
        conv = _single_message_conversation("one two three four five")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_hapax_legomena_ratio"] == 1.0


class TestDiversityAnalyzerEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_text(self):
        """Test handling of empty text."""
        analyzer = DiversityAnalyzer()
        conv = _single_message_conversation("")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # All metrics should be 0 for empty text
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 0.0
        assert result_df.iloc[0]["text_content_diversity_type_token_ratio"] == 0.0
        assert result_df.iloc[0]["text_content_diversity_vocabulary_richness"] == 0.0

    def test_single_word(self):
        """Test handling of single word text."""
        analyzer = DiversityAnalyzer()
        conv = _single_message_conversation("hello")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # 1 unique word out of 1 total = 1.0
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 1.0
        assert result_df.iloc[0]["text_content_diversity_type_token_ratio"] == 1.0
        # 1 / sqrt(1) = 1.0
        assert result_df.iloc[0]["text_content_diversity_vocabulary_richness"] == 1.0

    def test_whitespace_only(self):
        """Test handling of whitespace-only text."""
        analyzer = DiversityAnalyzer()
        conv = _single_message_conversation("   \t\n   ")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # Should be treated as empty
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 0.0


class TestDiversityAnalyzerCaseSensitivity:
    """Tests for case sensitivity option."""

    def test_case_insensitive_default(self):
        """Test that case-insensitive is the default."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=True,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
        )
        # "Hello" and "hello" should be treated as the same word
        # 2 unique words out of 4 total = 0.5
        conv = _single_message_conversation("Hello hello World world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 0.5

    def test_case_sensitive(self):
        """Test case-sensitive mode."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=True,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
            case_sensitive=True,
        )
        # "Hello" and "hello" should be treated as different words
        # 4 unique words out of 4 total = 1.0
        conv = _single_message_conversation("Hello hello World world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert result_df.iloc[0]["text_content_diversity_unique_words_ratio"] == 1.0


class TestDiversityAnalyzerInstantiation:
    """Tests for analyzer instantiation and configuration."""

    def test_default_instantiation(self):
        """Test analyzer with default parameters."""
        analyzer = DiversityAnalyzer()
        conv = _single_message_conversation("hello world hello")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # Should have unique_words_ratio, type_token_ratio, vocabulary_richness
        # but not hapax_legomena_ratio (default False)
        assert "text_content_diversity_unique_words_ratio" in result_df.columns
        assert "text_content_diversity_type_token_ratio" in result_df.columns
        assert "text_content_diversity_vocabulary_richness" in result_df.columns
        assert "text_content_diversity_hapax_legomena_ratio" not in result_df.columns

    def test_all_metrics_enabled(self):
        """Test analyzer with all metrics enabled."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=True,
            type_token_ratio=True,
            vocabulary_richness=True,
            hapax_legomena_ratio=True,
        )
        conv = _single_message_conversation("one two three")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert _count_analysis_columns(result_df) == 4

    def test_no_metrics_enabled(self):
        """Test analyzer with no metrics enabled returns unchanged DataFrame."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=False,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
        )
        conv = _single_message_conversation("hello world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        assert _count_analysis_columns(result_df) == 0


class TestDiversityAnalyzerValidation:
    """Tests for input validation."""

    def test_missing_schema_raises_error(self):
        """Test that missing schema raises ValueError."""
        analyzer = DiversityAnalyzer()
        conv = _single_message_conversation("hello world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(test_df, schema=None)

    def test_no_text_columns_returns_unchanged(self):
        """Test that schema with no text columns returns DataFrame unchanged."""
        analyzer = DiversityAnalyzer()
        conv = _single_message_conversation("hello world")
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "numeric"}}
        )
        # DataFrame should be returned unchanged (no new columns added)
        assert list(result_df.columns) == list(test_df.columns)


class TestDiversityAnalyzerMultipleMessages:
    """Tests for multi-message conversations."""

    def test_multiple_messages(self):
        """Test diversity analysis across multiple messages."""
        analyzer = DiversityAnalyzer(
            unique_words_ratio=True,
            type_token_ratio=False,
            vocabulary_richness=False,
            hapax_legomena_ratio=False,
        )
        conv = Conversation(
            messages=[
                Message(role=Role.USER, content="hello world hello"),  # 2/3 unique
                Message(
                    role=Role.ASSISTANT, content="goodbye world"
                ),  # 2/2 = 1.0 unique
            ]
        )
        _, test_df = conversation_to_dataframes(conv, "test_conv", 0)
        result_df = analyzer.analyze_sample(
            test_df, schema={"text_content": {"content_type": "text"}}
        )
        # Check both messages
        expected_first = 2 / 3
        assert (
            abs(
                result_df.iloc[0]["text_content_diversity_unique_words_ratio"]
                - expected_first
            )
            < 0.0001
        )
        assert result_df.iloc[1]["text_content_diversity_unique_words_ratio"] == 1.0
