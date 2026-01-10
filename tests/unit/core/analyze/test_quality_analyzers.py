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

"""Tests for Phase 1 and Phase 2 quality analyzers."""

import pandas as pd
import pytest

from oumi.core.analyze.duplicate_analyzer import DuplicateAnalyzer
from oumi.core.analyze.empty_content_analyzer import EmptyContentAnalyzer
from oumi.core.analyze.encoding_analyzer import EncodingAnalyzer
from oumi.core.analyze.format_validator import FormatValidationAnalyzer
from oumi.core.analyze.ngram_analyzer import NgramAnalyzer
from oumi.core.analyze.repetition_analyzer import RepetitionAnalyzer
from oumi.core.analyze.role_sequence_analyzer import RoleSequenceAnalyzer
from oumi.core.analyze.statistical_analyzer import StatisticalOutlierAnalyzer
from oumi.core.analyze.vocabulary_analyzer import VocabularyAnalyzer

TEXT_SCHEMA = {"text_content": {"content_type": "text"}}


# ============================================================================
# DuplicateAnalyzer Tests
# ============================================================================


class TestDuplicateAnalyzer:
    """Tests for DuplicateAnalyzer."""

    def test_detects_exact_duplicates(self):
        """Test detection of exact duplicate content."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "Hello world",
                    "Hello world",
                    "Different text",
                    "Hello world",
                ]
            }
        )
        analyzer = DuplicateAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First three "Hello world" entries should be flagged as duplicates
        assert result.iloc[0]["text_content_is_duplicate"] is True
        assert result.iloc[1]["text_content_is_duplicate"] is True
        assert result.iloc[2]["text_content_is_duplicate"] is False
        assert result.iloc[3]["text_content_is_duplicate"] is True
        assert result.iloc[0]["text_content_duplicate_count"] == 3

    def test_case_insensitive_by_default(self):
        """Test that duplicate detection is case-insensitive by default."""
        df = pd.DataFrame({"text_content": ["Hello World", "hello world", "HELLO WORLD"]})
        analyzer = DuplicateAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_is_duplicate"] is True
        assert result.iloc[0]["text_content_duplicate_count"] == 3

    def test_case_sensitive_option(self):
        """Test case-sensitive duplicate detection."""
        df = pd.DataFrame({"text_content": ["Hello World", "hello world"]})
        analyzer = DuplicateAnalyzer(case_sensitive=True)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_is_duplicate"] is False
        assert result.iloc[1]["text_content_is_duplicate"] is False

    def test_whitespace_normalization(self):
        """Test whitespace normalization in duplicate detection."""
        df = pd.DataFrame({"text_content": ["Hello  world", "Hello world", "Hello   world"]})
        analyzer = DuplicateAnalyzer(normalize_whitespace=True)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_is_duplicate"] is True
        assert result.iloc[0]["text_content_duplicate_count"] == 3


# ============================================================================
# EmptyContentAnalyzer Tests
# ============================================================================


class TestEmptyContentAnalyzer:
    """Tests for EmptyContentAnalyzer."""

    def test_detects_empty_string(self):
        """Test detection of empty strings."""
        df = pd.DataFrame({"text_content": ["", "Hello", ""]})
        analyzer = EmptyContentAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_is_empty"] == True
        assert result.iloc[1]["text_content_is_empty"] == False
        assert result.iloc[2]["text_content_is_empty"] == True

    def test_detects_whitespace_only(self):
        """Test detection of whitespace-only content."""
        df = pd.DataFrame({"text_content": ["   ", "\t\n", "Hello", "  x  "]})
        analyzer = EmptyContentAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_is_whitespace_only"] == True
        assert result.iloc[1]["text_content_is_whitespace_only"] == True
        assert result.iloc[2]["text_content_is_whitespace_only"] == False
        assert result.iloc[3]["text_content_is_whitespace_only"] == False

    def test_has_content_flag(self):
        """Test has_content flag."""
        df = pd.DataFrame({"text_content": ["", "   ", "Hi"]})
        analyzer = EmptyContentAnalyzer(min_content_length=1)
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_has_content"] == False
        assert result.iloc[1]["text_content_has_content"] == False
        assert result.iloc[2]["text_content_has_content"] == True

    def test_detects_error_tokens(self):
        """Test detection of error tokens."""
        df = pd.DataFrame({"text_content": ["Hello", "nan", "<noinput>", "World", "nan"]})
        analyzer = EmptyContentAnalyzer(error_tokens=["nan", "<noinput>"])
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Check error token detection
        assert result.iloc[0]["text_content_contains_error_token"] == False
        assert result.iloc[1]["text_content_contains_error_token"] == True
        assert result.iloc[2]["text_content_contains_error_token"] == True
        assert result.iloc[3]["text_content_contains_error_token"] == False
        assert result.iloc[4]["text_content_contains_error_token"] == True

        # Check error token types
        assert pd.isna(result.iloc[0]["text_content_error_token_type"])
        assert result.iloc[1]["text_content_error_token_type"] == "nan"
        assert result.iloc[2]["text_content_error_token_type"] == "<noinput>"
        assert result.iloc[4]["text_content_error_token_type"] == "nan"

    def test_detects_placeholder_tokens(self):
        """Test detection of placeholder tokens."""
        df = pd.DataFrame({"text_content": ["Hello", "<nooutput>", "World", "<nooutput>"]})
        analyzer = EmptyContentAnalyzer(placeholder_tokens=["<nooutput>"])
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Check placeholder token detection
        assert result.iloc[0]["text_content_contains_placeholder_token"] == False
        assert result.iloc[1]["text_content_contains_placeholder_token"] == True
        assert result.iloc[2]["text_content_contains_placeholder_token"] == False
        assert result.iloc[3]["text_content_contains_placeholder_token"] == True

        # Check placeholder token types
        assert pd.isna(result.iloc[0]["text_content_placeholder_token_type"])
        assert result.iloc[1]["text_content_placeholder_token_type"] == "<nooutput>"
        assert result.iloc[3]["text_content_placeholder_token_type"] == "<nooutput>"

    def test_detects_both_token_types(self):
        """Test detection of both error and placeholder tokens."""
        df = pd.DataFrame({
            "text_content": ["Hello", "nan", "<nooutput>", "<noinput>", "World"]
        })
        analyzer = EmptyContentAnalyzer(
            error_tokens=["nan", "<noinput>"],
            placeholder_tokens=["<nooutput>"]
        )
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Row 0: normal content
        assert result.iloc[0]["text_content_contains_error_token"] == False
        assert result.iloc[0]["text_content_contains_placeholder_token"] == False

        # Row 1: error token "nan"
        assert result.iloc[1]["text_content_contains_error_token"] == True
        assert result.iloc[1]["text_content_error_token_type"] == "nan"
        assert result.iloc[1]["text_content_contains_placeholder_token"] == False

        # Row 2: placeholder token "<nooutput>"
        assert result.iloc[2]["text_content_contains_error_token"] == False
        assert result.iloc[2]["text_content_contains_placeholder_token"] == True
        assert result.iloc[2]["text_content_placeholder_token_type"] == "<nooutput>"

        # Row 3: error token "<noinput>"
        assert result.iloc[3]["text_content_contains_error_token"] == True
        assert result.iloc[3]["text_content_error_token_type"] == "<noinput>"
        assert result.iloc[3]["text_content_contains_placeholder_token"] == False

    def test_no_tokens_configured(self):
        """Test that analyzer works when no special tokens are configured."""
        df = pd.DataFrame({"text_content": ["Hello", "nan", "<nooutput>"]})
        analyzer = EmptyContentAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Should not flag any tokens
        assert result["text_content_contains_error_token"].sum() == 0
        assert result["text_content_contains_placeholder_token"].sum() == 0


# ============================================================================
# EncodingAnalyzer Tests
# ============================================================================


class TestEncodingAnalyzer:
    """Tests for EncodingAnalyzer."""

    def test_detects_replacement_character(self):
        """Test detection of Unicode replacement characters."""
        df = pd.DataFrame({"text_content": ["Hello \ufffd world", "Normal text"]})
        analyzer = EncodingAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_has_replacement_chars"] is True
        assert result.iloc[0]["text_content_replacement_char_count"] == 1
        assert result.iloc[1]["text_content_has_replacement_chars"] is False

    def test_detects_control_characters(self):
        """Test detection of problematic control characters."""
        df = pd.DataFrame({"text_content": ["Hello\x00world", "Normal\ttext\n"]})
        analyzer = EncodingAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Null character should be flagged
        assert result.iloc[0]["text_content_control_char_count"] == 1
        # Tab and newline are allowed
        assert result.iloc[1]["text_content_control_char_count"] == 0

    def test_encoding_issues_flag(self):
        """Test overall encoding issues flag."""
        df = pd.DataFrame({"text_content": ["Normal", "\ufffd", "\x00"]})
        analyzer = EncodingAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_has_encoding_issues"] is False
        assert result.iloc[1]["text_content_has_encoding_issues"] is True
        assert result.iloc[2]["text_content_has_encoding_issues"] is True


# ============================================================================
# FormatValidationAnalyzer Tests
# ============================================================================


class TestFormatValidationAnalyzer:
    """Tests for FormatValidationAnalyzer."""

    def test_detects_empty_required_fields(self):
        """Test detection of empty required fields."""
        df = pd.DataFrame({"text_content": ["Hello", "", "   "], "other": [1, 2, 3]})
        analyzer = FormatValidationAnalyzer(non_empty_columns=["text_content"])
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["format_is_valid"] is True
        assert result.iloc[1]["format_is_valid"] is False
        assert result.iloc[2]["format_is_valid"] is False
        assert "text_content" in result.iloc[1]["format_empty_fields"]


# ============================================================================
# RoleSequenceAnalyzer Tests
# ============================================================================


class TestRoleSequenceAnalyzer:
    """Tests for RoleSequenceAnalyzer."""

    def test_detects_valid_sequence(self):
        """Test validation of correct role sequences."""
        df = pd.DataFrame(
            {
                "role": ["user", "assistant", "user", "assistant"],
                "conversation_id": ["conv1"] * 4,
            }
        )
        analyzer = RoleSequenceAnalyzer()
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["role_sequence_valid"] is True
        assert all(result["role_is_valid"])

    def test_detects_consecutive_same_role(self):
        """Test detection of consecutive same roles."""
        df = pd.DataFrame(
            {
                "role": ["user", "user", "assistant"],
                "conversation_id": ["conv1"] * 3,
            }
        )
        analyzer = RoleSequenceAnalyzer()
        result = analyzer.analyze_sample(df)

        assert result.iloc[1]["role_has_consecutive_same"] is True
        assert result.iloc[0]["role_has_consecutive_same"] is False

    def test_detects_invalid_roles(self):
        """Test detection of invalid role values."""
        df = pd.DataFrame(
            {
                "role": ["user", "bot", "assistant"],
                "conversation_id": ["conv1"] * 3,
            }
        )
        analyzer = RoleSequenceAnalyzer()
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["role_is_valid"] is True
        assert result.iloc[1]["role_is_valid"] is False
        assert result.iloc[2]["role_is_valid"] is True


# ============================================================================
# StatisticalOutlierAnalyzer Tests
# ============================================================================


class TestStatisticalOutlierAnalyzer:
    """Tests for StatisticalOutlierAnalyzer."""

    def test_detects_zscore_outliers(self):
        """Test z-score based outlier detection."""
        # Create data with one clear outlier
        df = pd.DataFrame({"value": [10, 11, 12, 10, 11, 100]})  # 100 is outlier
        analyzer = StatisticalOutlierAnalyzer(zscore_threshold=2.0)
        result = analyzer.analyze_sample(df)

        # The value 100 should be detected as an outlier
        assert result.iloc[5]["value_is_outlier_zscore"] is True
        assert result.iloc[0]["value_is_outlier_zscore"] is False

    def test_computes_percentiles(self):
        """Test percentile computation."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        analyzer = StatisticalOutlierAnalyzer()
        result = analyzer.analyze_sample(df)

        # Check percentiles are computed
        assert "value_percentile" in result.columns
        assert result.iloc[0]["value_percentile"] == 20.0  # 1 is 20th percentile
        assert result.iloc[4]["value_percentile"] == 100.0  # 5 is 100th percentile

    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})
        analyzer = StatisticalOutlierAnalyzer(iqr_multiplier=1.5)
        result = analyzer.analyze_sample(df)

        assert result.iloc[5]["value_is_outlier_iqr"] is True
        assert result.iloc[2]["value_is_outlier_iqr"] is False


# ============================================================================
# NgramAnalyzer Tests
# ============================================================================


class TestNgramAnalyzer:
    """Tests for NgramAnalyzer."""

    def test_counts_ngrams(self):
        """Test n-gram counting."""
        df = pd.DataFrame({"text_content": ["the quick brown fox"]})
        analyzer = NgramAnalyzer(n=2)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # "the quick brown fox" has 3 bigrams
        assert result.iloc[0]["text_content_ngram_count"] == 3

    def test_detects_overrepresented_ngrams(self):
        """Test detection of overrepresented n-grams."""
        # Create dataset where "hello world" appears in >50% of samples
        df = pd.DataFrame(
            {
                "text_content": [
                    "hello world today",
                    "hello world tomorrow",
                    "hello world forever",
                    "something else entirely",
                ]
            }
        )
        analyzer = NgramAnalyzer(n=2, min_document_frequency=0.5)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First 3 samples contain "hello world" which appears in 75% of samples
        assert result.iloc[0]["text_content_contains_overrepresented"] is True
        assert result.iloc[3]["text_content_contains_overrepresented"] is False

    def test_unique_ngram_ratio(self):
        """Test unique n-gram ratio computation."""
        df = pd.DataFrame({"text_content": ["the the the the"]})  # All same bigrams
        analyzer = NgramAnalyzer(n=2)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Only 1 unique bigram out of 3 total
        assert result.iloc[0]["text_content_unique_ngram_ratio"] == pytest.approx(
            1 / 3, rel=0.01
        )


# ============================================================================
# RepetitionAnalyzer Tests
# ============================================================================


class TestRepetitionAnalyzer:
    """Tests for RepetitionAnalyzer."""

    def test_detects_word_repetition(self):
        """Test word repetition detection."""
        df = pd.DataFrame(
            {"text_content": ["hello hello hello world", "one two three four"]}
        )
        analyzer = RepetitionAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First text has 2 repeated "hello" (3 occurrences - 1 = 2 repetitions)
        assert result.iloc[0]["text_content_word_repetition_ratio"] > 0
        assert result.iloc[1]["text_content_word_repetition_ratio"] == 0

    def test_unique_word_ratio(self):
        """Test unique word ratio computation."""
        df = pd.DataFrame({"text_content": ["a a a a"]})  # 1 unique / 4 total
        analyzer = RepetitionAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_unique_word_ratio"] == 0.25

    def test_excessive_repetition_flag(self):
        """Test excessive repetition flagging."""
        df = pd.DataFrame({"text_content": ["word word word word", "one two three four"]})
        analyzer = RepetitionAnalyzer(repetition_threshold=0.5)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_has_excessive_repetition"] is True
        assert result.iloc[1]["text_content_has_excessive_repetition"] is False


# ============================================================================
# VocabularyAnalyzer Tests
# ============================================================================


class TestVocabularyAnalyzer:
    """Tests for VocabularyAnalyzer."""

    def test_vocabulary_size(self):
        """Test vocabulary size computation."""
        df = pd.DataFrame({"text_content": ["one two three four five"]})
        analyzer = VocabularyAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_vocabulary_size"] == 5

    def test_type_token_ratio(self):
        """Test type-token ratio computation."""
        df = pd.DataFrame({"text_content": ["a a b b c c"]})  # 3 types, 6 tokens
        analyzer = VocabularyAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_type_token_ratio"] == 0.5

    def test_hapax_legomena(self):
        """Test hapax legomena (words appearing once) computation."""
        df = pd.DataFrame({"text_content": ["one two two three three three"]})
        analyzer = VocabularyAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Only "one" appears exactly once
        assert result.iloc[0]["text_content_hapax_count"] == 1

    def test_case_insensitive_by_default(self):
        """Test case-insensitive vocabulary analysis."""
        df = pd.DataFrame({"text_content": ["Hello HELLO hello"]})
        analyzer = VocabularyAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_vocabulary_size"] == 1
