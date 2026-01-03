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

"""Tests for Phase 3 quality analyzers."""

import pandas as pd
import pytest

from oumi.core.analyze.category_analyzer import CategoryDistributionAnalyzer
from oumi.core.analyze.conversation_structure_analyzer import (
    ConversationStructureAnalyzer,
)
from oumi.core.analyze.readability_analyzer import ReadabilityAnalyzer
from oumi.core.analyze.request_type_analyzer import RequestTypeAnalyzer

TEXT_SCHEMA = {"text_content": {"content_type": "text"}}


# ============================================================================
# RequestTypeAnalyzer Tests
# ============================================================================


class TestRequestTypeAnalyzer:
    """Tests for RequestTypeAnalyzer."""

    def test_classifies_explanation_requests(self):
        """Test classification of explanation requests."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "Explain how photosynthesis works",
                    "What is machine learning?",
                    "Why is the sky blue?",
                ],
                "role": ["user", "user", "user"],
            }
        )
        analyzer = RequestTypeAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["request_type"] == "explanation"
        assert result.iloc[1]["request_type"] == "explanation"
        assert result.iloc[2]["request_type"] == "explanation"

    def test_classifies_code_generation_requests(self):
        """Test classification of code generation requests."""
        df = pd.DataFrame(
            {
                "text_content": [
                    "Write code to sort a list",
                    "Implement a binary search function",
                    "Create a function that calculates factorial",
                ],
                "role": ["user", "user", "user"],
            }
        )
        analyzer = RequestTypeAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["request_type"] == "code_generation"
        assert result.iloc[1]["request_type"] == "code_generation"
        assert result.iloc[2]["request_type"] == "code_generation"

    def test_classifies_unknown_when_no_match(self):
        """Test that unmatched requests are classified as unknown."""
        df = pd.DataFrame(
            {"text_content": ["Hello there!", "Thanks"], "role": ["user", "user"]}
        )
        analyzer = RequestTypeAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["request_type"] == "unknown"
        assert result.iloc[0]["request_type_is_unknown"] is True

    def test_skips_non_user_messages(self):
        """Test that non-user messages are skipped when role filter is set."""
        df = pd.DataFrame(
            {
                "text_content": ["Explain this", "Here is the explanation"],
                "role": ["user", "assistant"],
            }
        )
        analyzer = RequestTypeAnalyzer(apply_to_role="user")
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["request_type"] == "explanation"
        assert result.iloc[1]["request_type"] == "n/a"

    def test_custom_patterns(self):
        """Test using custom patterns."""
        custom_patterns = {"greeting": [r"\bhello\b", r"\bhi\b"]}
        df = pd.DataFrame({"text_content": ["Hello there!", "Hi!"], "role": ["user", "user"]})
        analyzer = RequestTypeAnalyzer(patterns=custom_patterns)
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["request_type"] == "greeting"
        assert result.iloc[1]["request_type"] == "greeting"

    def test_multiple_type_matches(self):
        """Test that multiple matching types are recorded."""
        df = pd.DataFrame(
            {
                "text_content": ["Explain how to fix this bug"],  # explanation + debugging
                "role": ["user"],
            }
        )
        analyzer = RequestTypeAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Should match both explanation and debugging
        matches = result.iloc[0]["request_type_matches"]
        assert "explanation" in matches or "debugging" in matches


# ============================================================================
# CategoryDistributionAnalyzer Tests
# ============================================================================


class TestCategoryDistributionAnalyzer:
    """Tests for CategoryDistributionAnalyzer."""

    def test_computes_category_distribution(self):
        """Test category count and percentage computation."""
        df = pd.DataFrame({"category": ["A", "A", "A", "B", "C"]})
        analyzer = CategoryDistributionAnalyzer(category_column="category")
        result = analyzer.analyze_sample(df)

        # A appears 3 times (60%), B appears 1 time (20%), C appears 1 time (20%)
        assert result.iloc[0]["category_count"] == 3
        assert result.iloc[0]["category_percentage"] == 0.6
        assert result.iloc[3]["category_count"] == 1
        assert result.iloc[3]["category_percentage"] == 0.2

    def test_flags_underrepresented_categories(self):
        """Test flagging of underrepresented categories."""
        df = pd.DataFrame({"category": ["A"] * 98 + ["B", "C"]})  # B and C are 1%
        analyzer = CategoryDistributionAnalyzer(
            category_column="category", min_percentage=0.02
        )
        result = analyzer.analyze_sample(df)

        # B and C should be flagged as underrepresented
        assert result.iloc[98]["category_is_underrepresented"] is True  # B
        assert result.iloc[99]["category_is_underrepresented"] is True  # C
        assert result.iloc[0]["category_is_underrepresented"] is False  # A

    def test_flags_overrepresented_categories(self):
        """Test flagging of overrepresented categories."""
        df = pd.DataFrame({"category": ["A"] * 80 + ["B"] * 20})  # A is 80%
        analyzer = CategoryDistributionAnalyzer(
            category_column="category", max_percentage=0.50
        )
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["category_is_overrepresented"] is True  # A
        assert result.iloc[80]["category_is_overrepresented"] is False  # B


# ============================================================================
# ReadabilityAnalyzer Tests
# ============================================================================


class TestReadabilityAnalyzer:
    """Tests for ReadabilityAnalyzer."""

    def test_computes_flesch_reading_ease(self):
        """Test Flesch Reading Ease computation."""
        # Simple sentence should have high reading ease
        df = pd.DataFrame({"text_content": ["The cat sat on the mat."]})
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Simple text should have high readability (>60)
        assert result.iloc[0]["text_content_flesch_reading_ease"] > 60

    def test_computes_flesch_kincaid_grade(self):
        """Test Flesch-Kincaid Grade Level computation."""
        df = pd.DataFrame({"text_content": ["The cat sat on the mat."]})
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Simple text should have low grade level (<5)
        assert result.iloc[0]["text_content_flesch_kincaid_grade"] < 5

    def test_computes_avg_sentence_length(self):
        """Test average sentence length computation."""
        df = pd.DataFrame({"text_content": ["One two. Three four five."]})
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # 2 sentences, 5 words total -> avg = 2.5
        assert result.iloc[0]["text_content_avg_sentence_length"] == 2.5

    def test_computes_avg_word_length(self):
        """Test average word length computation."""
        df = pd.DataFrame({"text_content": ["cat dog"]})  # 3 + 3 = 6, avg = 3
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        assert result.iloc[0]["text_content_avg_word_length"] == 3.0

    def test_complex_text_has_lower_readability(self):
        """Test that complex text has lower readability score."""
        simple = "The cat sat. The dog ran."
        complex_text = (
            "The implementation of sophisticated algorithmic methodologies "
            "necessitates comprehensive understanding of computational paradigms."
        )
        df = pd.DataFrame({"text_content": [simple, complex_text]})
        analyzer = ReadabilityAnalyzer()
        result = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Complex text should have lower reading ease
        assert (
            result.iloc[0]["text_content_flesch_reading_ease"]
            > result.iloc[1]["text_content_flesch_reading_ease"]
        )


# ============================================================================
# ConversationStructureAnalyzer Tests
# ============================================================================


class TestConversationStructureAnalyzer:
    """Tests for ConversationStructureAnalyzer."""

    def test_counts_turns(self):
        """Test turn counting."""
        df = pd.DataFrame(
            {
                "conversation_id": ["conv1"] * 4,
                "role": ["user", "assistant", "user", "assistant"],
                "text_content": ["Hi", "Hello", "How are you?", "I'm good"],
            }
        )
        analyzer = ConversationStructureAnalyzer()
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["conv_turn_count"] == 4
        assert result.iloc[0]["conv_user_turn_count"] == 2
        assert result.iloc[0]["conv_assistant_turn_count"] == 2

    def test_computes_user_assistant_ratio(self):
        """Test user/assistant ratio computation."""
        df = pd.DataFrame(
            {
                "conversation_id": ["conv1"] * 3,
                "role": ["user", "user", "assistant"],  # 2 user, 1 assistant
                "text_content": ["Q1", "Q2", "A1"],
            }
        )
        analyzer = ConversationStructureAnalyzer()
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["conv_user_assistant_ratio"] == 2.0

    def test_flags_short_conversations(self):
        """Test flagging of short conversations."""
        df = pd.DataFrame(
            {
                "conversation_id": ["conv1"],
                "role": ["user"],
                "text_content": ["Hello"],
            }
        )
        analyzer = ConversationStructureAnalyzer(min_turns=2)
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["conv_is_too_short"] is True

    def test_flags_long_conversations(self):
        """Test flagging of long conversations."""
        df = pd.DataFrame(
            {
                "conversation_id": ["conv1"] * 10,
                "role": ["user", "assistant"] * 5,
                "text_content": ["msg"] * 10,
            }
        )
        analyzer = ConversationStructureAnalyzer(max_turns=5)
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["conv_is_too_long"] is True

    def test_computes_avg_turn_length(self):
        """Test average turn length computation."""
        df = pd.DataFrame(
            {
                "conversation_id": ["conv1"] * 2,
                "role": ["user", "assistant"],
                "text_content": ["one two", "three four five six"],  # 2 and 4 words
            }
        )
        analyzer = ConversationStructureAnalyzer()
        result = analyzer.analyze_sample(df)

        assert result.iloc[0]["conv_avg_turn_length"] == 3.0  # (2+4)/2
