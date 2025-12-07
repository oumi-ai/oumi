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

"""Tests for the RecommendationsEngine."""

import pandas as pd
import pytest

from oumi.core.analyze.recommendations import (
    Recommendation,
    RecommendationCategory,
    RecommendationsEngine,
    RecommendationSeverity,
)


class TestRecommendation:
    """Tests for the Recommendation dataclass."""

    def test_recommendation_to_dict(self):
        """Test converting recommendation to dictionary."""
        rec = Recommendation(
            category=RecommendationCategory.WARNING,
            severity=RecommendationSeverity.HIGH,
            title="Test Warning",
            description="This is a test warning.",
            affected_samples=100,
            metric_name="test_metric",
            threshold=3.0,
            details={"key": "value"},
        )
        result = rec.to_dict()
        assert result["category"] == "warning"
        assert result["severity"] == "high"
        assert result["title"] == "Test Warning"
        assert result["description"] == "This is a test warning."
        assert result["affected_samples"] == 100
        assert result["metric_name"] == "test_metric"
        assert result["threshold"] == 3.0
        assert result["details"] == {"key": "value"}

    def test_recommendation_default_details(self):
        """Test recommendation with default empty details."""
        rec = Recommendation(
            category=RecommendationCategory.INSIGHT,
            severity=RecommendationSeverity.LOW,
            title="Test",
            description="Description",
            affected_samples=10,
        )
        assert rec.details == {}


class TestRecommendationsEngineOutliers:
    """Tests for outlier detection."""

    def test_detect_high_outliers(self):
        """Test detection of high outliers."""
        engine = RecommendationsEngine(outlier_std_threshold=2.0)

        # Create data with outliers
        # Normal values around 100, one outlier at 500
        values = [100] * 99 + [500]
        df = pd.DataFrame(
            {
                "text_content": ["text"] * 100,
                "text_content_length_char_count": values,
            }
        )

        recommendations = engine._check_outliers(df, {})

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.category == RecommendationCategory.WARNING
        assert "outlier" in rec.title.lower()
        assert rec.affected_samples >= 1

    def test_detect_low_outliers(self):
        """Test detection of low outliers."""
        engine = RecommendationsEngine(outlier_std_threshold=2.0)

        # Create data with outliers
        # Normal values around 100, one outlier at 1
        values = [100] * 99 + [1]
        df = pd.DataFrame(
            {
                "text_content": ["text"] * 100,
                "text_content_length_char_count": values,
            }
        )

        recommendations = engine._check_outliers(df, {})

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.details["low_outliers"] >= 1

    def test_no_outliers_when_uniform(self):
        """Test no outliers detected for uniform data."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "text_content": ["text"] * 100,
                "text_content_length_char_count": [100] * 100,
            }
        )

        recommendations = engine._check_outliers(df, {})
        assert len(recommendations) == 0


class TestRecommendationsEngineDuplicates:
    """Tests for duplicate detection."""

    def test_detect_duplicates(self):
        """Test detection of duplicate content."""
        engine = RecommendationsEngine(duplicate_warn_threshold=0.05)

        # Create data with duplicates
        df = pd.DataFrame(
            {
                "text_content": ["duplicate"] * 20 + [f"unique_{i}" for i in range(80)],
            }
        )

        recommendations = engine._check_duplicates(df)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert rec.category == RecommendationCategory.WARNING
        assert "duplicate" in rec.title.lower()
        assert rec.affected_samples == 20

    def test_no_duplicates(self):
        """Test no recommendation when no duplicates."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "text_content": [f"unique_{i}" for i in range(100)],
            }
        )

        recommendations = engine._check_duplicates(df)
        assert len(recommendations) == 0

    def test_duplicates_below_threshold(self):
        """Test no recommendation when duplicates below threshold."""
        engine = RecommendationsEngine(duplicate_warn_threshold=0.5)

        # 10% duplicates, but threshold is 50%
        df = pd.DataFrame(
            {
                "text_content": ["duplicate"] * 10 + [f"unique_{i}" for i in range(90)],
            }
        )

        recommendations = engine._check_duplicates(df)
        assert len(recommendations) == 0


class TestRecommendationsEngineEmptyContent:
    """Tests for empty content detection."""

    def test_detect_empty_content(self):
        """Test detection of empty content."""
        engine = RecommendationsEngine(empty_content_threshold=5)

        df = pd.DataFrame(
            {
                "text_content": [""] * 10 + ["normal text content"] * 90,
            }
        )

        recommendations = engine._check_empty_content(df)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert "empty" in rec.title.lower()
        assert rec.affected_samples == 10

    def test_no_empty_content(self):
        """Test no recommendation when no empty content."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "text_content": ["normal text content"] * 100,
            }
        )

        recommendations = engine._check_empty_content(df)
        assert len(recommendations) == 0


class TestRecommendationsEngineShortContent:
    """Tests for short content detection."""

    def test_detect_short_content(self):
        """Test detection of short content."""
        engine = RecommendationsEngine(short_content_threshold=10)

        # Create data where >10% is short
        df = pd.DataFrame(
            {
                "text_content": ["short"] * 50 + ["this is longer content"] * 50,
                "text_content_length_word_count": [1] * 50 + [20] * 50,
            }
        )

        recommendations = engine._check_short_content(df)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert "short" in rec.title.lower()

    def test_no_short_content(self):
        """Test no recommendation when no short content."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "text_content": ["this is a longer message with many words"] * 100,
                "text_content_length_word_count": [20] * 100,
            }
        )

        recommendations = engine._check_short_content(df)
        assert len(recommendations) == 0


class TestRecommendationsEngineRoleDistribution:
    """Tests for role distribution analysis."""

    def test_detect_imbalanced_roles(self):
        """Test detection of imbalanced role distribution."""
        engine = RecommendationsEngine(imbalance_threshold=0.8)

        # 90% assistant, 10% user
        df = pd.DataFrame(
            {
                "role": ["assistant"] * 90 + ["user"] * 10,
                "text_content": ["text"] * 100,
            }
        )

        recommendations = engine._check_role_distribution(df)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert "imbalance" in rec.title.lower()
        assert "assistant" in rec.title.lower()

    def test_balanced_roles(self):
        """Test no recommendation for balanced roles."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "role": ["assistant"] * 50 + ["user"] * 50,
                "text_content": ["text"] * 100,
            }
        )

        recommendations = engine._check_role_distribution(df)
        assert len(recommendations) == 0


class TestRecommendationsEngineTokenLengths:
    """Tests for token length analysis."""

    def test_detect_exceeding_context_window(self):
        """Test detection of messages exceeding context window."""
        engine = RecommendationsEngine(token_warn_thresholds=[4096])

        # Some messages exceed 4096 tokens
        df = pd.DataFrame(
            {
                "text_content": ["text"] * 100,
                "text_content_length_token_count": [100] * 90 + [5000] * 10,
            }
        )

        recommendations = engine._check_token_lengths(df, {})

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert "token" in rec.title.lower()
        assert "4k" in rec.title or "4096" in rec.title

    def test_no_token_issues(self):
        """Test no recommendation when all tokens within limits."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "text_content": ["text"] * 100,
                "text_content_length_token_count": [100] * 100,
            }
        )

        recommendations = engine._check_token_lengths(df, {})
        assert len(recommendations) == 0


class TestRecommendationsEngineConversationLength:
    """Tests for conversation length analysis."""

    def test_detect_single_turn_conversations(self):
        """Test detection of many single-turn conversations."""
        engine = RecommendationsEngine()

        # 60% single-turn
        df = pd.DataFrame(
            {
                "conversation_id": range(100),
                "num_messages": [2] * 60 + [10] * 40,
            }
        )

        recommendations = engine._check_conversation_length_distribution(df)

        # Should have at least one recommendation about single-turn
        single_turn_recs = [r for r in recommendations if "single" in r.title.lower()]
        assert len(single_turn_recs) == 1

    def test_detect_very_long_conversations(self):
        """Test detection of very long conversations."""
        engine = RecommendationsEngine()

        df = pd.DataFrame(
            {
                "conversation_id": range(100),
                "num_messages": [5] * 90 + [50] * 10,
            }
        )

        recommendations = engine._check_conversation_length_distribution(df)

        # Should have recommendation about long conversations
        long_recs = [r for r in recommendations if "long" in r.title.lower()]
        assert len(long_recs) == 1


class TestRecommendationsEngineFullPipeline:
    """Tests for the full recommendation generation pipeline."""

    def test_generate_recommendations_full(self):
        """Test full recommendation generation."""
        engine = RecommendationsEngine(
            outlier_std_threshold=2.0,
            duplicate_warn_threshold=0.05,
        )

        # Create message DataFrame with various issues
        message_df = pd.DataFrame(
            {
                "text_content": ["duplicate"] * 10 + [f"unique_{i}" for i in range(90)],
                "text_content_length_char_count": [100] * 99 + [1000],
                "text_content_length_word_count": [20] * 100,
                "role": ["assistant"] * 50 + ["user"] * 50,
            }
        )

        # Create conversation DataFrame
        conversation_df = pd.DataFrame(
            {
                "conversation_id": range(50),
                "num_messages": [4] * 50,
            }
        )

        summary = {"message_level_summary": {}}

        recommendations = engine.generate_recommendations(
            message_df, conversation_df, summary
        )

        # Should have some recommendations
        assert len(recommendations) > 0

        # Should be sorted by severity
        severities = [r.severity for r in recommendations]
        severity_order = {
            RecommendationSeverity.HIGH: 0,
            RecommendationSeverity.MEDIUM: 1,
            RecommendationSeverity.LOW: 2,
        }
        for i in range(len(severities) - 1):
            assert severity_order[severities[i]] <= severity_order[severities[i + 1]]

    def test_generate_recommendations_clean_data(self):
        """Test recommendation generation for clean data."""
        engine = RecommendationsEngine()

        # Create clean data
        message_df = pd.DataFrame(
            {
                "text_content": [f"unique message {i} with good content" for i in range(100)],
                "text_content_length_char_count": [100] * 100,
                "text_content_length_word_count": [20] * 100,
                "role": ["assistant"] * 50 + ["user"] * 50,
            }
        )

        conversation_df = pd.DataFrame(
            {
                "conversation_id": range(50),
                "num_messages": [4] * 50,
            }
        )

        summary = {}

        recommendations = engine.generate_recommendations(
            message_df, conversation_df, summary
        )

        # Should have few or no recommendations for clean data
        # (may still have some insights)
        high_severity = [
            r for r in recommendations if r.severity == RecommendationSeverity.HIGH
        ]
        assert len(high_severity) == 0


class TestRecommendationsEngineEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        engine = RecommendationsEngine()

        message_df = pd.DataFrame(columns=["text_content"])
        conversation_df = pd.DataFrame(columns=["num_messages"])

        recommendations = engine.generate_recommendations(message_df, conversation_df, {})

        # Should not crash, may return empty or minimal recommendations
        assert isinstance(recommendations, list)

    def test_missing_columns(self):
        """Test handling of missing columns."""
        engine = RecommendationsEngine()

        # DataFrame without expected columns
        message_df = pd.DataFrame({"other_column": [1, 2, 3]})
        conversation_df = pd.DataFrame({"other_column": [1, 2, 3]})

        recommendations = engine.generate_recommendations(message_df, conversation_df, {})

        # Should not crash
        assert isinstance(recommendations, list)

    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        engine = RecommendationsEngine()

        message_df = pd.DataFrame(
            {
                "text_content": ["single message"],
                "text_content_length_char_count": [100],
            }
        )
        conversation_df = pd.DataFrame(
            {
                "num_messages": [1],
            }
        )

        recommendations = engine.generate_recommendations(message_df, conversation_df, {})

        # Should not crash
        assert isinstance(recommendations, list)
