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

"""Tests for the TrainingQualityAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.training_quality_analyzer import TrainingQualityAnalyzer


def _create_test_df(texts: list[str], roles: list[str]) -> pd.DataFrame:
    """Create a test DataFrame with text and role columns."""
    return pd.DataFrame({"text_content": texts, "role": roles})


def _get_schema() -> dict:
    """Get the standard schema for testing."""
    return {
        "text_content": {"content_type": "text"},
        "role": {"content_type": "categorical"},
    }


class TestResponseCompleteness:
    """Tests for response completeness scoring."""

    def test_complete_response_high_score(self):
        """Test that complete responses get high scores."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(
            [
                "Here is the solution to your problem:\n\n"
                "1. First, import the library.\n"
                "2. Then, call the function.\n"
                "3. Finally, print the result.\n\n"
                "This should solve your issue."
            ],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        score = result.iloc[0][
            "text_content_training_quality_response_completeness_score"
        ]
        assert score >= 0.9, f"Complete response should have high score, got {score}"
        assert result.iloc[0]["text_content_training_quality_has_structure"] == True

    def test_truncated_response_low_score(self):
        """Test that truncated responses get penalized."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(
            ["The answer is to use the following approach and"],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert (
            result.iloc[0]["text_content_training_quality_has_proper_ending"] == False
        )

    def test_very_short_response(self):
        """Test that very short responses get low scores."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(["Sure."], ["assistant"])
        result = analyzer.analyze_sample(df, _get_schema())

        score = result.iloc[0][
            "text_content_training_quality_response_completeness_score"
        ]
        assert score < 0.8, f"Short response should have low score, got {score}"

    def test_structured_response_with_code(self):
        """Test that responses with code blocks are detected as structured."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(
            [
                "Here's the code:\n\n```python\ndef hello():\n    print('Hello')\n```"
                "\n\nThis function prints hello."
            ],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_has_structure"] == True

    def test_structured_response_with_bullets(self):
        """Test that responses with bullet points are detected as structured."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(
            [
                "Here are the steps:\n"
                "- First, open the file.\n"
                "- Second, read the content.\n"
                "- Third, process the data."
            ],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_has_structure"] == True

    def test_response_word_count(self):
        """Test that response word count is computed correctly."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(
            ["Here is a response with exactly eight words total."],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        word_count = result.iloc[0][
            "text_content_training_quality_response_word_count"
        ]
        assert word_count == 9  # Count actual words

    def test_user_messages_not_analyzed(self):
        """Test that user messages don't get response completeness metrics."""
        analyzer = TrainingQualityAnalyzer(compute_response_completeness=True)
        df = _create_test_df(
            ["Can you help me with this problem?"],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        # User messages should have None/NaN for response completeness
        assert pd.isna(
            result.iloc[0]["text_content_training_quality_response_completeness_score"]
        )


class TestIntegration:
    """Integration tests for the full analyzer."""

    def test_multi_turn_conversation(self):
        """Test analysis of a multi-turn conversation."""
        analyzer = TrainingQualityAnalyzer()
        df = _create_test_df(
            [
                "Write a function to calculate factorial.",
                "Here's a recursive factorial function:\n\n"
                "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n"
                "    return n * factorial(n-1)\n```",
                "Can you also add error handling?",
                "Sure! Here's the updated version "
                "with error handling:\n\n```python\ndef factorial(n):\n    "
                "if not isinstance(n, int) or n < 0:\n        "
                "raise ValueError('n must be non-negative integer')\n    "
                "if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```",
            ],
            ["user", "assistant", "user", "assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        # User messages should have None/NaN for response metrics
        assert pd.isna(
            result.iloc[0]["text_content_training_quality_response_completeness_score"]
        )
        assert pd.isna(
            result.iloc[2]["text_content_training_quality_response_completeness_score"]
        )

        # Assistant messages should have response metrics
        assert (
            result.iloc[1]["text_content_training_quality_response_completeness_score"]
            is not None
        )
        assert (
            result.iloc[3]["text_content_training_quality_response_completeness_score"]
            is not None
        )

    def test_no_role_column(self):
        """Test that analyzer works without a role column."""
        analyzer = TrainingQualityAnalyzer()
        df = pd.DataFrame({"text_content": ["Write a function."]})
        schema = {"text_content": {"content_type": "text"}}

        result = analyzer.analyze_sample(df, schema)

        # Should process without errors
        assert len(result) == 1

    def test_schema_required(self):
        """Test that schema is required."""
        analyzer = TrainingQualityAnalyzer()
        df = _create_test_df(["Test"], ["user"])

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, None)

    def test_empty_text_columns(self):
        """Test handling when no text columns in schema."""
        analyzer = TrainingQualityAnalyzer()
        df = pd.DataFrame({"other_column": ["test"]})
        schema = {"other_column": {"content_type": "metadata"}}

        result = analyzer.analyze_sample(df, schema)

        # Should return unchanged DataFrame
        assert list(result.columns) == list(df.columns)

    def test_custom_min_response_words(self):
        """Test analyzer with custom min_response_words parameter."""
        analyzer = TrainingQualityAnalyzer(min_response_words=10)
        df = _create_test_df(["A short response."], ["assistant"])
        result = analyzer.analyze_sample(df, _get_schema())

        # Short response should be penalized more with higher threshold
        score = result.iloc[0][
            "text_content_training_quality_response_completeness_score"
        ]
        assert score < 1.0
