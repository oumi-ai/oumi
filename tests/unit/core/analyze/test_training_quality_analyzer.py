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


class TestInstructionClarity:
    """Tests for instruction clarity scoring."""

    def test_clear_instruction_high_score(self):
        """Test that clear instructions get high scores."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=True,
            compute_response_completeness=False,
            compute_turn_quality=False,
        )
        df = _create_test_df(
            ["Write a Python function that calculates the factorial of 5."],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        score = result.iloc[0]["text_content_training_quality_instruction_clarity_score"]
        assert score >= 0.9, f"Clear instruction should have high score, got {score}"
        assert result.iloc[0]["text_content_training_quality_has_clear_intent"] == True

    def test_vague_instruction_low_score(self):
        """Test that vague instructions get low scores."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=True,
            compute_response_completeness=False,
            compute_turn_quality=False,
        )
        df = _create_test_df(
            ["Do something with stuff"],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        score = result.iloc[0]["text_content_training_quality_instruction_clarity_score"]
        assert score < 0.7, f"Vague instruction should have low score, got {score}"

    def test_too_short_instruction(self):
        """Test that very short instructions get penalized."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=True,
            compute_response_completeness=False,
            compute_turn_quality=False,
        )
        df = _create_test_df(["Hi"], ["user"])
        result = analyzer.analyze_sample(df, _get_schema())

        score = result.iloc[0]["text_content_training_quality_instruction_clarity_score"]
        assert score < 0.8, f"Short instruction should be penalized, got {score}"

    def test_question_marks_indicate_intent(self):
        """Test that question marks indicate clear intent."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=True,
            compute_response_completeness=False,
            compute_turn_quality=False,
        )
        df = _create_test_df(
            ["What is the capital of France?"],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_has_clear_intent"] == True

    def test_specificity_with_numbers(self):
        """Test that numbers indicate specificity."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=True,
            compute_response_completeness=False,
            compute_turn_quality=False,
        )
        df = _create_test_df(
            ["List 5 reasons why Python is popular."],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_has_specificity"] == True

    def test_specificity_with_code(self):
        """Test that code indicators show specificity."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=True,
            compute_response_completeness=False,
            compute_turn_quality=False,
        )
        df = _create_test_df(
            ["Fix the `calculate_sum` function."],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_has_specificity"] == True


class TestResponseCompleteness:
    """Tests for response completeness scoring."""

    def test_complete_response_high_score(self):
        """Test that complete responses get high scores."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=True,
            compute_turn_quality=False,
        )
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
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=True,
            compute_turn_quality=False,
        )
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
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=True,
            compute_turn_quality=False,
        )
        df = _create_test_df(["Sure."], ["assistant"])
        result = analyzer.analyze_sample(df, _get_schema())

        score = result.iloc[0][
            "text_content_training_quality_response_completeness_score"
        ]
        assert score < 0.5, f"Short response should have low score, got {score}"

    def test_structured_response_with_code(self):
        """Test that responses with code blocks are detected as structured."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=True,
            compute_turn_quality=False,
        )
        df = _create_test_df(
            [
                "Here's the code:\n\n```python\ndef hello():\n    print('Hello')\n```"
                "\n\nThis function prints hello."
            ],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_has_structure"] == True


class TestTurnQuality:
    """Tests for turn quality scoring."""

    def test_context_reference_detected(self):
        """Test that context references are detected."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=False,
            compute_turn_quality=True,
        )
        df = _create_test_df(
            ["As you mentioned earlier, the solution involves using recursion."],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_references_context"] == True

    def test_role_appropriate_user(self):
        """Test that valid user messages are role-appropriate."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=False,
            compute_turn_quality=True,
        )
        df = _create_test_df(
            ["Can you help me with this problem?"],
            ["user"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_role_appropriate"] == True

    def test_role_appropriate_assistant(self):
        """Test that valid assistant messages are role-appropriate."""
        analyzer = TrainingQualityAnalyzer(
            compute_instruction_clarity=False,
            compute_response_completeness=False,
            compute_turn_quality=True,
        )
        df = _create_test_df(
            ["Here is a helpful response to your question."],
            ["assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["text_content_training_quality_role_appropriate"] == True


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
                "Sure! As you mentioned earlier, here's the updated version "
                "with error handling:\n\n```python\ndef factorial(n):\n    "
                "if not isinstance(n, int) or n < 0:\n        "
                "raise ValueError('n must be non-negative integer')\n    "
                "if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```",
            ],
            ["user", "assistant", "user", "assistant"],
        )
        result = analyzer.analyze_sample(df, _get_schema())

        # Check user messages have instruction metrics
        assert (
            result.iloc[0]["text_content_training_quality_instruction_clarity_score"]
            is not None
        )
        assert (
            result.iloc[2]["text_content_training_quality_instruction_clarity_score"]
            is not None
        )

        # Check assistant messages have response metrics
        assert (
            result.iloc[1]["text_content_training_quality_response_completeness_score"]
            is not None
        )
        assert (
            result.iloc[3]["text_content_training_quality_response_completeness_score"]
            is not None
        )

        # Check all messages have turn quality metrics
        for i in range(4):
            assert (
                result.iloc[i]["text_content_training_quality_turn_quality_score"]
                is not None
            )

        # Second assistant response should reference context
        assert (
            result.iloc[3]["text_content_training_quality_references_context"] == True
        )

    def test_no_role_column(self):
        """Test that analyzer works without a role column."""
        analyzer = TrainingQualityAnalyzer()
        df = pd.DataFrame({"text_content": ["Write a function."]})
        schema = {"text_content": {"content_type": "text"}}

        result = analyzer.analyze_sample(df, schema)

        # Should still produce turn quality metrics
        assert (
            "text_content_training_quality_turn_quality_score" in result.columns
        )

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

    def test_custom_parameters(self):
        """Test analyzer with custom parameters."""
        analyzer = TrainingQualityAnalyzer(
            min_instruction_words=5,
            max_instruction_words=100,
            clarity_vague_penalty=0.3,
        )
        df = _create_test_df(["Hi"], ["user"])  # Very short
        result = analyzer.analyze_sample(df, _get_schema())

        # Should be penalized for being short
        score = result.iloc[0]["text_content_training_quality_instruction_clarity_score"]
        assert score < 1.0
