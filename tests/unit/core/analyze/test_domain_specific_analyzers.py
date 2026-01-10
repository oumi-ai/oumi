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

"""Tests for domain-specific duplicate analyzers."""

import pandas as pd
import pytest

from oumi.core.analyze.qa_pair_analyzer import QuestionAnswerPairAnalyzer
from oumi.core.analyze.question_duplicate_analyzer import QuestionDuplicateAnalyzer
from oumi.core.analyze.response_duplicate_analyzer import ResponseDuplicateAnalyzer
from oumi.core.analyze.system_prompt_analyzer import SystemPromptAnalyzer

TEXT_SCHEMA = {"text_content": {"content_type": "text"}}


# ============================================================================
# SystemPromptAnalyzer Tests
# ============================================================================


class TestSystemPromptAnalyzer:
    """Tests for SystemPromptAnalyzer."""

    def test_detects_missing_system_prompts(self):
        """Test detection of conversations without system prompts."""
        df = pd.DataFrame(
            {
                "conversation_id": [1, 1, 2, 2],
                "role": ["user", "assistant", "user", "assistant"],
                "text_content": ["Question 1", "Answer 1", "Question 2", "Answer 2"],
            }
        )
        analyzer = SystemPromptAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Both conversations should be marked as missing system prompts
        assert result.iloc[0]["system_prompt_missing"] == True
        assert result.iloc[2]["system_prompt_missing"] == True

    def test_detects_common_system_templates(self):
        """Test detection of common system prompt templates."""
        df = pd.DataFrame(
            {
                "conversation_id": [1, 1, 2, 2, 3, 3],
                "role": ["system", "user", "system", "user", "system", "user"],
                "text_content": [
                    "You are a helpful assistant.",
                    "Question 1",
                    "You are a helpful assistant.",
                    "Question 2",
                    "Different system prompt",
                    "Question 3",
                ],
            }
        )
        analyzer = SystemPromptAnalyzer(min_template_frequency=0.4)
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First two conversations should have common template
        assert result.iloc[0]["system_prompt_is_common_template"] == True
        assert result.iloc[2]["system_prompt_is_common_template"] == True
        # Third conversation has unusual system prompt
        assert result.iloc[4]["system_prompt_is_unusual"] == True

    def test_ranks_system_prompt_templates(self):
        """Test ranking of system prompt templates by frequency."""
        df = pd.DataFrame(
            {
                "conversation_id": [1, 1, 2, 2, 3, 3],
                "role": ["system", "user", "system", "user", "system", "user"],
                "text_content": [
                    "Template A",
                    "Q1",
                    "Template A",
                    "Q2",
                    "Template B",
                    "Q3",
                ],
            }
        )
        analyzer = SystemPromptAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Template A should be rank 1 (most common)
        assert result.iloc[0]["system_prompt_template_rank"] == 1
        # Template B should be rank 2
        assert result.iloc[4]["system_prompt_template_rank"] == 2


# ============================================================================
# QuestionAnswerPairAnalyzer Tests
# ============================================================================


class TestQuestionAnswerPairAnalyzer:
    """Tests for QuestionAnswerPairAnalyzer."""

    def test_detects_duplicate_qa_pairs(self):
        """Test detection of duplicate question-answer pairs."""
        df = pd.DataFrame(
            {
                "conversation_id": [1, 1, 2, 2, 3, 3],
                "message_index": [0, 1, 0, 1, 0, 1],
                "role": ["user", "assistant", "user", "assistant", "user", "assistant"],
                "text_content": [
                    "What is 2+2?",
                    "4",
                    "What is 2+2?",
                    "4",
                    "What is 3+3?",
                    "6",
                ],
            }
        )
        analyzer = QuestionAnswerPairAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First two QA pairs should be duplicates
        assert result.iloc[0]["qa_pair_is_duplicate"] == True
        assert result.iloc[1]["qa_pair_is_duplicate"] == True
        assert result.iloc[2]["qa_pair_is_duplicate"] == True
        assert result.iloc[3]["qa_pair_is_duplicate"] == True

        # Third QA pair should be unique
        assert result.iloc[4]["qa_pair_is_duplicate"] == False
        assert result.iloc[5]["qa_pair_is_duplicate"] == False

    def test_case_insensitive_qa_pairs(self):
        """Test case-insensitive QA pair matching."""
        df = pd.DataFrame(
            {
                "conversation_id": [1, 1, 2, 2],
                "message_index": [0, 1, 0, 1],
                "role": ["user", "assistant", "user", "assistant"],
                "text_content": ["Hello", "Hi", "HELLO", "HI"],
            }
        )
        analyzer = QuestionAnswerPairAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Should detect as duplicates (case-insensitive)
        assert result.iloc[0]["qa_pair_is_duplicate"] == True


# ============================================================================
# QuestionDuplicateAnalyzer Tests
# ============================================================================


class TestQuestionDuplicateAnalyzer:
    """Tests for QuestionDuplicateAnalyzer."""

    def test_detects_duplicate_questions(self):
        """Test detection of duplicate user questions."""
        df = pd.DataFrame(
            {
                "role": ["user", "assistant", "user", "assistant", "user", "assistant"],
                "text_content": [
                    "What is Python?",
                    "Python is a programming language.",
                    "What is Python?",
                    "Python is a language used for coding.",
                    "What is Java?",
                    "Java is another programming language.",
                ],
            }
        )
        analyzer = QuestionDuplicateAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First and third questions should be duplicates
        assert result.iloc[0]["question_is_duplicate"] == True
        assert result.iloc[2]["question_is_duplicate"] == True
        assert result.iloc[0]["question_duplicate_count"] == 2

        # Fifth question should be unique
        assert result.iloc[4]["question_is_duplicate"] == False

    def test_classifies_duplication_levels(self):
        """Test classification of duplication levels."""
        # Create dataset with 10 questions, 3 duplicates (30% duplication)
        df = pd.DataFrame(
            {
                "role": ["user"] * 10,
                "text_content": ["Q1", "Q2", "Q3", "Q1", "Q4", "Q5", "Q2", "Q6", "Q7", "Q8"],
            }
        )
        analyzer = QuestionDuplicateAnalyzer(
            acceptable_duplication=0.15, high_duplication_threshold=0.25
        )
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # With 30% duplication (above threshold), should classify as "high"
        assert result.iloc[0]["question_duplication_level"] == "high"

    def test_only_analyzes_user_messages(self):
        """Test that only user messages are analyzed."""
        df = pd.DataFrame(
            {
                "role": ["user", "assistant", "system"],
                "text_content": ["Question", "Answer", "System prompt"],
            }
        )
        analyzer = QuestionDuplicateAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Only user message should have question_hash
        assert pd.notna(result.iloc[0]["question_hash"])
        assert pd.isna(result.iloc[1]["question_hash"])
        assert pd.isna(result.iloc[2]["question_hash"])


# ============================================================================
# ResponseDuplicateAnalyzer Tests
# ============================================================================


class TestResponseDuplicateAnalyzer:
    """Tests for ResponseDuplicateAnalyzer."""

    def test_detects_duplicate_responses(self):
        """Test detection of duplicate assistant responses."""
        df = pd.DataFrame(
            {
                "role": ["user", "assistant", "user", "assistant", "user", "assistant"],
                "text_content": ["Q1", "Yes", "Q2", "Yes", "Q3", "No"],
            }
        )
        analyzer = ResponseDuplicateAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First and third responses ("Yes") should be duplicates
        assert result.iloc[1]["response_is_duplicate"] == True
        assert result.iloc[3]["response_is_duplicate"] == True
        assert result.iloc[1]["response_duplicate_count"] == 2

        # Fifth response should be unique
        assert result.iloc[5]["response_is_duplicate"] == False

    def test_detects_short_responses(self):
        """Test detection of short responses."""
        df = pd.DataFrame(
            {
                "role": ["assistant", "assistant", "assistant"],
                "text_content": [
                    "Yes",
                    "This is a much longer response with more content.",
                    "No",
                ],
            }
        )
        analyzer = ResponseDuplicateAnalyzer(short_response_length=20)
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # First and third responses should be marked as short
        assert result.iloc[0]["response_is_short"] == True
        assert result.iloc[1]["response_is_short"] == False
        assert result.iloc[2]["response_is_short"] == True

    def test_detects_generic_responses(self):
        """Test detection of generic/common responses."""
        # Create dataset with 100 responses, "Yes" appears 15 times (>1% threshold)
        responses = ["Yes"] * 15 + [f"Response {i}" for i in range(85)]
        df = pd.DataFrame({"role": ["assistant"] * 100, "text_content": responses})

        analyzer = ResponseDuplicateAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # "Yes" should be marked as generic (appears >1% of times)
        assert result.iloc[0]["response_is_generic"] == True

    def test_only_analyzes_assistant_messages(self):
        """Test that only assistant messages are analyzed."""
        df = pd.DataFrame(
            {
                "role": ["user", "assistant", "system"],
                "text_content": ["Question", "Answer", "System prompt"],
            }
        )
        analyzer = ResponseDuplicateAnalyzer()
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # Only assistant message should have response_hash
        assert pd.isna(result.iloc[0]["response_hash"])
        assert pd.notna(result.iloc[1]["response_hash"])
        assert pd.isna(result.iloc[2]["response_hash"])

    def test_classifies_duplication_levels(self):
        """Test classification of response duplication levels."""
        # Create dataset with 20 responses, 3 duplicates (15% duplication)
        # Use longer response to avoid "acceptable_short" classification
        df = pd.DataFrame(
            {
                "role": ["assistant"] * 20,
                "text_content": ["This is a longer response text"] * 3 + [f"Response {i}" for i in range(17)],
            }
        )
        analyzer = ResponseDuplicateAnalyzer(
            acceptable_duplication=0.05, high_duplication_threshold=0.10
        )
        result, _ = analyzer.analyze_sample(df, schema=TEXT_SCHEMA)

        # With 15% duplication (above threshold), should classify as "high"
        assert result.iloc[0]["response_duplication_level"] == "high"
