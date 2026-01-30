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

"""Standardized LLM judgment result model."""

from pydantic import BaseModel, Field


class LLMJudgmentMetrics(BaseModel):
    """Standardized result model for all LLM-based evaluations.

    Provides a consistent output format across different evaluation criteria
    (usefulness, safety, factuality, etc.). Supports multiple judgment types:
    - SCORE: 0-100 numeric score (default)
    - BOOL: True/False pass/fail judgment
    - ENUM: Categorical labels

    The score field is always populated (normalized to 0-100) for consistency,
    even when using BOOL or ENUM judgment types.

    Example with score:
        >>> result = LLMJudgmentMetrics(
        ...     score=85,
        ...     reasoning="The response accurately addresses the question...",
        ...     criteria="usefulness",
        ... )
        >>> print(f"{result.criteria}: {result.score}/100 ({result.label})")
        usefulness: 85/100 (excellent)

    Example with boolean judgment:
        >>> result = LLMJudgmentMetrics(
        ...     score=100,  # True -> 100
        ...     judgment=True,
        ...     reasoning="Response follows instructions",
        ...     criteria="instruction_following",
        ... )
        >>> print(f"Passed: {result.judgment}")
        Passed: True

    Example with enum category:
        >>> result = LLMJudgmentMetrics(
        ...     score=66,  # Mapped from category position
        ...     category="hard",
        ...     reasoning="Requires domain expertise",
        ...     criteria="prompt_difficulty",
        ... )
        >>> print(f"Difficulty: {result.category}")
        Difficulty: hard
    """

    # Normalized score (0-100) - always computed
    score: int = Field(
        description="Evaluation score from 0-100 (0=worst, 100=best)",
        ge=0,
        le=100,
    )

    # LLM reasoning/explanation
    reasoning: str = Field(
        description="LLM's explanation for the judgment"
    )

    # Which criteria was evaluated
    criteria: str = Field(
        description="The evaluation criteria used (e.g., 'usefulness', 'safety')"
    )

    # Derived categorical label based on score
    label: str = Field(
        description="Categorical label: 'poor' (0-24), 'fair' (25-49), "
        "'good' (50-74), 'excellent' (75-100)"
    )

    # Pass/fail flag (configurable threshold, default 50)
    passed: bool = Field(
        description="Whether the score meets the passing threshold (default: >= 50)"
    )

    # Boolean judgment (for BOOL judgment type)
    judgment: bool | None = Field(
        default=None,
        description="Boolean judgment for BOOL type (True=pass, False=fail)"
    )

    # Category (for ENUM judgment type)
    category: str | None = Field(
        default=None,
        description="Categorical result for ENUM type (e.g., 'easy', 'medium', 'hard')"
    )

    # Raw LLM response for debugging
    raw_response: str | None = Field(
        default=None,
        description="Raw LLM response before parsing (for debugging)"
    )

    # Error tracking
    error: str | None = Field(
        default=None,
        description="Error message if LLM evaluation failed"
    )

    @staticmethod
    def score_to_label(score: int) -> str:
        """Convert a 0-100 score to a categorical label.

        Args:
            score: Score from 0-100.

        Returns:
            Label: 'poor', 'fair', 'good', or 'excellent'.
        """
        if score < 25:
            return "poor"
        elif score < 50:
            return "fair"
        elif score < 75:
            return "good"
        else:
            return "excellent"

    @classmethod
    def from_score(
        cls,
        score: int,
        reasoning: str,
        criteria: str,
        pass_threshold: int = 50,
        judgment: bool | None = None,
        category: str | None = None,
        raw_response: str | None = None,
        error: str | None = None,
    ) -> "LLMJudgmentMetrics":
        """Create a result from a score, auto-computing label and passed.

        Args:
            score: Score from 0-100.
            reasoning: LLM's explanation.
            criteria: The evaluation criteria name.
            pass_threshold: Score threshold for passing (default 50).
            judgment: Boolean judgment for BOOL type.
            category: Categorical result for ENUM type.
            raw_response: Optional raw LLM response.
            error: Optional error message.

        Returns:
            LLMJudgmentMetrics instance with computed label and passed.
        """
        # Clamp score to valid range
        score = max(0, min(100, score))

        return cls(
            score=score,
            reasoning=reasoning,
            criteria=criteria,
            label=cls.score_to_label(score),
            passed=score >= pass_threshold,
            judgment=judgment,
            category=category,
            raw_response=raw_response,
            error=error,
        )

    @classmethod
    def from_judgment(
        cls,
        judgment: bool,
        reasoning: str,
        criteria: str,
        raw_response: str | None = None,
        error: str | None = None,
    ) -> "LLMJudgmentMetrics":
        """Create a result from a boolean judgment.

        Args:
            judgment: True for pass, False for fail.
            reasoning: LLM's explanation.
            criteria: The evaluation criteria name.
            raw_response: Optional raw LLM response.
            error: Optional error message.

        Returns:
            LLMJudgmentMetrics with score set to 100 (True) or 0 (False).
        """
        score = 100 if judgment else 0
        return cls(
            score=score,
            reasoning=reasoning,
            criteria=criteria,
            label=cls.score_to_label(score),
            passed=judgment,
            judgment=judgment,
            category=None,
            raw_response=raw_response,
            error=error,
        )

    @classmethod
    def from_category(
        cls,
        category: str,
        enum_values: list[str],
        reasoning: str,
        criteria: str,
        pass_threshold: int = 50,
        raw_response: str | None = None,
        error: str | None = None,
    ) -> "LLMJudgmentMetrics":
        """Create a result from a categorical judgment.

        Args:
            category: The selected category.
            enum_values: List of valid category values (ordered low to high).
            reasoning: LLM's explanation.
            criteria: The evaluation criteria name.
            pass_threshold: Score threshold for passing (default 50).
            raw_response: Optional raw LLM response.
            error: Optional error message.

        Returns:
            LLMJudgmentMetrics with score mapped from category position.
        """
        # Map category to score based on position
        if category in enum_values and len(enum_values) > 1:
            idx = enum_values.index(category)
            score = int((idx / (len(enum_values) - 1)) * 100)
        else:
            score = 0

        return cls(
            score=score,
            reasoning=reasoning,
            criteria=criteria,
            label=cls.score_to_label(score),
            passed=score >= pass_threshold,
            judgment=None,
            category=category,
            raw_response=raw_response,
            error=error,
        )
