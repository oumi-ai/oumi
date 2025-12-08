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

"""Input quality analyzer for rating instruction/input quality.

Based on the Magpie framework from "Fixing It in Post" paper,
this analyzer rates input quality from "very poor" to "excellent"
to help identify high-quality instruction-response pairs.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("input_quality")
class InputQualityAnalyzer(SampleAnalyzer):
    """Analyzer for rating input/instruction quality.

    This analyzer evaluates user instructions on multiple dimensions
    to produce an overall quality rating (very_poor to excellent).

    Quality dimensions assessed:
        - Clarity: Is the instruction clear and unambiguous?
        - Completeness: Does it provide sufficient context?
        - Answerability: Can this instruction be meaningfully answered?
        - Specificity: Is the instruction specific enough?

    Metrics computed:
        - input_quality_tier: Quality tier (very_poor, poor, fair, good, excellent)
        - input_quality_score: Overall quality score (0-1)
        - is_ambiguous: Whether the instruction is ambiguous
        - is_answerable: Whether the instruction is answerable
        - has_sufficient_context: Whether enough context is provided
    """

    # Patterns indicating clear instructions
    _CLEAR_PATTERNS = [
        # Imperative verbs at the start
        re.compile(
            r"^(?:write|create|explain|describe|list|summarize|"
            r"analyze|compare|translate|calculate|solve|find|"
            r"implement|design|develop|generate|provide|give|"
            r"tell|show|help|fix|debug|optimize|review|edit)\b",
            re.IGNORECASE,
        ),
        # Clear question structures
        re.compile(
            r"^(?:what|who|when|where|why|how|which|can\s+you|"
            r"could\s+you|would\s+you|please)\b",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating ambiguous instructions
    _AMBIGUOUS_PATTERNS = [
        re.compile(
            r"\b(?:something|stuff|things?|whatever|anything|"
            r"somehow|somewhat)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:kind\s+of|sort\s+of|basically|actually|really|"
            r"like\s+(?:that|this)|you\s+know)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:etc\.?|and\s+so\s+on|and\s+stuff)\b", re.IGNORECASE),
        re.compile(r"\b(?:maybe|perhaps|probably|possibly)\b", re.IGNORECASE),
    ]

    # Patterns indicating unanswerable/problematic instructions
    _UNANSWERABLE_PATTERNS = [
        # Too vague
        re.compile(r"^(?:hi|hello|hey|yo|sup)\s*[.!?]?\s*$", re.IGNORECASE),
        re.compile(r"^(?:thanks|thank\s+you|ok|okay|cool|nice)\s*[.!?]?\s*$", re.IGNORECASE),
        # Contradictory
        re.compile(
            r"\b(?:but\s+also|and\s+also\s+not|both\s+.+\s+and\s+not)\b",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating good context/specificity
    _CONTEXT_PATTERNS = [
        # Specific references
        re.compile(r"\b(?:specifically|in\s+particular|for\s+example)\b", re.IGNORECASE),
        # Numbers and quantities
        re.compile(r"\b\d+\b"),
        # Quoted content
        re.compile(r'["\'][^"\']+["\']'),
        # Code indicators
        re.compile(r"`[^`]+`|```"),
        # Named entities (capitalized words)
        re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"),
    ]

    # Minimum word counts for quality tiers
    _MIN_WORDS_GOOD = 5
    _MIN_WORDS_EXCELLENT = 10

    def __init__(
        self,
        *,
        analyze_user_only: bool = True,
        include_component_flags: bool = True,
    ):
        """Initialize the InputQualityAnalyzer.

        Args:
            analyze_user_only: If True, only analyze user messages.
            include_component_flags: Include individual quality flags.
        """
        self.analyze_user_only = analyze_user_only
        self.include_component_flags = include_component_flags

    def _compute_clarity_score(self, text: str) -> float:
        """Compute clarity score for the input.

        Args:
            text: Input text.

        Returns:
            Clarity score (0-1).
        """
        score = 0.5  # Start neutral

        # Check for clear patterns
        for pattern in self._CLEAR_PATTERNS:
            if pattern.search(text):
                score += 0.2
                break

        # Check for ambiguous patterns
        ambiguous_count = sum(
            len(pattern.findall(text)) for pattern in self._AMBIGUOUS_PATTERNS
        )
        score -= ambiguous_count * 0.1

        return max(0.0, min(1.0, score))

    def _is_ambiguous(self, text: str) -> bool:
        """Check if the input is ambiguous.

        Args:
            text: Input text.

        Returns:
            True if ambiguous.
        """
        ambiguous_count = sum(
            len(pattern.findall(text)) for pattern in self._AMBIGUOUS_PATTERNS
        )
        return ambiguous_count >= 2

    def _is_answerable(self, text: str) -> bool:
        """Check if the input is answerable.

        Args:
            text: Input text.

        Returns:
            True if answerable.
        """
        # Check for unanswerable patterns
        for pattern in self._UNANSWERABLE_PATTERNS:
            if pattern.match(text.strip()):
                return False

        # Too short to be answerable
        word_count = len(text.split())
        if word_count < 2:
            return False

        return True

    def _has_sufficient_context(self, text: str) -> bool:
        """Check if the input has sufficient context.

        Args:
            text: Input text.

        Returns:
            True if sufficient context.
        """
        word_count = len(text.split())

        # Very short inputs lack context
        if word_count < self._MIN_WORDS_GOOD:
            return False

        # Check for context indicators
        context_count = sum(
            1 for pattern in self._CONTEXT_PATTERNS if pattern.search(text)
        )

        # Need some context indicators for longer instructions
        if word_count >= self._MIN_WORDS_EXCELLENT and context_count == 0:
            return False

        return True

    def _compute_quality_score(
        self,
        text: str,
        clarity: float,
        is_ambiguous: bool,
        is_answerable: bool,
        has_context: bool,
    ) -> float:
        """Compute overall quality score.

        Args:
            text: Input text.
            clarity: Clarity score.
            is_ambiguous: Ambiguity flag.
            is_answerable: Answerability flag.
            has_context: Context sufficiency flag.

        Returns:
            Quality score (0-1).
        """
        score = clarity * 0.4  # Clarity is important

        # Answerable is critical
        if not is_answerable:
            return 0.1

        # Context contribution
        if has_context:
            score += 0.3

        # Ambiguity penalty
        if is_ambiguous:
            score -= 0.2

        # Length bonus
        word_count = len(text.split())
        if word_count >= self._MIN_WORDS_EXCELLENT:
            score += 0.2
        elif word_count >= self._MIN_WORDS_GOOD:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _get_quality_tier(self, score: float) -> str:
        """Convert quality score to tier label.

        Args:
            score: Quality score (0-1).

        Returns:
            Tier label.
        """
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        elif score >= 0.2:
            return "poor"
        else:
            return "very_poor"

    def _analyze_input(self, text: str) -> dict[str, Any]:
        """Analyze an input for quality metrics.

        Args:
            text: Input text.

        Returns:
            Dictionary of quality metrics.
        """
        if not text or not text.strip():
            return {
                "input_quality_tier": "very_poor",
                "input_quality_score": 0.0,
                "is_ambiguous": True,
                "is_answerable": False,
                "has_sufficient_context": False,
            }

        clarity = self._compute_clarity_score(text)
        is_ambiguous = self._is_ambiguous(text)
        is_answerable = self._is_answerable(text)
        has_context = self._has_sufficient_context(text)

        quality_score = self._compute_quality_score(
            text, clarity, is_ambiguous, is_answerable, has_context
        )

        result = {
            "input_quality_tier": self._get_quality_tier(quality_score),
            "input_quality_score": round(quality_score, 3),
        }

        if self.include_component_flags:
            result["is_ambiguous"] = is_ambiguous
            result["is_answerable"] = is_answerable
            result["has_sufficient_context"] = has_context

        return result

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for input quality metrics.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added input quality analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for input quality "
                "analysis. Please provide a column schema dict that specifies "
                "which columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df

        # Find the role column if needed
        role_column = None
        if self.analyze_user_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "input_quality")

        for column in text_columns:
            if self.analyze_user_only and role_column is not None:
                # Only analyze user messages
                analysis_results = df.apply(
                    lambda row: (
                        self._analyze_input(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "user"
                        else {
                            "input_quality_tier": None,
                            "input_quality_score": None,
                            "is_ambiguous": None,
                            "is_answerable": None,
                            "has_sufficient_context": None,
                        }
                    ),
                    axis=1,
                )
            else:
                analysis_results = df[column].astype(str).apply(self._analyze_input)

            # Extract results to columns
            result_df[f"{column}_{analyzer_id}_tier"] = analysis_results.apply(
                lambda r: r.get("input_quality_tier")
            )
            result_df[f"{column}_{analyzer_id}_score"] = analysis_results.apply(
                lambda r: r.get("input_quality_score")
            )

            if self.include_component_flags:
                result_df[f"{column}_{analyzer_id}_is_ambiguous"] = analysis_results.apply(
                    lambda r: r.get("is_ambiguous")
                )
                result_df[f"{column}_{analyzer_id}_is_answerable"] = analysis_results.apply(
                    lambda r: r.get("is_answerable")
                )
                result_df[
                    f"{column}_{analyzer_id}_has_sufficient_context"
                ] = analysis_results.apply(lambda r: r.get("has_sufficient_context"))

        return result_df
