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

"""Instruct reward analyzer for scoring response quality.

Based on the Magpie/ArmoRM framework from "Fixing It in Post" paper,
this analyzer scores response quality on a 0-5 scale to help identify
high-quality samples for dataset curation.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("instruct_reward")
class InstructRewardAnalyzer(SampleAnalyzer):
    """Analyzer for scoring response quality (instruct reward).

    This analyzer evaluates assistant responses on multiple quality dimensions
    to produce an overall reward score (0-5 scale) similar to the Magpie framework.

    Quality dimensions assessed:
        - Helpfulness: Does the response address the instruction?
        - Completeness: Is the response thorough and complete?
        - Accuracy: Does the response appear factually sound?
        - Clarity: Is the response well-organized and clear?
        - Safety: Is the response appropriate and safe?

    Metrics computed:
        - reward_score: Overall quality score (0-5 scale, continuous)
        - reward_tier: Quality tier (poor, fair, good, excellent)
        - helpfulness_score: How well it addresses the instruction (0-1)
        - completeness_score: How thorough the response is (0-1)
        - clarity_score: How clear and organized (0-1)
    """

    # Patterns indicating helpful responses
    _HELPFUL_PATTERNS = [
        re.compile(
            r"^(?:here|let\s+me|I'll|I\s+will|sure|of\s+course|"
            r"certainly|absolutely|yes)",
            re.IGNORECASE | re.MULTILINE,
        ),
        re.compile(
            r"\b(?:to\s+(?:help|assist|address|answer)\s+"
            r"(?:your|this|the)\s+(?:question|request|query))\b",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating incomplete/unhelpful responses
    _UNHELPFUL_PATTERNS = [
        re.compile(
            r"^\s*(?:I\s+(?:don't|cannot|can't|am\s+not\s+able)\s+"
            r"(?:know|help|provide|answer))",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:I'm\s+not\s+sure|I\s+don't\s+have\s+(?:enough|"
            r"the)\s+information)\b",
            re.IGNORECASE,
        ),
        re.compile(r"^\s*(?:N/?A|None|-)s*$", re.IGNORECASE),
    ]

    # Patterns indicating structured/clear responses
    _CLARITY_PATTERNS = [
        re.compile(r"^\s*[-*]\s+", re.MULTILINE),  # Bullet points
        re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),  # Numbered lists
        re.compile(r"^\s*#{1,6}\s+", re.MULTILINE),  # Headers
        re.compile(r"```[\s\S]*?```"),  # Code blocks
        re.compile(r"\*\*[^*]+\*\*"),  # Bold text
        re.compile(
            r"\b(?:first|second|third|finally|in\s+conclusion|"
            r"to\s+summarize|in\s+summary)\b",
            re.IGNORECASE,
        ),
    ]

    # Patterns indicating uncertain/hedging language
    _HEDGING_PATTERNS = [
        re.compile(
            r"\b(?:might|maybe|perhaps|possibly|probably|"
            r"could\s+be|seems?\s+like|appears?\s+to)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:I\s+think|I\s+believe|in\s+my\s+opinion)\b", re.IGNORECASE),
    ]

    # Patterns indicating unsafe/inappropriate content
    _UNSAFE_PATTERNS = [
        re.compile(
            r"\b(?:kill|murder|harm|hurt|attack|assault|"
            r"weapon|bomb|explosive)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:illegal|illicit|unlawful|criminal|"
            r"hack|exploit|steal)\b",
            re.IGNORECASE,
        ),
    ]

    def __init__(
        self,
        *,
        min_response_words: int = 10,
        max_response_words: int = 2000,
        analyze_assistant_only: bool = True,
        include_component_scores: bool = True,
    ):
        """Initialize the InstructRewardAnalyzer.

        Args:
            min_response_words: Minimum words for a quality response.
            max_response_words: Maximum words before length penalty.
            analyze_assistant_only: If True, only analyze assistant messages.
            include_component_scores: Include individual dimension scores.
        """
        self.min_response_words = min_response_words
        self.max_response_words = max_response_words
        self.analyze_assistant_only = analyze_assistant_only
        self.include_component_scores = include_component_scores

    def _compute_helpfulness(self, text: str) -> float:
        """Compute helpfulness score.

        Args:
            text: Response text.

        Returns:
            Helpfulness score (0-1).
        """
        score = 0.5  # Start neutral

        # Check for helpful patterns
        for pattern in self._HELPFUL_PATTERNS:
            if pattern.search(text):
                score += 0.15

        # Check for unhelpful patterns
        for pattern in self._UNHELPFUL_PATTERNS:
            if pattern.search(text):
                score -= 0.3

        return max(0.0, min(1.0, score))

    def _compute_completeness(self, text: str) -> float:
        """Compute completeness score.

        Args:
            text: Response text.

        Returns:
            Completeness score (0-1).
        """
        words = text.split()
        word_count = len(words)

        # Base score from length
        if word_count < self.min_response_words:
            base_score = word_count / self.min_response_words * 0.5
        elif word_count > self.max_response_words:
            # Slight penalty for very long responses
            base_score = 0.9
        else:
            # Optimal range
            base_score = 0.7 + (0.3 * min(1.0, word_count / 200))

        # Check for proper ending
        text_stripped = text.strip()
        if text_stripped and text_stripped[-1] in ".!?)]\"'":
            base_score += 0.1
        elif text_stripped.endswith("..."):
            base_score -= 0.2

        return max(0.0, min(1.0, base_score))

    def _compute_clarity(self, text: str) -> float:
        """Compute clarity score.

        Args:
            text: Response text.

        Returns:
            Clarity score (0-1).
        """
        score = 0.5  # Start neutral

        # Check for structured content
        structure_count = 0
        for pattern in self._CLARITY_PATTERNS:
            if pattern.search(text):
                structure_count += 1

        # More structure = more clarity (up to a point)
        score += min(0.3, structure_count * 0.1)

        # Check sentence structure
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if sentences:
            # Average sentence length (prefer medium-length sentences)
            avg_words_per_sentence = len(text.split()) / len(sentences)
            if 10 <= avg_words_per_sentence <= 25:
                score += 0.2
            elif avg_words_per_sentence < 5 or avg_words_per_sentence > 50:
                score -= 0.1

        # Penalize excessive hedging
        hedge_count = sum(
            len(pattern.findall(text)) for pattern in self._HEDGING_PATTERNS
        )
        if hedge_count > 5:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _compute_safety(self, text: str) -> float:
        """Compute safety score.

        Args:
            text: Response text.

        Returns:
            Safety score (0-1).
        """
        score = 1.0  # Start safe

        # Check for unsafe patterns
        for pattern in self._UNSAFE_PATTERNS:
            matches = pattern.findall(text)
            score -= len(matches) * 0.1

        return max(0.0, min(1.0, score))

    def _compute_reward_score(
        self,
        helpfulness: float,
        completeness: float,
        clarity: float,
        safety: float,
    ) -> float:
        """Compute overall reward score (0-5 scale).

        Args:
            helpfulness: Helpfulness score (0-1).
            completeness: Completeness score (0-1).
            clarity: Clarity score (0-1).
            safety: Safety score (0-1).

        Returns:
            Overall reward score (0-5).
        """
        # Weighted combination (safety is critical)
        weights = {
            "helpfulness": 0.3,
            "completeness": 0.25,
            "clarity": 0.2,
            "safety": 0.25,
        }

        combined = (
            weights["helpfulness"] * helpfulness
            + weights["completeness"] * completeness
            + weights["clarity"] * clarity
            + weights["safety"] * safety
        )

        # Scale to 0-5
        return round(combined * 5, 2)

    def _get_reward_tier(self, score: float) -> str:
        """Convert reward score to tier label.

        Args:
            score: Reward score (0-5).

        Returns:
            Tier label.
        """
        if score >= 4.0:
            return "excellent"
        elif score >= 3.0:
            return "good"
        elif score >= 2.0:
            return "fair"
        else:
            return "poor"

    def _analyze_response(self, text: str) -> dict[str, Any]:
        """Analyze a response for reward metrics.

        Args:
            text: Response text.

        Returns:
            Dictionary of reward metrics.
        """
        if not text or not text.strip():
            return {
                "reward_score": 0.0,
                "reward_tier": "poor",
                "helpfulness_score": 0.0,
                "completeness_score": 0.0,
                "clarity_score": 0.0,
            }

        helpfulness = self._compute_helpfulness(text)
        completeness = self._compute_completeness(text)
        clarity = self._compute_clarity(text)
        safety = self._compute_safety(text)

        reward_score = self._compute_reward_score(
            helpfulness, completeness, clarity, safety
        )

        result = {
            "reward_score": reward_score,
            "reward_tier": self._get_reward_tier(reward_score),
        }

        if self.include_component_scores:
            result["helpfulness_score"] = round(helpfulness, 3)
            result["completeness_score"] = round(completeness, 3)
            result["clarity_score"] = round(clarity, 3)

        return result

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for instruct reward metrics.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added instruct reward analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for instruct reward "
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
        if self.analyze_assistant_only:
            for col, config in schema.items():
                if (
                    config.get("content_type") == ContentType.CATEGORICAL
                    and col in df.columns
                    and "role" in col.lower()
                ):
                    role_column = col
                    break

        analyzer_id = getattr(self, "analyzer_id", "instruct_reward")

        for column in text_columns:
            if self.analyze_assistant_only and role_column is not None:
                # Only analyze assistant messages
                analysis_results = df.apply(
                    lambda row: (
                        self._analyze_response(str(row[column]))
                        if str(row.get(role_column, "")).lower() == "assistant"
                        else {
                            "reward_score": None,
                            "reward_tier": None,
                            "helpfulness_score": None,
                            "completeness_score": None,
                            "clarity_score": None,
                        }
                    ),
                    axis=1,
                )
            else:
                analysis_results = df[column].astype(str).apply(self._analyze_response)

            # Extract results to columns
            result_df[f"{column}_{analyzer_id}_score"] = analysis_results.apply(
                lambda r: r.get("reward_score")
            )
            result_df[f"{column}_{analyzer_id}_tier"] = analysis_results.apply(
                lambda r: r.get("reward_tier")
            )

            if self.include_component_scores:
                result_df[
                    f"{column}_{analyzer_id}_helpfulness"
                ] = analysis_results.apply(lambda r: r.get("helpfulness_score"))
                result_df[
                    f"{column}_{analyzer_id}_completeness"
                ] = analysis_results.apply(lambda r: r.get("completeness_score"))
                result_df[f"{column}_{analyzer_id}_clarity"] = analysis_results.apply(
                    lambda r: r.get("clarity_score")
                )

        return result_df
