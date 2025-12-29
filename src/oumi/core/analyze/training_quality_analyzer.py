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

"""Training quality analyzer for SFT instruction datasets.

This analyzer evaluates the quality of instruction-response pairs for
supervised fine-tuning, providing metrics that predict training effectiveness.
"""

import re
from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("training_quality")
class TrainingQualityAnalyzer(SampleAnalyzer):
    """Analyzer for evaluating SFT response quality metrics.

    This analyzer computes metrics that predict how well assistant responses
    will train a language model:

    Response Completeness Metrics:
        - response_completeness_score: Composite score (0-1) for response quality
        - has_proper_ending: Whether the response ends properly (not truncated)
        - has_structure: Whether the response uses lists, code blocks, etc.
        - response_word_count: Word count for the response
    """

    # Patterns for proper endings
    _TRUNCATION_PATTERNS = [
        re.compile(r"\.\.\.$"),  # Trailing ellipsis
        re.compile(r"[^.!?)\]\"'`]$"),  # No punctuation at end
        re.compile(r"\b(?:and|or|but|the|a|an|to|of|in|for|with)\s*$", re.IGNORECASE),
    ]

    # Patterns for structured responses
    _STRUCTURE_PATTERNS = [
        re.compile(r"^\s*[-*]\s+", re.MULTILINE),  # Bullet points
        re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),  # Numbered lists
        re.compile(r"```[\s\S]*?```"),  # Code blocks
        re.compile(r"^\s*#{1,6}\s+", re.MULTILINE),  # Headers
        re.compile(r"\*\*[^*]+\*\*"),  # Bold text
        re.compile(r"^\s*>\s+", re.MULTILINE),  # Block quotes
    ]

    def __init__(
        self,
        *,
        compute_response_completeness: bool = True,
        min_response_words: int = 5,
    ):
        """Initialize the TrainingQualityAnalyzer.

        Args:
            compute_response_completeness: Whether to compute response completeness.
            min_response_words: Minimum words for a complete response.
        """
        self.compute_response_completeness = compute_response_completeness
        self.min_response_words = min_response_words

    def _compute_response_completeness(self, text: str) -> dict[str, Any]:
        """Compute response completeness metrics.

        Args:
            text: Response text to analyze.

        Returns:
            Dictionary with completeness metrics.
        """
        words = text.split()
        word_count = len(words)

        score = 1.0
        has_proper_ending = True
        has_structure = False

        # Check for truncation patterns
        text_stripped = text.strip()
        for pattern in self._TRUNCATION_PATTERNS:
            if pattern.search(text_stripped):
                has_proper_ending = False
                score -= 0.3
                break

        # Check for structure
        for pattern in self._STRUCTURE_PATTERNS:
            if pattern.search(text):
                has_structure = True
                score += 0.05  # Small bonus for structure
                break

        # Check minimum length
        if word_count < self.min_response_words:
            score -= 0.3

        # Empty or near-empty responses
        if word_count < 2:
            score = 0.0

        # Clamp score
        score = max(0.0, min(1.0, score))

        return {
            "response_completeness_score": round(score, 3),
            "has_proper_ending": has_proper_ending,
            "has_structure": has_structure,
            "response_word_count": word_count,
        }

    def _analyze_message(self, text: str, role: str) -> dict[str, Any]:
        """Analyze a single message for training quality.

        Args:
            text: Message text to analyze.
            role: Role of the message.

        Returns:
            Dictionary of training quality metrics.
        """
        results = {}
        role_lower = role.lower() if role else ""

        # Compute response completeness for assistant messages
        if self.compute_response_completeness and role_lower == "assistant":
            completeness_results = self._compute_response_completeness(text)
            results.update(completeness_results)

        return results

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze text fields for training quality metrics.

        This analyzer is role-aware:
        - Assistant messages get response completeness metrics

        Args:
            df: Input DataFrame with text fields and role column.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added training quality analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for training quality "
                "analysis. Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            return result_df, generated_schema

        # Find the role column
        role_column = None
        for col, config in schema.items():
            if (
                config.get("content_type") == ContentType.CATEGORICAL
                and col in df.columns
                and "role" in col.lower()
            ):
                role_column = col
                break

        analyzer_id = getattr(self, "analyzer_id", "training_quality")

        for column in text_columns:
            # Analyze each row with its role
            if role_column is not None:
                analysis_results = df.apply(
                    lambda row: self._analyze_message(
                        str(row[column]), str(row.get(role_column, ""))
                    ),
                    axis=1,
                )
            else:
                # No role column - analyze as generic text
                analysis_results = df[column].astype(str).apply(
                    lambda text: self._analyze_message(text, "")
                )

            # Extract response completeness metrics (for assistant messages)
            if self.compute_response_completeness:
                result_df[
                    f"{column}_{analyzer_id}_response_completeness_score"
                ] = analysis_results.apply(
                    lambda r: r.get("response_completeness_score", None)
                )
                result_df[
                    f"{column}_{analyzer_id}_has_proper_ending"
                ] = analysis_results.apply(lambda r: r.get("has_proper_ending", None))
                result_df[
                    f"{column}_{analyzer_id}_has_structure"
                ] = analysis_results.apply(lambda r: r.get("has_structure", None))
                result_df[
                    f"{column}_{analyzer_id}_response_word_count"
                ] = analysis_results.apply(
                    lambda r: r.get("response_word_count", None)
                )

        return result_df, generated_schema
