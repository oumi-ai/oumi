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

"""Repetition analyzer for detecting repeated content within samples."""

from collections import Counter
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("repetition")
class RepetitionAnalyzer(SampleAnalyzer):
    """Analyzer that detects repetition within individual samples.

    Identifies repeated words, phrases, and patterns.
    """

    def __init__(
        self,
        *,
        ngram_sizes: Optional[list[int]] = None,
        repetition_threshold: float = 0.3,
        tokenizer=None,
    ):
        """Initialize the RepetitionAnalyzer.

        Args:
            ngram_sizes: N-gram sizes to check for repetition. Default: [1, 2, 3].
            repetition_threshold: Flag samples with repetition ratio above this.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.ngram_sizes = ngram_sizes or [1, 2, 3]
        self.repetition_threshold = repetition_threshold

    def _get_ngrams(self, words: list[str], n: int) -> list[str]:
        """Extract n-grams from word list."""
        if len(words) < n:
            return []
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    def _compute_repetition_ratio(self, ngrams: list[str]) -> float:
        """Compute ratio of repeated n-grams."""
        if not ngrams:
            return 0.0
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(ngrams)

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for internal repetition.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added columns:
            - {column}_word_repetition_ratio: Ratio of repeated words
            - {column}_unique_word_ratio: Ratio of unique words
            - {column}_max_word_frequency: Frequency of most common word
            - {column}_has_excessive_repetition: Exceeds threshold
        """
        if not schema:
            raise ValueError("schema is required to identify text fields.")

        result_df = df.copy()

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            raise ValueError("No text fields found in DataFrame.")

        for column in text_columns:

            def analyze_text(text: str) -> dict:
                words = text.lower().split()
                if not words:
                    return {
                        "word_repetition_ratio": 0.0,
                        "unique_word_ratio": 0.0,
                        "max_word_frequency": 0.0,
                    }

                word_counts = Counter(words)
                unique_count = len(word_counts)
                total_count = len(words)

                # Repetition ratio (repeated occurrences / total)
                repeated = sum(c - 1 for c in word_counts.values() if c > 1)
                rep_ratio = repeated / total_count

                # Unique word ratio
                unique_ratio = unique_count / total_count

                # Max frequency
                max_freq = max(word_counts.values()) / total_count

                return {
                    "word_repetition_ratio": rep_ratio,
                    "unique_word_ratio": unique_ratio,
                    "max_word_frequency": max_freq,
                }

            analysis = df[column].astype(str).apply(analyze_text)

            result_df[f"{column}_word_repetition_ratio"] = analysis.apply(
                lambda x: x["word_repetition_ratio"]
            )
            result_df[f"{column}_unique_word_ratio"] = analysis.apply(
                lambda x: x["unique_word_ratio"]
            )
            result_df[f"{column}_max_word_frequency"] = analysis.apply(
                lambda x: x["max_word_frequency"]
            )
            result_df[f"{column}_has_excessive_repetition"] = (
                result_df[f"{column}_word_repetition_ratio"] > self.repetition_threshold
            )

        return result_df, {}
