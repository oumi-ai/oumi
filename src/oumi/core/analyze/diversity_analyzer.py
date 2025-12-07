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

"""Diversity analyzer for measuring vocabulary diversity in text content."""

import math
from collections import Counter
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("diversity")
class DiversityAnalyzer(SampleAnalyzer):
    """Analyzer that computes vocabulary diversity metrics for text content.

    This analyzer measures how varied the vocabulary is in text samples,
    which can be useful for identifying repetitive content, assessing
    writing quality, or filtering datasets.

    Metrics computed:
        - unique_words_ratio: Number of unique words / total words
        - type_token_ratio (TTR): Same as unique_words_ratio, standard term
        - vocabulary_richness: Log-adjusted TTR for length normalization (Root TTR)
        - hapax_legomena_ratio: Words appearing only once / total unique words
    """

    def __init__(
        self,
        *,
        unique_words_ratio: bool = True,
        type_token_ratio: bool = True,
        vocabulary_richness: bool = True,
        hapax_legomena_ratio: bool = False,
        case_sensitive: bool = False,
    ):
        """Initialize the DiversityAnalyzer.

        Args:
            unique_words_ratio: Whether to compute unique words / total words ratio.
                This is the basic measure of vocabulary diversity.
            type_token_ratio: Whether to compute type-token ratio (TTR).
                Same as unique_words_ratio but uses standard linguistic terminology.
            vocabulary_richness: Whether to compute vocabulary richness score.
                Uses Root TTR (types / sqrt(tokens)) which is less sensitive
                to text length than raw TTR.
            hapax_legomena_ratio: Whether to compute hapax legomena ratio.
                Hapax legomena are words that appear only once. This ratio
                indicates how many words are used just once vs repeatedly.
            case_sensitive: Whether word comparison should be case-sensitive.
                If False (default), "Hello" and "hello" are treated as the same word.
        """
        self.unique_words_ratio = unique_words_ratio
        self.type_token_ratio = type_token_ratio
        self.vocabulary_richness = vocabulary_richness
        self.hapax_legomena_ratio = hapax_legomena_ratio
        self.case_sensitive = case_sensitive

    def _tokenize(self, text: str) -> list[str]:
        """Split text into words/tokens.

        Args:
            text: Input text to tokenize.

        Returns:
            List of words/tokens.
        """
        if not self.case_sensitive:
            text = text.lower()
        # Simple whitespace tokenization, removing empty strings
        return [word for word in text.split() if word]

    def _compute_diversity_metrics(self, text: str) -> dict[str, float]:
        """Compute all diversity metrics for a single text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary mapping metric names to values.
        """
        tokens = self._tokenize(text)
        total_tokens = len(tokens)

        # Handle empty text
        if total_tokens == 0:
            return {
                "unique_words_ratio": 0.0,
                "type_token_ratio": 0.0,
                "vocabulary_richness": 0.0,
                "hapax_legomena_ratio": 0.0,
            }

        # Count word frequencies
        word_counts = Counter(tokens)
        unique_words = len(word_counts)

        # Calculate metrics
        ttr = unique_words / total_tokens

        # Root TTR (vocabulary richness) - less sensitive to text length
        root_ttr = unique_words / math.sqrt(total_tokens)

        # Hapax legomena - words appearing only once
        hapax_count = sum(1 for count in word_counts.values() if count == 1)
        hapax_ratio = hapax_count / unique_words if unique_words > 0 else 0.0

        return {
            "unique_words_ratio": ttr,
            "type_token_ratio": ttr,
            "vocabulary_richness": root_ttr,
            "hapax_legomena_ratio": hapax_ratio,
        }

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields and return diversity metrics.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema dict to identify text fields.

        Returns:
            DataFrame with added diversity analysis columns.
        """
        result_df = df.copy()

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for diversity analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        text_columns = [
            col
            for col, config in schema.items()
            if config.get("content_type") == ContentType.TEXT and col in df.columns
        ]

        if not text_columns:
            # No text columns to analyze in this DataFrame, return unchanged
            return result_df

        # Get analyzer ID for column naming
        analyzer_id = getattr(self, "analyzer_id", "diversity")

        for column in text_columns:
            # Compute all metrics for the column
            metrics_list = df[column].astype(str).apply(self._compute_diversity_metrics)

            if self.unique_words_ratio:
                col_name = f"{column}_{analyzer_id}_unique_words_ratio"
                result_df[col_name] = metrics_list.apply(
                    lambda m: m["unique_words_ratio"]
                )

            if self.type_token_ratio:
                col_name = f"{column}_{analyzer_id}_type_token_ratio"
                result_df[col_name] = metrics_list.apply(
                    lambda m: m["type_token_ratio"]
                )

            if self.vocabulary_richness:
                col_name = f"{column}_{analyzer_id}_vocabulary_richness"
                result_df[col_name] = metrics_list.apply(
                    lambda m: m["vocabulary_richness"]
                )

            if self.hapax_legomena_ratio:
                col_name = f"{column}_{analyzer_id}_hapax_legomena_ratio"
                result_df[col_name] = metrics_list.apply(
                    lambda m: m["hapax_legomena_ratio"]
                )

        return result_df
