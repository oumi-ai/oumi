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
    """

    def __init__(
        self,
        *,
        unique_words_ratio: bool = True,
        case_sensitive: bool = False,
    ):
        """Initialize the DiversityAnalyzer.

        Args:
            unique_words_ratio: Whether to compute unique words / total words ratio.
                This is the basic measure of vocabulary diversity.
            case_sensitive: Whether word comparison should be case-sensitive.
                If False (default), "Hello" and "hello" are treated as the same word.
        """
        self.unique_words_ratio = unique_words_ratio
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

    def _compute_unique_words_ratio(self, text: str) -> float:
        """Compute unique words ratio for a single text.

        Args:
            text: Input text to analyze.

        Returns:
            Ratio of unique words to total words (0.0 to 1.0).
        """
        tokens = self._tokenize(text)
        total_tokens = len(tokens)

        # Handle empty text
        if total_tokens == 0:
            return 0.0

        unique_words = len(set(tokens))
        return unique_words / total_tokens

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
            if self.unique_words_ratio:
                col_name = f"{column}_{analyzer_id}_unique_words_ratio"
                result_df[col_name] = (
                    df[column].astype(str).apply(self._compute_unique_words_ratio)
                )

        return result_df
