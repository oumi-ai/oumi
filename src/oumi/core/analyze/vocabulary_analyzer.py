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

"""Vocabulary analyzer for measuring lexical diversity."""

from collections import Counter
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("vocabulary")
class VocabularyAnalyzer(SampleAnalyzer):
    """Analyzer that measures vocabulary richness and diversity.

    Computes type-token ratio, hapax legomena, and other lexical metrics.
    """

    def __init__(
        self,
        *,
        case_sensitive: bool = False,
        min_word_length: int = 1,
        tokenizer=None,
    ):
        """Initialize the VocabularyAnalyzer.

        Args:
            case_sensitive: Whether to treat words case-sensitively.
            min_word_length: Minimum word length to consider.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.case_sensitive = case_sensitive
        self.min_word_length = min_word_length

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        if not self.case_sensitive:
            text = text.lower()
        words = text.split()
        return [w for w in words if len(w) >= self.min_word_length]

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for vocabulary metrics.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added columns:
            - {column}_vocabulary_size: Number of unique words
            - {column}_type_token_ratio: Unique words / total words
            - {column}_hapax_count: Words appearing exactly once
            - {column}_hapax_ratio: Hapax words / total words
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

            def analyze_vocabulary(text: str) -> dict:
                words = self._tokenize(text)
                if not words:
                    return {
                        "vocabulary_size": 0,
                        "type_token_ratio": 0.0,
                        "hapax_count": 0,
                        "hapax_ratio": 0.0,
                    }

                word_counts = Counter(words)
                vocab_size = len(word_counts)
                total_words = len(words)

                # Hapax legomena: words appearing exactly once
                hapax = sum(1 for count in word_counts.values() if count == 1)

                return {
                    "vocabulary_size": vocab_size,
                    "type_token_ratio": vocab_size / total_words,
                    "hapax_count": hapax,
                    "hapax_ratio": hapax / total_words,
                }

            analysis = df[column].astype(str).apply(analyze_vocabulary)

            result_df[f"{column}_vocabulary_size"] = analysis.apply(
                lambda x: x["vocabulary_size"]
            )
            result_df[f"{column}_type_token_ratio"] = analysis.apply(
                lambda x: x["type_token_ratio"]
            )
            result_df[f"{column}_hapax_count"] = analysis.apply(
                lambda x: x["hapax_count"]
            )
            result_df[f"{column}_hapax_ratio"] = analysis.apply(
                lambda x: x["hapax_ratio"]
            )

        return result_df, {}
