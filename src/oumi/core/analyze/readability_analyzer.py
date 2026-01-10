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

"""Readability analyzer for computing text complexity metrics."""

import re
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("readability")
class ReadabilityAnalyzer(SampleAnalyzer):
    """Analyzer that computes readability metrics for text content.

    Implements standard readability formulas without external dependencies.
    """

    def __init__(
        self,
        *,
        flesch_reading_ease: bool = True,
        flesch_kincaid_grade: bool = True,
        avg_sentence_length: bool = True,
        avg_word_length: bool = True,
        tokenizer=None,
    ):
        """Initialize the ReadabilityAnalyzer.

        Args:
            flesch_reading_ease: Compute Flesch Reading Ease score.
            flesch_kincaid_grade: Compute Flesch-Kincaid Grade Level.
            avg_sentence_length: Compute average sentence length.
            avg_word_length: Compute average word length.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.flesch_reading_ease = flesch_reading_ease
        self.flesch_kincaid_grade = flesch_kincaid_grade
        self.avg_sentence_length = avg_sentence_length
        self.avg_word_length = avg_word_length

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word.

        Uses a simple heuristic based on vowel groups.
        """
        word = word.lower().strip()
        if not word:
            return 0

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e") and count > 1:
            count -= 1

        # Adjust for -le endings
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(1, count)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_words(self, text: str) -> list[str]:
        """Split text into words."""
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        return words

    def _compute_metrics(self, text: str) -> dict[str, float]:
        """Compute all readability metrics for a text."""
        sentences = self._split_sentences(text)
        words = self._split_words(text)

        num_sentences = max(1, len(sentences))
        num_words = max(1, len(words))
        num_syllables = sum(self._count_syllables(w) for w in words)

        avg_sent_len = num_words / num_sentences
        avg_syll_per_word = num_syllables / num_words
        avg_word_len = sum(len(w) for w in words) / num_words if words else 0

        metrics = {}

        if self.flesch_reading_ease:
            # Flesch Reading Ease: 206.835 - 1.015 * ASL - 84.6 * ASW
            fre = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syll_per_word
            metrics["flesch_reading_ease"] = max(0, min(100, fre))

        if self.flesch_kincaid_grade:
            # Flesch-Kincaid Grade: 0.39 * ASL + 11.8 * ASW - 15.59
            fkg = 0.39 * avg_sent_len + 11.8 * avg_syll_per_word - 15.59
            metrics["flesch_kincaid_grade"] = max(0, fkg)

        if self.avg_sentence_length:
            metrics["avg_sentence_length"] = avg_sent_len

        if self.avg_word_length:
            metrics["avg_word_length"] = avg_word_len

        return metrics

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for readability.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added readability columns:
            - {column}_flesch_reading_ease: Flesch Reading Ease (0-100)
            - {column}_flesch_kincaid_grade: Grade level
            - {column}_avg_sentence_length: Words per sentence
            - {column}_avg_word_length: Characters per word
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
            metrics_list = df[column].astype(str).apply(self._compute_metrics)

            for metric_name in [
                "flesch_reading_ease",
                "flesch_kincaid_grade",
                "avg_sentence_length",
                "avg_word_length",
            ]:
                if any(metric_name in m for m in metrics_list):
                    result_df[f"{column}_{metric_name}"] = metrics_list.apply(
                        lambda x: x.get(metric_name, 0)
                    )

        return result_df, {}
