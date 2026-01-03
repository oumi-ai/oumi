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

"""N-gram analyzer for detecting overrepresented phrases across dataset."""

import math
from collections import Counter
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("ngram")
class NgramAnalyzer(SampleAnalyzer):
    """Analyzer that detects overrepresented n-grams across the dataset.

    Computes n-gram frequencies and flags samples containing overused phrases.
    """

    def __init__(
        self,
        *,
        n: int = 3,
        min_document_frequency: float = 0.05,
        top_k: int = 50,
        case_sensitive: bool = False,
    ):
        """Initialize the NgramAnalyzer.

        Args:
            n: Size of n-grams to analyze.
            min_document_frequency: Flag n-grams appearing in more than this
                fraction of samples.
            top_k: Number of top n-grams to track.
            case_sensitive: Whether to use case-sensitive matching.
        """
        self.n = n
        self.min_document_frequency = min_document_frequency
        self.top_k = top_k
        self.case_sensitive = case_sensitive

    def _extract_ngrams(self, text: str) -> list[str]:
        """Extract word n-grams from text."""
        if not self.case_sensitive:
            text = text.lower()
        words = text.split()
        if len(words) < self.n:
            return []
        return [" ".join(words[i : i + self.n]) for i in range(len(words) - self.n + 1)]

    def _compute_entropy(self, counts: Counter) -> float:
        """Compute Shannon entropy of n-gram distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def _compute_gini(self, counts: Counter) -> float:
        """Compute Gini coefficient of n-gram distribution."""
        values = sorted(counts.values())
        n = len(values)
        if n == 0 or sum(values) == 0:
            return 0.0
        cumsum = 0
        total = sum(values)
        gini_sum = 0
        for i, v in enumerate(values):
            cumsum += v
            gini_sum += cumsum
        return 1 - (2 * gini_sum) / (n * total) + 1 / n

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for n-gram frequency.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added columns:
            - {column}_ngram_count: Number of n-grams in sample
            - {column}_unique_ngram_ratio: Ratio of unique n-grams
            - {column}_overrepresented_count: Count of overrepresented n-grams
            - {column}_contains_overrepresented: Has any overrepresented n-grams
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
            texts = df[column].astype(str)
            num_samples = len(texts)

            # Extract n-grams for each sample
            sample_ngrams = texts.apply(self._extract_ngrams)

            # Count n-gram occurrences and document frequency
            ngram_counts = Counter()
            doc_frequency = Counter()

            for ngrams in sample_ngrams:
                ngram_counts.update(ngrams)
                doc_frequency.update(set(ngrams))  # Count each n-gram once per doc

            # Find overrepresented n-grams
            overrepresented = {
                ngram
                for ngram, count in doc_frequency.items()
                if count / num_samples > self.min_document_frequency
            }

            # Per-sample metrics
            result_df[f"{column}_ngram_count"] = sample_ngrams.apply(len)

            result_df[f"{column}_unique_ngram_ratio"] = sample_ngrams.apply(
                lambda x: len(set(x)) / len(x) if x else 0.0
            )

            result_df[f"{column}_overrepresented_count"] = sample_ngrams.apply(
                lambda x: sum(1 for ng in x if ng in overrepresented)
            )

            result_df[f"{column}_contains_overrepresented"] = (
                result_df[f"{column}_overrepresented_count"] > 0
            )

        return result_df
