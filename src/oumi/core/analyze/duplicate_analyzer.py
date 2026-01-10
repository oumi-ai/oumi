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

"""Exact duplicate analyzer for detecting duplicate samples."""

import hashlib
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("duplicate")
class DuplicateAnalyzer(SampleAnalyzer):
    """Analyzer that detects exact duplicate content in datasets.

    Computes SHA256 hash of text content and identifies duplicates.
    """

    def __init__(
        self,
        *,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
        tokenizer=None,
    ):
        """Initialize the DuplicateAnalyzer.

        Args:
            normalize_whitespace: Collapse multiple whitespace to single space.
            case_sensitive: If False, convert to lowercase before hashing.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing."""
        if not self.case_sensitive:
            text = text.lower()
        if self.normalize_whitespace:
            text = " ".join(text.split())
        return text

    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of normalized text."""
        normalized = self._normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for duplicates.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added duplicate detection columns:
            - {column}_hash: SHA256 hash of content
            - {column}_is_duplicate: Whether this content has duplicates
            - {column}_duplicate_count: Number of occurrences of this content
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
            # Compute hash for each row
            hashes = df[column].astype(str).apply(self._compute_hash)
            result_df[f"{column}_hash"] = hashes

            # Count occurrences of each hash
            hash_counts = hashes.value_counts()

            # Map counts back to rows
            result_df[f"{column}_duplicate_count"] = hashes.map(hash_counts)

            # Flag duplicates (count > 1)
            result_df[f"{column}_is_duplicate"] = result_df[f"{column}_duplicate_count"] > 1

        return result_df, {}
