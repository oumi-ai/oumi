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

"""Encoding analyzer for detecting text encoding issues."""

import unicodedata
from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("encoding")
class EncodingAnalyzer(SampleAnalyzer):
    """Analyzer that detects text encoding issues.

    Identifies replacement characters, control characters, and other encoding problems.
    """

    # Unicode replacement character (appears when decoding fails)
    REPLACEMENT_CHAR = "\ufffd"

    # Control characters to flag (C0 controls except tab/newline/carriage return)
    ALLOWED_CONTROL = {"\t", "\n", "\r"}

    def __init__(self, *, tokenizer=None):
        """Initialize the EncodingAnalyzer.

        Args:
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        pass

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for encoding issues.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added columns:
            - {column}_has_replacement_chars: Contains U+FFFD
            - {column}_replacement_char_count: Count of replacement characters
            - {column}_control_char_count: Count of problematic control characters
            - {column}_has_encoding_issues: Any encoding problems detected
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
            text_series = df[column].astype(str)

            # Count replacement characters
            replacement_counts = text_series.str.count(self.REPLACEMENT_CHAR)
            result_df[f"{column}_replacement_char_count"] = replacement_counts
            result_df[f"{column}_has_replacement_chars"] = replacement_counts > 0

            # Count control characters
            control_counts = text_series.apply(self._count_control_chars)
            result_df[f"{column}_control_char_count"] = control_counts

            # Overall encoding issues flag
            result_df[f"{column}_has_encoding_issues"] = (replacement_counts > 0) | (
                control_counts > 0
            )

        return result_df, {}

    def _count_control_chars(self, text: str) -> int:
        """Count problematic control characters in text."""
        count = 0
        for char in text:
            if char in self.ALLOWED_CONTROL:
                continue
            category = unicodedata.category(char)
            # Cc = control, Cf = format (some are okay), Co = private use
            if category in ("Cc", "Co"):
                count += 1
        return count
