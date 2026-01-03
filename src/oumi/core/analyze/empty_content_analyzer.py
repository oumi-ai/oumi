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

"""Empty content analyzer for detecting empty or whitespace-only content."""

from typing import Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("empty_content")
class EmptyContentAnalyzer(SampleAnalyzer):
    """Analyzer that detects empty or whitespace-only content."""

    def __init__(self, *, min_content_length: int = 1):
        """Initialize the EmptyContentAnalyzer.

        Args:
            min_content_length: Minimum non-whitespace characters for valid content.
        """
        self.min_content_length = min_content_length

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze text fields for empty content.

        Args:
            df: Input DataFrame with text fields.
            schema: Column schema to identify text fields.

        Returns:
            DataFrame with added columns:
            - {column}_is_empty: Content is empty string
            - {column}_is_whitespace_only: Content contains only whitespace
            - {column}_has_content: Has meaningful content
            - {column}_stripped_length: Length after stripping whitespace
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

            # Check empty
            result_df[f"{column}_is_empty"] = text_series == ""

            # Check whitespace only
            stripped = text_series.str.strip()
            result_df[f"{column}_is_whitespace_only"] = (text_series != "") & (
                stripped == ""
            )

            # Stripped length
            result_df[f"{column}_stripped_length"] = stripped.str.len()

            # Has meaningful content
            result_df[f"{column}_has_content"] = (
                result_df[f"{column}_stripped_length"] >= self.min_content_length
            )

        return result_df
