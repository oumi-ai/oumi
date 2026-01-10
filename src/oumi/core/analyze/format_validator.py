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

"""Format validation analyzer for checking data structure and schema compliance."""

from typing import Optional

import pandas as pd

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("format_validation")
class FormatValidationAnalyzer(SampleAnalyzer):
    """Analyzer that validates data format and schema compliance.

    Checks for required fields, empty values, and structural issues.
    """

    def __init__(
        self,
        *,
        required_columns: Optional[list[str]] = None,
        non_empty_columns: Optional[list[str]] = None,
        tokenizer=None,
    ):
        """Initialize the FormatValidationAnalyzer.

        Args:
            required_columns: Columns that must exist. If None, uses schema.
            non_empty_columns: Columns that must have non-empty values.
            tokenizer: Optional tokenizer (not used by this analyzer).
        """
        self.required_columns = required_columns
        self.non_empty_columns = non_empty_columns

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Validate format and structure of each row.

        Args:
            df: Input DataFrame.
            schema: Column schema dict.

        Returns:
            DataFrame with added validation columns:
            - format_missing_fields: List of missing required fields
            - format_empty_fields: List of fields with empty values
            - format_is_valid: Whether row passes all validation
        """
        result_df = df.copy()

        # Determine required columns
        required = self.required_columns or []
        if schema and not self.required_columns:
            required = [col for col in schema.keys() if col in df.columns]

        # Determine non-empty columns
        non_empty = self.non_empty_columns or required

        # Check for missing columns (same for all rows)
        missing_cols = [col for col in required if col not in df.columns]

        # Check for empty values per row
        def check_empty_fields(row: pd.Series) -> list[str]:
            empty = []
            for col in non_empty:
                if col in row.index:
                    val = row[col]
                    if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                        empty.append(col)
            return empty

        result_df["format_missing_fields"] = [missing_cols] * len(df)
        result_df["format_empty_fields"] = df.apply(check_empty_fields, axis=1)
        result_df["format_is_valid"] = result_df.apply(
            lambda row: len(row["format_missing_fields"]) == 0
            and len(row["format_empty_fields"]) == 0,
            axis=1,
        )

        return result_df, {}
