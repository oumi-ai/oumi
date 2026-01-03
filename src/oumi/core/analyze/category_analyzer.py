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

"""Category distribution analyzer for analyzing labeled data distributions."""

import math
from typing import Optional

import pandas as pd

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("category_distribution")
class CategoryDistributionAnalyzer(SampleAnalyzer):
    """Analyzer that examines distribution of categorical labels.

    Computes distribution metrics and identifies imbalanced categories.
    """

    def __init__(
        self,
        *,
        category_column: str,
        expected_categories: Optional[list[str]] = None,
        min_percentage: float = 0.01,
        max_percentage: float = 0.50,
    ):
        """Initialize the CategoryDistributionAnalyzer.

        Args:
            category_column: Column containing category labels.
            expected_categories: Optional list of expected categories.
            min_percentage: Flag categories below this percentage.
            max_percentage: Flag categories above this percentage.
        """
        self.category_column = category_column
        self.expected_categories = expected_categories
        self.min_percentage = min_percentage
        self.max_percentage = max_percentage

    def _compute_entropy(self, counts: dict[str, int]) -> float:
        """Compute Shannon entropy of distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def _compute_gini(self, counts: dict[str, int]) -> float:
        """Compute Gini coefficient of distribution."""
        values = sorted(counts.values())
        n = len(values)
        if n == 0:
            return 0.0
        total = sum(values)
        if total == 0:
            return 0.0
        cumsum = 0
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
        """Analyze category distribution.

        Args:
            df: Input DataFrame with category column.
            schema: Column schema (optional).

        Returns:
            DataFrame with added columns:
            - category_count: Count of this category in dataset
            - category_percentage: Percentage of dataset
            - category_is_underrepresented: Below min threshold
            - category_is_overrepresented: Above max threshold
        """
        result_df = df.copy()

        if self.category_column not in df.columns:
            raise ValueError(
                f"Category column '{self.category_column}' not found in DataFrame."
            )

        categories = df[self.category_column].astype(str)
        category_counts = categories.value_counts().to_dict()
        total = len(categories)

        # Map counts and percentages back to rows
        result_df["category_count"] = categories.map(category_counts)
        result_df["category_percentage"] = result_df["category_count"] / total

        # Flag under/over-represented
        result_df["category_is_underrepresented"] = (
            result_df["category_percentage"] < self.min_percentage
        )
        result_df["category_is_overrepresented"] = (
            result_df["category_percentage"] > self.max_percentage
        )

        # Check for missing expected categories
        if self.expected_categories:
            found_categories = set(category_counts.keys())
            missing = set(self.expected_categories) - found_categories
            result_df["category_is_missing_expected"] = False
            # This is dataset-level info, but we add it for completeness
            if missing:
                result_df.attrs["missing_categories"] = list(missing)

        return result_df
