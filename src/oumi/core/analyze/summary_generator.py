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

from typing import Any, Optional, cast

import pandas as pd

from oumi.utils.analysis_utils import compute_statistics


class SummaryGenerator:
    """Generates comprehensive analysis summaries from processed DataFrames.

    This class handles all summary generation logic, including:
    - Dataset overview statistics
    - Row-level aggregated metrics
    - Item-level aggregated metrics
    - Turn statistics for conversation-like data
    """

    def __init__(self, decimal_precision: int = 2):
        """Initialize the summary generator.

        Args:
            decimal_precision: Number of decimal places for rounding metrics
        """
        self.decimal_precision = decimal_precision

    def generate_analysis_summary(
        self,
        analysis_df: Optional[pd.DataFrame],
        items_df: Optional[pd.DataFrame],
        rows_df: Optional[pd.DataFrame],
        analysis_results: Optional[Any],  # DatasetAnalysisResult
        sample_analyzers: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a comprehensive summary of dataset analysis results.

        This method aggregates metrics from all analyzers to provide insights useful
        for assessing datasets. It computes statistics like averages,
        standard deviations, min/max values, and efficiency metrics.

        Args:
            analysis_df: Merged analysis DataFrame
            items_df: Items-level DataFrame
            rows_df: Rows-level DataFrame
            analysis_results: Analysis results metadata
            sample_analyzers: Dictionary of sample analyzers used

        Returns:
            Dictionary containing comprehensive dataset analysis summary with:
            - Dataset overview statistics
            - Row-level aggregated metrics
            - Item-level aggregated metrics
        """
        # Check if we have data to analyze
        if analysis_df is None or analysis_df.empty:
            return {"error": "No analysis data available"}

        summary = {
            "dataset_overview": self.get_dataset_overview(
                analysis_results, rows_df, sample_analyzers
            ),
            "row_level_summary": self.get_row_level_summary(rows_df),
            "item_level_summary": self.get_item_level_summary(items_df),
            "item_turns": self.get_item_turns_summary(rows_df),
        }

        return summary

    def get_dataset_overview(
        self,
        analysis_results: Optional[Any],  # DatasetAnalysisResult
        rows_df: Optional[pd.DataFrame],
        sample_analyzers: dict[str, Any],
    ) -> dict[str, Any]:
        """Get basic dataset overview statistics.

        Args:
            analysis_results: Analysis results metadata
            rows_df: Rows-level DataFrame
            sample_analyzers: Dictionary of sample analyzers used

        Returns:
            Dictionary with dataset overview statistics
        """
        if analysis_results is None:
            return {}

        return {
            "dataset_name": analysis_results.dataset_name,
            "total_conversations": analysis_results.total_conversations,
            "conversations_analyzed": analysis_results.conversations_analyzed,
            "dataset_coverage_percentage": round(
                100.0
                * analysis_results.conversations_analyzed
                / analysis_results.total_conversations
                if analysis_results.total_conversations > 0
                else 0,
                self.decimal_precision,
            ),
            "total_rows": len(rows_df) if rows_df is not None else 0,
            "analyzers_used": list(sample_analyzers.keys()),
        }

    def get_row_level_summary(self, rows_df: Optional[pd.DataFrame]) -> dict[str, Any]:
        """Get aggregated row-level metrics across all analyzers.

        Args:
            rows_df: Rows-level DataFrame with analysis results

        Returns:
            Dictionary with row-level summary statistics
        """
        if rows_df is None or rows_df.empty:
            return {}

        # Get all row-level analyzer columns
        row_columns = [col for col in rows_df.columns if col.startswith("row_")]

        summary = {}

        for col in row_columns:
            if col in [
                "row_index",
                "content",
                "item_index",
            ]:
                continue

            # Extract analyzer name and metric from column
            # Format: row_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(rows_df[col]):
                    values = cast(pd.Series, rows_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self.decimal_precision
                        )

        return summary

    def get_item_level_summary(
        self, items_df: Optional[pd.DataFrame]
    ) -> dict[str, Any]:
        """Get aggregated item-level metrics across all analyzers.

        Args:
            items_df: Items-level DataFrame with analysis results

        Returns:
            Dictionary with item-level summary statistics
        """
        if items_df is None or items_df.empty:
            return {}

        # Get all item-level analyzer columns
        item_columns = [col for col in items_df.columns if col.startswith("item_")]

        summary = {}

        for col in item_columns:
            if col in ["item_index"]:
                continue

            # Extract analyzer name and metric from column
            # Format: item_{analyzer}_{metric}
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(items_df[col]):
                    values = cast(pd.Series, items_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self.decimal_precision
                        )

        return summary

    def get_item_turns_summary(self, rows_df: Optional[pd.DataFrame]) -> dict[str, Any]:
        """Get item turn statistics summary.

        This is useful for conversation-like data where you want to know
        the distribution of turns (messages) per item (conversation).

        Args:
            rows_df: Rows-level DataFrame

        Returns:
            Dictionary containing item turn statistics
        """
        if rows_df is None or rows_df.empty:
            return {}

        # Use item_index for grouping
        if "item_index" not in rows_df.columns:
            return {}

        # groupby().size() always returns a Series, but we cast it because
        # type checker can't infer this
        turns_per_item = cast(pd.Series, rows_df.groupby("item_index").size())
        return compute_statistics(turns_per_item, self.decimal_precision)
