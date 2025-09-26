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
        conversations_df: Optional[pd.DataFrame] = None,
        messages_df: Optional[pd.DataFrame] = None,
        analysis_results: Optional[Any] = None,  # DatasetAnalysisResult
        sample_analyzers: Optional[dict[str, Any]] = None,
        # Deprecated parameters for backward compatibility
        items_df: Optional[pd.DataFrame] = None,
        rows_df: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive summary of dataset analysis results.

        This method aggregates metrics from all analyzers to provide insights useful
        for assessing datasets. It computes statistics like averages,
        standard deviations, min/max values, and efficiency metrics.

        Args:
            analysis_df: Merged analysis DataFrame
            conversations_df: Conversations-level DataFrame
            messages_df: Messages-level DataFrame
            analysis_results: Analysis results metadata
            sample_analyzers: Dictionary of sample analyzers used
            items_df: Deprecated, use conversations_df instead
            rows_df: Deprecated, use messages_df instead

        Returns:
            Dictionary containing comprehensive dataset analysis summary with:
            - Dataset overview statistics
            - Message-level aggregated metrics
            - Conversation-level aggregated metrics
        """
        # Handle backward compatibility
        if conversations_df is None:
            conversations_df = items_df
        if messages_df is None:
            messages_df = rows_df
        if sample_analyzers is None:
            sample_analyzers = {}
        # Check if we have data to analyze
        if analysis_df is None or analysis_df.empty:
            return {"error": "No analysis data available"}

        summary = {
            "dataset_overview": self.get_dataset_overview(
                analysis_results, messages_df, sample_analyzers
            ),
            "row_level_summary": self.get_message_level_summary(messages_df),
            "item_level_summary": self.get_conversation_level_summary(conversations_df),
            "item_turns": self.get_conversation_turns_summary(messages_df),
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

    def get_message_level_summary(
        self, messages_df: Optional[pd.DataFrame]
    ) -> dict[str, Any]:
        """Get aggregated message-level metrics across all analyzers.

        Args:
            messages_df: Messages-level DataFrame with analysis results

        Returns:
            Dictionary with message-level summary statistics
        """
        if messages_df is None or messages_df.empty:
            return {}

        # Get all message-level analyzer columns
        message_columns = [
            col for col in messages_df.columns if col.startswith(("message_", "row_"))
        ]

        summary = {}

        for col in message_columns:
            if col in [
                "row_index",
                "message_index",
                "content",
                "item_index",
            ]:
                continue

            # Extract analyzer name and metric from column
            # Format: message_{analyzer}_{metric} or row_{analyzer}_{metric}
            # (backward compatibility)
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(messages_df[col]):
                    values = cast(pd.Series, messages_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self.decimal_precision
                        )

        return summary

    def get_conversation_level_summary(
        self, conversations_df: Optional[pd.DataFrame]
    ) -> dict[str, Any]:
        """Get aggregated conversation-level metrics across all analyzers.

        Args:
            conversations_df: Conversations-level DataFrame with analysis results

        Returns:
            Dictionary with conversation-level summary statistics
        """
        if conversations_df is None or conversations_df.empty:
            return {}

        # Get all conversation-level analyzer columns
        conversation_columns = [
            col
            for col in conversations_df.columns
            if col.startswith(("conversation_", "item_"))
        ]

        summary = {}

        for col in conversation_columns:
            if col in ["item_index", "conversation_index"]:
                continue

            # Extract analyzer name and metric from column
            # Format: conversation_{analyzer}_{metric} or item_{analyzer}_{metric}
            # (backward compatibility)
            parts = col.split("_", 2)
            if len(parts) >= 3:
                analyzer_name = parts[1]
                metric_name = "_".join(parts[2:])

                if analyzer_name not in summary:
                    summary[analyzer_name] = {}

                # Compute statistics for numeric columns
                if pd.api.types.is_numeric_dtype(conversations_df[col]):
                    values = cast(pd.Series, conversations_df[col].dropna())
                    if len(values) > 0:
                        summary[analyzer_name][metric_name] = compute_statistics(
                            values, self.decimal_precision
                        )

        return summary

    def get_conversation_turns_summary(
        self, messages_df: Optional[pd.DataFrame]
    ) -> dict[str, Any]:
        """Get conversation turn statistics summary.

        This is useful for conversation-like data where you want to know
        the distribution of turns (messages) per conversation.

        Args:
            messages_df: Messages-level DataFrame

        Returns:
            Dictionary containing conversation turn statistics
        """
        if messages_df is None or messages_df.empty:
            return {}

        # Use item_index for grouping
        if "item_index" not in messages_df.columns:
            return {}

        # groupby().size() always returns a Series, but we cast it because
        # type checker can't infer this
        turns_per_conversation = cast(
            pd.Series, messages_df.groupby("item_index").size()
        )
        return compute_statistics(turns_per_conversation, self.decimal_precision)

    # Backward compatibility methods
    def get_row_level_summary(self, rows_df: Optional[pd.DataFrame]) -> dict[str, Any]:
        """Deprecated: Use get_message_level_summary instead."""
        return self.get_message_level_summary(rows_df)

    def get_item_level_summary(
        self, items_df: Optional[pd.DataFrame]
    ) -> dict[str, Any]:
        """Deprecated: Use get_conversation_level_summary instead."""
        return self.get_conversation_level_summary(items_df)

    def get_item_turns_summary(self, rows_df: Optional[pd.DataFrame]) -> dict[str, Any]:
        """Deprecated: Use get_conversation_turns_summary instead."""
        return self.get_conversation_turns_summary(rows_df)
