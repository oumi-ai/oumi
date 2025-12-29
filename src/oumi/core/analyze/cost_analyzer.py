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

"""Cost analyzer for training optimization insights.

This analyzer provides metrics related to training cost optimization,
including context window utilization and packing efficiency.
"""

from typing import Any, Optional

import pandas as pd

from oumi.core.analyze.column_types import ContentType
from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer


@register_sample_analyzer("cost")
class CostAnalyzer(SampleAnalyzer):
    """Analyzer for computing training cost optimization metrics.

    This analyzer helps users understand how their dataset will utilize
    model context windows and provides insights for cost optimization:

    Context Window Metrics:
        - fits_context_{size}: Whether each sample fits in the context window
        - context_utilization_{size}: Fraction of context window used (0-1)
        - tokens_wasted_{size}: Tokens wasted due to incomplete packing

    Packing Metrics:
        - packing_efficiency: How efficiently samples can be packed together
        - estimated_batches: Estimated number of batches needed

    Example:
        >>> analyzer = CostAnalyzer(
        ...     target_context_windows=[4096, 8192],
        ...     compute_packing_efficiency=True,
        ... )
        >>> result = analyzer.analyze_sample(df, schema)
    """

    # Common context window sizes for reference
    DEFAULT_CONTEXT_WINDOWS = [2048, 4096, 8192, 16384, 32768]

    def __init__(
        self,
        *,
        target_context_windows: Optional[list[int]] = None,
        compute_packing_efficiency: bool = True,
        packing_overhead_tokens: int = 10,
    ):
        """Initialize the CostAnalyzer.

        Args:
            target_context_windows: List of context window sizes to analyze.
                Defaults to [2048, 4096, 8192, 16384, 32768].
            compute_packing_efficiency: Whether to compute packing efficiency.
                Defaults to True.
            packing_overhead_tokens: Estimated overhead tokens per sample
                (for separator tokens, special tokens, etc.). Defaults to 10.
        """
        self.target_context_windows = (
            target_context_windows or self.DEFAULT_CONTEXT_WINDOWS
        )
        self.compute_packing_efficiency = compute_packing_efficiency
        self.packing_overhead_tokens = packing_overhead_tokens

    def _get_context_window_name(self, size: int) -> str:
        """Get a human-readable name for a context window size.

        Args:
            size: Context window size in tokens.

        Returns:
            Human-readable name like "4k" or "128k".
        """
        if size >= 1024:
            k = size // 1024
            return f"{k}k"
        return str(size)

    def _compute_context_metrics(
        self,
        token_count: int,
        context_size: int,
    ) -> dict[str, Any]:
        """Compute context window utilization metrics.

        Args:
            token_count: Number of tokens in the sample.
            context_size: Target context window size.

        Returns:
            Dictionary with context utilization metrics.
        """
        size_name = self._get_context_window_name(context_size)
        effective_count = token_count + self.packing_overhead_tokens

        fits = effective_count <= context_size
        utilization = min(effective_count / context_size, 1.0) if context_size > 0 else 0
        wasted = max(0, context_size - effective_count) if fits else 0

        return {
            f"fits_context_{size_name}": fits,
            f"context_utilization_{size_name}": round(utilization, 4),
            f"tokens_wasted_{size_name}": wasted,
        }

    def _compute_packing_efficiency(
        self,
        token_counts: list[int],
        context_size: int,
    ) -> dict[str, Any]:
        """Compute packing efficiency for a set of samples.

        Uses a simple first-fit decreasing bin packing approximation.

        Args:
            token_counts: List of token counts for all samples.
            context_size: Target context window size.

        Returns:
            Dictionary with packing efficiency metrics.
        """
        if not token_counts or context_size <= 0:
            return {
                "packing_efficiency": 0.0,
                "estimated_batches": 0,
                "avg_batch_utilization": 0.0,
            }

        # Add overhead to each sample
        effective_counts = [
            count + self.packing_overhead_tokens for count in token_counts
        ]

        # Sort in decreasing order for first-fit decreasing
        sorted_counts = sorted(effective_counts, reverse=True)

        # First-fit decreasing bin packing
        bins: list[int] = []  # Current fill level of each bin

        for count in sorted_counts:
            if count > context_size:
                # Sample doesn't fit in any bin, needs its own "bin"
                # (will be truncated in practice)
                bins.append(context_size)
                continue

            # Find first bin that can fit this sample
            placed = False
            for i, bin_fill in enumerate(bins):
                if bin_fill + count <= context_size:
                    bins[i] = bin_fill + count
                    placed = True
                    break

            if not placed:
                bins.append(count)

        # Compute metrics
        total_tokens = sum(effective_counts)
        total_capacity = len(bins) * context_size
        efficiency = total_tokens / total_capacity if total_capacity > 0 else 0
        avg_utilization = sum(bins) / (len(bins) * context_size) if bins else 0

        return {
            "packing_efficiency": round(efficiency, 4),
            "estimated_batches": len(bins),
            "avg_batch_utilization": round(avg_utilization, 4),
        }

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Analyze samples for cost optimization metrics.

        This method looks for token count columns in the DataFrame and
        computes context window utilization metrics.

        Args:
            df: Input DataFrame with token count columns.
            schema: Column schema dict to identify text fields.

        Returns:
            Tuple of (DataFrame with added cost analysis columns.
            generated column schema dict).
        """
        result_df = df.copy()
        generated_schema = {}

        if not schema:
            raise ValueError(
                "schema is required to identify text fields for cost analysis. "
                "Please provide a column schema dict that specifies which "
                "columns contain text content."
            )

        # Find token count columns
        token_count_cols = [col for col in df.columns if "token_count" in col]

        if not token_count_cols:
            # No token count columns - return unchanged
            return result_df, generated_schema

        # Use the first token count column found
        token_col = token_count_cols[0]
        analyzer_id = getattr(self, "analyzer_id", "cost")

        # Compute per-sample context metrics for each window size
        for context_size in self.target_context_windows:
            size_name = self._get_context_window_name(context_size)

            # Compute metrics for each row
            context_metrics = df[token_col].apply(
                lambda count: self._compute_context_metrics(
                    int(count) if pd.notna(count) else 0, context_size
                )
            )

            # Extract individual metrics
            result_df[f"{analyzer_id}_fits_context_{size_name}"] = context_metrics.apply(
                lambda m: m[f"fits_context_{size_name}"]
            )
            result_df[
                f"{analyzer_id}_context_utilization_{size_name}"
            ] = context_metrics.apply(lambda m: m[f"context_utilization_{size_name}"])
            result_df[
                f"{analyzer_id}_tokens_wasted_{size_name}"
            ] = context_metrics.apply(lambda m: m[f"tokens_wasted_{size_name}"])

        return result_df, generated_schema

    def compute_dataset_metrics(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Compute dataset-level cost metrics.

        This method provides aggregate metrics useful for cost optimization
        decisions, including packing efficiency and context window coverage.

        Args:
            df: Input DataFrame with token count columns.
            schema: Column schema dict.

        Returns:
            Dictionary with dataset-level cost metrics.
        """
        metrics: dict[str, Any] = {}

        # Find token count column
        token_count_cols = [col for col in df.columns if "token_count" in col]
        if not token_count_cols:
            return metrics

        token_col = token_count_cols[0]
        token_counts = df[token_col].dropna().astype(int).tolist()

        if not token_counts:
            return metrics

        # Basic token statistics
        metrics["total_tokens"] = sum(token_counts)
        metrics["avg_tokens_per_sample"] = round(sum(token_counts) / len(token_counts))
        metrics["max_tokens"] = max(token_counts)
        metrics["min_tokens"] = min(token_counts)

        # Context window coverage for each target size
        for context_size in self.target_context_windows:
            size_name = self._get_context_window_name(context_size)
            effective_counts = [c + self.packing_overhead_tokens for c in token_counts]

            fits_count = sum(1 for c in effective_counts if c <= context_size)
            fits_pct = (fits_count / len(token_counts)) * 100

            metrics[f"fits_context_{size_name}_pct"] = round(fits_pct, 2)
            metrics[f"exceeds_context_{size_name}_count"] = len(token_counts) - fits_count

            # Packing efficiency for this context size
            if self.compute_packing_efficiency:
                packing = self._compute_packing_efficiency(token_counts, context_size)
                metrics[f"packing_efficiency_{size_name}"] = packing["packing_efficiency"]
                metrics[f"estimated_batches_{size_name}"] = packing["estimated_batches"]
                metrics[f"avg_batch_utilization_{size_name}"] = packing[
                    "avg_batch_utilization"
                ]

        # Compute potential savings from deduplication
        if "text_content" in df.columns:
            unique_count = df["text_content"].nunique()
            total_count = len(df)
            if total_count > unique_count:
                dup_count = total_count - unique_count
                dup_tokens = sum(
                    token_counts[i]
                    for i, dup in enumerate(df["text_content"].duplicated())
                    if dup
                )
                metrics["duplicate_tokens_savings"] = dup_tokens
                metrics["duplicate_samples_count"] = dup_count

        return metrics
