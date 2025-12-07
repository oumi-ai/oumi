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

"""Tests for the CostAnalyzer."""

import pandas as pd
import pytest

from oumi.core.analyze.cost_analyzer import CostAnalyzer


def _create_test_df(token_counts: list[int]) -> pd.DataFrame:
    """Create a test DataFrame with token counts."""
    return pd.DataFrame({
        "text_content": [f"text_{i}" for i in range(len(token_counts))],
        "text_content_length_token_count": token_counts,
    })


def _get_schema() -> dict:
    """Get the standard schema for testing."""
    return {
        "text_content": {"content_type": "text"},
    }


class TestContextWindowMetrics:
    """Tests for context window utilization metrics."""

    def test_sample_fits_in_context(self):
        """Test that samples fitting in context window are marked correctly."""
        analyzer = CostAnalyzer(target_context_windows=[4096])
        df = _create_test_df([1000, 2000, 3000])
        result = analyzer.analyze_sample(df, _get_schema())

        # All samples should fit in 4k context (with 10 token overhead)
        assert result.iloc[0]["cost_fits_context_4k"] == True
        assert result.iloc[1]["cost_fits_context_4k"] == True
        assert result.iloc[2]["cost_fits_context_4k"] == True

    def test_sample_exceeds_context(self):
        """Test that samples exceeding context window are marked correctly."""
        analyzer = CostAnalyzer(target_context_windows=[2048])
        df = _create_test_df([1000, 2000, 3000])
        result = analyzer.analyze_sample(df, _get_schema())

        # 1000 + 10 overhead = 1010, fits in 2048
        assert result.iloc[0]["cost_fits_context_2k"] == True
        # 2000 + 10 overhead = 2010, fits in 2048
        assert result.iloc[1]["cost_fits_context_2k"] == True
        # 3000 + 10 overhead = 3010, exceeds 2048
        assert result.iloc[2]["cost_fits_context_2k"] == False

    def test_context_utilization_calculation(self):
        """Test context utilization is calculated correctly."""
        analyzer = CostAnalyzer(
            target_context_windows=[4096], packing_overhead_tokens=0
        )
        df = _create_test_df([2048])
        result = analyzer.analyze_sample(df, _get_schema())

        # 2048/4096 = 0.5
        assert result.iloc[0]["cost_context_utilization_4k"] == 0.5

    def test_tokens_wasted_calculation(self):
        """Test tokens wasted is calculated correctly."""
        analyzer = CostAnalyzer(
            target_context_windows=[4096], packing_overhead_tokens=0
        )
        df = _create_test_df([1000])
        result = analyzer.analyze_sample(df, _get_schema())

        # 4096 - 1000 = 3096 tokens wasted
        assert result.iloc[0]["cost_tokens_wasted_4k"] == 3096

    def test_multiple_context_windows(self):
        """Test analysis with multiple context window sizes."""
        analyzer = CostAnalyzer(target_context_windows=[2048, 4096, 8192])
        df = _create_test_df([3000])
        result = analyzer.analyze_sample(df, _get_schema())

        # 3000 + 10 = 3010, exceeds 2048, fits 4096 and 8192
        assert result.iloc[0]["cost_fits_context_2k"] == False
        assert result.iloc[0]["cost_fits_context_4k"] == True
        assert result.iloc[0]["cost_fits_context_8k"] == True


class TestPackingEfficiency:
    """Tests for packing efficiency calculations."""

    def test_perfect_packing(self):
        """Test packing efficiency with perfectly fitting samples."""
        analyzer = CostAnalyzer(
            target_context_windows=[1024], packing_overhead_tokens=0
        )
        # 4 samples of 512 tokens each = 2 batches at 100% efficiency
        token_counts = [512, 512, 512, 512]
        result = analyzer._compute_packing_efficiency(token_counts, 1024)

        assert result["estimated_batches"] == 2
        assert result["packing_efficiency"] == 1.0
        assert result["avg_batch_utilization"] == 1.0

    def test_inefficient_packing(self):
        """Test packing efficiency with poor fit."""
        analyzer = CostAnalyzer(
            target_context_windows=[1024], packing_overhead_tokens=0
        )
        # 3 samples of 600 tokens each - can't pack efficiently
        # Each needs its own bin since 600+600 > 1024
        token_counts = [600, 600, 600]
        result = analyzer._compute_packing_efficiency(token_counts, 1024)

        assert result["estimated_batches"] == 3
        # 1800 / 3072 = ~0.586
        assert 0.58 <= result["packing_efficiency"] <= 0.59

    def test_mixed_sizes_packing(self):
        """Test packing with mixed sample sizes."""
        analyzer = CostAnalyzer(
            target_context_windows=[1000], packing_overhead_tokens=0
        )
        # Mix of sizes that can pack together
        # 700 + 300 = 1000, 500 + 400 = 900, 200 alone
        token_counts = [700, 500, 400, 300, 200]
        result = analyzer._compute_packing_efficiency(token_counts, 1000)

        # First-fit decreasing: 700+300, 500+400, 200 = 3 bins
        assert result["estimated_batches"] == 3

    def test_empty_input(self):
        """Test packing efficiency with empty input."""
        analyzer = CostAnalyzer()
        result = analyzer._compute_packing_efficiency([], 4096)

        assert result["packing_efficiency"] == 0.0
        assert result["estimated_batches"] == 0

    def test_oversized_samples(self):
        """Test packing with samples larger than context window."""
        analyzer = CostAnalyzer(
            target_context_windows=[1024], packing_overhead_tokens=0
        )
        # Sample larger than window gets its own "bin" (will be truncated)
        token_counts = [2000, 500]
        result = analyzer._compute_packing_efficiency(token_counts, 1024)

        # Two bins needed
        assert result["estimated_batches"] == 2


class TestDatasetMetrics:
    """Tests for dataset-level metrics."""

    def test_basic_statistics(self):
        """Test basic token statistics calculation."""
        analyzer = CostAnalyzer(target_context_windows=[4096])
        df = _create_test_df([100, 200, 300, 400, 500])
        metrics = analyzer.compute_dataset_metrics(df, _get_schema())

        assert metrics["total_tokens"] == 1500
        assert metrics["avg_tokens_per_sample"] == 300
        assert metrics["max_tokens"] == 500
        assert metrics["min_tokens"] == 100

    def test_context_coverage_percentage(self):
        """Test context window coverage percentage."""
        analyzer = CostAnalyzer(
            target_context_windows=[200], packing_overhead_tokens=0
        )
        df = _create_test_df([100, 150, 250, 300])  # 2 fit, 2 exceed
        metrics = analyzer.compute_dataset_metrics(df, _get_schema())

        assert metrics["fits_context_200_pct"] == 50.0
        assert metrics["exceeds_context_200_count"] == 2

    def test_duplicate_savings_calculation(self):
        """Test duplicate token savings calculation."""
        analyzer = CostAnalyzer(target_context_windows=[4096])
        df = pd.DataFrame({
            "text_content": ["hello", "hello", "world", "hello"],
            "text_content_length_token_count": [10, 10, 20, 10],
        })
        metrics = analyzer.compute_dataset_metrics(df, _get_schema())

        # 2 duplicates of "hello" (10 tokens each)
        assert metrics["duplicate_samples_count"] == 2
        assert metrics["duplicate_tokens_savings"] == 20


class TestIntegration:
    """Integration tests for the full analyzer."""

    def test_full_analysis_flow(self):
        """Test complete analysis flow."""
        analyzer = CostAnalyzer(
            target_context_windows=[2048, 4096],
            compute_packing_efficiency=True,
        )
        df = _create_test_df([500, 1000, 1500, 2500, 3500])
        result = analyzer.analyze_sample(df, _get_schema())

        # Check that all expected columns are present
        assert "cost_fits_context_2k" in result.columns
        assert "cost_fits_context_4k" in result.columns
        assert "cost_context_utilization_2k" in result.columns
        assert "cost_context_utilization_4k" in result.columns
        assert "cost_tokens_wasted_2k" in result.columns
        assert "cost_tokens_wasted_4k" in result.columns

    def test_schema_required(self):
        """Test that schema is required."""
        analyzer = CostAnalyzer()
        df = _create_test_df([100])

        with pytest.raises(ValueError, match="schema is required"):
            analyzer.analyze_sample(df, None)

    def test_no_token_columns(self):
        """Test handling when no token count columns exist."""
        analyzer = CostAnalyzer()
        df = pd.DataFrame({"text_content": ["hello"]})
        result = analyzer.analyze_sample(df, _get_schema())

        # Should return unchanged DataFrame
        assert list(result.columns) == list(df.columns)

    def test_custom_overhead_tokens(self):
        """Test custom packing overhead tokens."""
        # With 100 overhead, 950 + 100 = 1050 > 1024
        analyzer = CostAnalyzer(
            target_context_windows=[1024], packing_overhead_tokens=100
        )
        df = _create_test_df([950])
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["cost_fits_context_1k"] == False

        # With 0 overhead, 950 < 1024
        analyzer = CostAnalyzer(
            target_context_windows=[1024], packing_overhead_tokens=0
        )
        result = analyzer.analyze_sample(df, _get_schema())

        assert result.iloc[0]["cost_fits_context_1k"] == True

    def test_context_window_naming(self):
        """Test human-readable context window naming."""
        analyzer = CostAnalyzer()

        assert analyzer._get_context_window_name(512) == "512"
        assert analyzer._get_context_window_name(1024) == "1k"
        assert analyzer._get_context_window_name(2048) == "2k"
        assert analyzer._get_context_window_name(4096) == "4k"
        assert analyzer._get_context_window_name(8192) == "8k"
        assert analyzer._get_context_window_name(16384) == "16k"
        assert analyzer._get_context_window_name(32768) == "32k"
        assert analyzer._get_context_window_name(131072) == "128k"
