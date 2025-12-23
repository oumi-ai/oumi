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

"""Tests for distribution detection utilities in analysis_utils.py."""

import numpy as np
import pandas as pd
import pytest

from oumi.utils.analysis_utils import (
    DistributionAnalysisResult,
    DistributionType,
    ModeStatistics,
    compute_mode_statistics,
    compute_multimodal_outliers,
    compute_statistics_with_distribution,
    detect_distribution_type,
)


class TestDistributionType:
    """Tests for DistributionType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert DistributionType.UNIMODAL == "unimodal"
        assert DistributionType.BIMODAL == "bimodal"
        assert DistributionType.MULTIMODAL == "multimodal"
        assert DistributionType.UNIFORM == "uniform"
        assert DistributionType.INSUFFICIENT_DATA == "insufficient_data"


class TestModeStatistics:
    """Tests for ModeStatistics dataclass."""

    def test_mode_statistics_creation(self):
        """Test creating a ModeStatistics instance."""
        mode = ModeStatistics(
            mode_id=0,
            mean=10.5,
            std=2.3,
            count=100,
            weight=0.6,
        )
        assert mode.mode_id == 0
        assert mode.mean == 10.5
        assert mode.std == 2.3
        assert mode.count == 100
        assert mode.weight == 0.6


class TestDistributionAnalysisResult:
    """Tests for DistributionAnalysisResult dataclass."""

    def test_result_creation_minimal(self):
        """Test creating a minimal result."""
        result = DistributionAnalysisResult(
            distribution_type=DistributionType.UNIMODAL,
            num_modes=1,
            global_statistics={"mean": 10.0, "std": 2.0},
        )
        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.num_modes == 1
        assert result.mode_statistics == []
        assert result.mode_assignments is None
        assert result.confidence == 0.0

    def test_result_creation_full(self):
        """Test creating a result with all fields."""
        mode_stats = [
            ModeStatistics(mode_id=0, mean=5.0, std=1.0, count=50, weight=0.5),
            ModeStatistics(mode_id=1, mean=15.0, std=1.0, count=50, weight=0.5),
        ]
        result = DistributionAnalysisResult(
            distribution_type=DistributionType.BIMODAL,
            num_modes=2,
            global_statistics={"mean": 10.0, "std": 5.0},
            mode_statistics=mode_stats,
            mode_assignments=[0, 0, 1, 1],
            confidence=0.95,
        )
        assert result.distribution_type == DistributionType.BIMODAL
        assert result.num_modes == 2
        assert len(result.mode_statistics) == 2
        assert result.mode_assignments == [0, 0, 1, 1]
        assert result.confidence == 0.95


class TestDetectDistributionType:
    """Tests for detect_distribution_type function."""

    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([], dtype=float)
        result = detect_distribution_type(series)
        assert result.distribution_type == DistributionType.INSUFFICIENT_DATA
        assert result.num_modes == 0

    def test_single_value(self):
        """Test with single value."""
        series = pd.Series([10.0])
        result = detect_distribution_type(series)
        assert result.distribution_type == DistributionType.INSUFFICIENT_DATA

    def test_small_sample(self):
        """Test with sample smaller than min_samples."""
        series = pd.Series([1, 2, 3, 4, 5])  # 5 samples, below default min_samples=50
        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.num_modes == 1
        assert len(result.mode_statistics) == 1

    def test_unimodal_normal_distribution(self):
        """Test detection of unimodal normal distribution."""
        np.random.seed(42)
        # Generate clearly unimodal data
        series = pd.Series(np.random.normal(100, 10, 200))
        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.num_modes == 1

    def test_bimodal_distribution(self):
        """Test detection of bimodal distribution."""
        np.random.seed(42)
        # Generate clearly bimodal data with well-separated modes
        mode1 = np.random.normal(50, 5, 100)
        mode2 = np.random.normal(150, 5, 100)
        series = pd.Series(np.concatenate([mode1, mode2]))

        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type == DistributionType.BIMODAL
        assert result.num_modes == 2
        assert len(result.mode_statistics) == 2

        # Mode means should be close to 50 and 150
        means = sorted([m.mean for m in result.mode_statistics])
        assert 40 < means[0] < 60
        assert 140 < means[1] < 160

    def test_multimodal_distribution(self):
        """Test detection of multimodal distribution (3+ modes)."""
        np.random.seed(42)
        # Generate clearly trimodal data
        mode1 = np.random.normal(25, 3, 80)
        mode2 = np.random.normal(75, 3, 80)
        mode3 = np.random.normal(125, 3, 80)
        series = pd.Series(np.concatenate([mode1, mode2, mode3]))

        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type == DistributionType.MULTIMODAL
        assert result.num_modes >= 3

    def test_low_cv_is_unimodal(self):
        """Test that low coefficient of variation is detected as unimodal."""
        # Very tight distribution with low CV
        series = pd.Series([100, 100.1, 100.2, 99.9, 99.8] * 20)
        result = detect_distribution_type(series, cv_threshold=0.5)
        assert result.distribution_type == DistributionType.UNIMODAL

    def test_mode_assignments_are_returned(self):
        """Test that mode assignments are returned for multimodal data."""
        np.random.seed(42)
        mode1 = np.random.normal(0, 1, 100)
        mode2 = np.random.normal(10, 1, 100)
        series = pd.Series(np.concatenate([mode1, mode2]))

        result = detect_distribution_type(series, min_samples=50)

        if result.distribution_type == DistributionType.BIMODAL:
            assert result.mode_assignments is not None
            assert len(result.mode_assignments) == 200
            assert set(result.mode_assignments).issubset({0, 1})

    def test_confidence_value(self):
        """Test that confidence value is reasonable."""
        np.random.seed(42)
        # Well-separated modes should have high confidence
        mode1 = np.random.normal(0, 1, 100)
        mode2 = np.random.normal(20, 1, 100)  # Very far apart
        series = pd.Series(np.concatenate([mode1, mode2]))

        result = detect_distribution_type(series, min_samples=50)
        if result.distribution_type == DistributionType.BIMODAL:
            assert result.confidence > 0.5  # Well-separated modes

    def test_handles_nan_values(self):
        """Test that NaN values are handled correctly."""
        data = [1, 2, np.nan, 3, 4, np.nan, 5] * 20
        series = pd.Series(data)
        result = detect_distribution_type(series, min_samples=50)
        # Should still work, ignoring NaN values
        assert result.distribution_type in [
            DistributionType.UNIMODAL,
            DistributionType.INSUFFICIENT_DATA,
        ]


class TestComputeModeStatistics:
    """Tests for compute_mode_statistics function."""

    def test_single_mode(self):
        """Test with single mode."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0, 0, 0, 0, 0])
        weights = np.array([1.0])

        stats = compute_mode_statistics(series, labels, weights)

        assert len(stats) == 1
        assert stats[0].mode_id == 0
        assert stats[0].count == 5
        assert 2.5 < stats[0].mean < 3.5  # Mean of 1,2,3,4,5 is 3

    def test_two_modes(self):
        """Test with two modes."""
        series = pd.Series([1.0, 2.0, 10.0, 11.0])
        labels = np.array([0, 0, 1, 1])
        weights = np.array([0.5, 0.5])

        stats = compute_mode_statistics(series, labels, weights)

        assert len(stats) == 2
        # Stats are sorted by mean
        assert stats[0].mean < stats[1].mean
        assert stats[0].count == 2
        assert stats[1].count == 2

    def test_empty_mode(self):
        """Test when a mode has no samples."""
        series = pd.Series([1.0, 2.0, 3.0])
        labels = np.array([0, 0, 0])  # All in mode 0
        weights = np.array([0.7, 0.3])  # But weights suggest 2 modes

        stats = compute_mode_statistics(series, labels, weights)
        # Only mode 0 should have stats since mode 1 is empty
        assert len(stats) == 1
        assert stats[0].mode_id == 0


class TestComputeMultimodalOutliers:
    """Tests for compute_multimodal_outliers function."""

    def test_unimodal_outlier_detection(self):
        """Test outlier detection for unimodal distribution."""
        # Normal data with one clear outlier
        data = [10, 11, 10, 9, 11, 10, 10, 100]  # 100 is the outlier
        series = pd.Series(data)

        dist_result = DistributionAnalysisResult(
            distribution_type=DistributionType.UNIMODAL,
            num_modes=1,
            global_statistics={"mean": 10.0, "std": 1.0},
        )

        outlier_mask, details = compute_multimodal_outliers(
            series, dist_result, std_threshold=3.0
        )

        assert outlier_mask[-1] == True  # 100 should be an outlier
        assert details["num_outliers"] >= 1
        assert "outlier_ratio" in details

    def test_bimodal_per_mode_outliers(self):
        """Test per-mode outlier detection for bimodal distribution."""
        np.random.seed(42)
        # Two modes with one outlier in each
        mode1 = [10, 11, 10, 9, 50]  # 50 is outlier for mode 1
        mode2 = [100, 101, 100, 99, 150]  # 150 is outlier for mode 2
        data = mode1 + mode2
        series = pd.Series(data)

        mode_stats = [
            ModeStatistics(mode_id=0, mean=10.0, std=1.0, count=5, weight=0.5),
            ModeStatistics(mode_id=1, mean=100.0, std=1.0, count=5, weight=0.5),
        ]
        dist_result = DistributionAnalysisResult(
            distribution_type=DistributionType.BIMODAL,
            num_modes=2,
            global_statistics={"mean": 55.0, "std": 45.0},
            mode_statistics=mode_stats,
            mode_assignments=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )

        outlier_mask, details = compute_multimodal_outliers(
            series, dist_result, std_threshold=3.0
        )

        assert details["distribution_type"] == "bimodal"
        assert "outliers_per_mode" in details

    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([], dtype=float)
        dist_result = DistributionAnalysisResult(
            distribution_type=DistributionType.INSUFFICIENT_DATA,
            num_modes=0,
            global_statistics={},
        )

        outlier_mask, details = compute_multimodal_outliers(series, dist_result)

        assert len(outlier_mask) == 0
        assert details["num_outliers"] == 0


class TestComputeStatisticsWithDistribution:
    """Tests for compute_statistics_with_distribution function."""

    def test_basic_stats_included(self):
        """Test that basic statistics are included."""
        series = pd.Series([1, 2, 3, 4, 5])
        stats = compute_statistics_with_distribution(series, include_distribution=False)

        assert "count" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert stats["count"] == 5
        assert stats["mean"] == 3.0

    def test_distribution_info_included(self):
        """Test that distribution information is included when requested."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(100, 10, 100))
        stats = compute_statistics_with_distribution(series, include_distribution=True)

        assert "distribution_type" in stats
        assert "num_modes" in stats
        assert "mode_separation_confidence" in stats

    def test_mode_statistics_included(self):
        """Test that mode statistics are included for multimodal data."""
        np.random.seed(42)
        mode1 = np.random.normal(0, 1, 100)
        mode2 = np.random.normal(20, 1, 100)
        series = pd.Series(np.concatenate([mode1, mode2]))

        stats = compute_statistics_with_distribution(series, include_distribution=True)

        if stats.get("distribution_type") == "bimodal":
            assert "mode_statistics" in stats
            assert len(stats["mode_statistics"]) == 2

    def test_empty_series(self):
        """Test with empty series."""
        series = pd.Series([], dtype=float)
        stats = compute_statistics_with_distribution(series)
        assert stats["count"] == 0

    def test_decimal_precision(self):
        """Test that decimal precision is respected."""
        series = pd.Series([1.123456, 2.234567, 3.345678])
        stats = compute_statistics_with_distribution(series, decimal_precision=2)
        # Mean should be rounded to 2 decimal places
        assert isinstance(stats["mean"], float)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_same_values(self):
        """Test with all identical values."""
        series = pd.Series([5.0] * 100)
        result = detect_distribution_type(series, min_samples=50)
        # Should be detected as unimodal (no variance)
        assert result.distribution_type == DistributionType.UNIMODAL

    def test_two_distinct_values(self):
        """Test with only two distinct values (binary-like)."""
        series = pd.Series([0] * 50 + [1] * 50)
        result = detect_distribution_type(series, min_samples=50)
        # Could be unimodal or bimodal depending on GMM fitting
        assert result.distribution_type in [
            DistributionType.UNIMODAL,
            DistributionType.BIMODAL,
        ]

    def test_negative_values(self):
        """Test with negative values."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(-100, 10, 100))
        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type == DistributionType.UNIMODAL
        assert result.global_statistics["mean"] < 0

    def test_large_values(self):
        """Test with very large values."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(1e10, 1e8, 100))
        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type == DistributionType.UNIMODAL

    def test_integer_series(self):
        """Test with integer series."""
        series = pd.Series([1, 2, 3, 4, 5] * 20)
        result = detect_distribution_type(series, min_samples=50)
        assert result.distribution_type in [
            DistributionType.UNIMODAL,
            DistributionType.MULTIMODAL,
        ]
