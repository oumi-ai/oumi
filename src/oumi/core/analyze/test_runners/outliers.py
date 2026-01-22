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

"""Outliers test runner - detects statistical outliers in metrics."""

from typing import Any

import numpy as np
import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams


class OutliersTestRunner(BaseTestRunner):
    """Runner for statistical outlier detection tests.

    Detects outliers using standard deviation from mean and checks
    if the outlier rate is within bounds. Supports multimodal-aware
    detection if the analysis utilities are available.

    Example config:
        - id: token_count_outliers
          type: outliers
          metric: "length__token_count"
          std_threshold: 3.0
          max_percentage: 5.0
          severity: low
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the outliers test.

        Args:
            test: Test configuration with metric, std_threshold, max_percentage.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary (unused).

        Returns:
            TestResult indicating pass/fail.
        """
        df = self.get_dataframe(test, message_df, conversation_df)

        # Check column exists
        if test.metric is None:
            return self.create_error_result(
                test, "No metric specified for outliers test"
            )

        error = self.check_column_exists(df, test.metric, test)
        if error:
            return error

        # Get numeric values
        values = df[test.metric].dropna()
        total_samples = len(values)

        if total_samples < 3:
            return self.create_error_result(
                test,
                f"Insufficient data for outlier detection in '{test.metric}' "
                f"(need at least 3 samples, got {total_samples})",
            )

        # Try to use multimodal-aware outlier detection
        try:
            from oumi.utils.analysis_utils import (
                compute_multimodal_outliers,
                detect_distribution_type,
            )

            dist_result = detect_distribution_type(values)
            outlier_mask, outlier_details = compute_multimodal_outliers(
                values, dist_result, test.std_threshold
            )
            detection_method = "multimodal"
            num_modes = dist_result.num_modes

        except ImportError:
            # Fallback to simple z-score outlier detection
            outlier_mask, outlier_details = self._simple_outlier_detection(
                values, test.std_threshold
            )
            detection_method = "zscore"
            num_modes = 1

        outlier_count = outlier_mask.sum()
        outlier_percentage = (outlier_count / total_samples) * 100

        # Get indices of outlier samples
        outlier_indices = values[outlier_mask].index.tolist()

        # Determine pass/fail based on max_percentage
        threshold = test.max_percentage if test.max_percentage is not None else 5.0
        passed = outlier_percentage <= threshold

        # Compute statistics for details
        mean_val = float(values.mean())
        std_val = float(values.std())

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=int(outlier_count),
            total_samples=total_samples,
            threshold=threshold,
            actual_value=round(outlier_percentage, 2),
            details={
                "std_threshold": test.std_threshold,
                "detection_method": detection_method,
                "num_modes": num_modes,
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "min": round(float(values.min()), 2),
                "max": round(float(values.max()), 2),
                "upper_bound": round(mean_val + (test.std_threshold * std_val), 2),
                "lower_bound": round(mean_val - (test.std_threshold * std_val), 2),
                **outlier_details,
            },
            sample_indices=outlier_indices,
            metric=test.metric,
        )

    def _simple_outlier_detection(
        self,
        values: pd.Series,
        std_threshold: float,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """Simple z-score based outlier detection.

        Args:
            values: Series of numeric values.
            std_threshold: Number of standard deviations for outlier threshold.

        Returns:
            Tuple of (outlier_mask, details_dict).
        """
        mean_val = values.mean()
        std_val = values.std()

        if std_val == 0:
            # No variation, no outliers
            return pd.Series([False] * len(values), index=values.index), {
                "num_outliers": 0,
                "high_outliers": 0,
                "low_outliers": 0,
            }

        z_scores = np.abs((values - mean_val) / std_val)
        outlier_mask = z_scores > std_threshold

        # Count high and low outliers
        high_mask = values > (mean_val + std_threshold * std_val)
        low_mask = values < (mean_val - std_threshold * std_val)

        return outlier_mask, {
            "num_outliers": int(outlier_mask.sum()),
            "high_outliers": int(high_mask.sum()),
            "low_outliers": int(low_mask.sum()),
        }
