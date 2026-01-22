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

"""Distribution test runner - checks distribution properties of metrics."""

from typing import Any

import numpy as np
import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import DistributionCheck, TestParams


class DistributionTestRunner(BaseTestRunner):
    """Runner for distribution-based tests.

    Checks various properties of value distributions such as
    max fraction, dominant fraction, entropy, unique counts.

    Example config:
        - id: role_imbalance
          type: distribution
          metric: "role"
          check: "max_fraction"
          threshold: 0.8
          severity: medium
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the distribution test.

        Args:
            test: Test configuration with metric, check type, threshold.
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
                test, "No metric specified for distribution test"
            )

        error = self.check_column_exists(df, test.metric, test)
        if error:
            return error

        if test.check is None:
            return self.create_error_result(
                test, "No check type specified for distribution test"
            )

        if test.threshold is None:
            return self.create_error_result(
                test, "No threshold specified for distribution test"
            )

        # Get values
        values = df[test.metric].dropna()
        total_samples = len(values)

        if total_samples == 0:
            return self.create_error_result(
                test, f"No non-null values in column '{test.metric}'"
            )

        # Route to appropriate check
        check_type = test.check.lower()

        if check_type == DistributionCheck.MAX_FRACTION.value:
            return self._check_max_fraction(test, values, total_samples)
        elif check_type == DistributionCheck.DOMINANT_FRACTION.value:
            return self._check_dominant_fraction(test, values, total_samples)
        elif check_type == DistributionCheck.ENTROPY.value:
            return self._check_entropy(test, values, total_samples)
        elif check_type == DistributionCheck.UNIQUE_COUNT.value:
            return self._check_unique_count(test, values, total_samples)
        elif check_type == DistributionCheck.UNIQUE_RATIO.value:
            return self._check_unique_ratio(test, values, total_samples)
        else:
            return self.create_error_result(
                test, f"Unknown distribution check type: {test.check}"
            )

    def _check_max_fraction(
        self,
        test: TestParams,
        values: pd.Series,
        total_samples: int,
    ) -> TestResult:
        """Check if any value exceeds the max fraction threshold.

        Fails if any single value accounts for more than threshold fraction.
        """
        value_counts = values.value_counts(normalize=True)
        max_fraction = float(value_counts.iloc[0])
        max_value = value_counts.index[0]

        # Test passes if max fraction is BELOW threshold (balanced distribution)
        passed = max_fraction <= test.threshold

        # Affected samples are those in the dominant category
        affected_count = int(value_counts.iloc[0] * total_samples)

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=affected_count,
            total_samples=total_samples,
            threshold=test.threshold,
            actual_value=round(max_fraction, 4),
            details={
                "check_type": "max_fraction",
                "max_value": str(max_value),
                "max_fraction": round(max_fraction, 4),
                "value_distribution": {
                    str(k): round(v, 4) for k, v in value_counts.head(5).items()
                },
                "unique_values": len(value_counts),
            },
            metric=test.metric,
        )

    def _check_dominant_fraction(
        self,
        test: TestParams,
        values: pd.Series,
        total_samples: int,
    ) -> TestResult:
        """Check if the dominant (most common) value meets the threshold.

        Passes if the dominant value accounts for at least threshold fraction.
        Useful for checking language consistency (e.g., 90% should be English).
        """
        value_counts = values.value_counts(normalize=True)
        dominant_fraction = float(value_counts.iloc[0])
        dominant_value = value_counts.index[0]

        # Test passes if dominant fraction is AT OR ABOVE threshold
        passed = dominant_fraction >= test.threshold

        # Affected samples are those NOT in the dominant category
        non_dominant_count = total_samples - int(dominant_fraction * total_samples)

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=non_dominant_count,
            total_samples=total_samples,
            threshold=test.threshold,
            actual_value=round(dominant_fraction, 4),
            details={
                "check_type": "dominant_fraction",
                "dominant_value": str(dominant_value),
                "dominant_fraction": round(dominant_fraction, 4),
                "non_dominant_count": non_dominant_count,
                "value_distribution": {
                    str(k): round(v, 4) for k, v in value_counts.head(5).items()
                },
            },
            metric=test.metric,
        )

    def _check_entropy(
        self,
        test: TestParams,
        values: pd.Series,
        total_samples: int,
    ) -> TestResult:
        """Check if distribution entropy is within bounds.

        Higher entropy means more uniform distribution.
        Passes if entropy is at or above threshold.
        """
        value_counts = values.value_counts(normalize=True)
        probabilities = value_counts.values

        # Compute Shannon entropy
        entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))

        # Normalize by max possible entropy (uniform distribution)
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Test passes if normalized entropy is at or above threshold
        passed = normalized_entropy >= test.threshold

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=0,  # Entropy is a global measure
            total_samples=total_samples,
            threshold=test.threshold,
            actual_value=round(normalized_entropy, 4),
            details={
                "check_type": "entropy",
                "raw_entropy": round(entropy, 4),
                "normalized_entropy": round(normalized_entropy, 4),
                "max_possible_entropy": round(max_entropy, 4),
                "unique_values": len(value_counts),
            },
            metric=test.metric,
        )

    def _check_unique_count(
        self,
        test: TestParams,
        values: pd.Series,
        total_samples: int,
    ) -> TestResult:
        """Check if the number of unique values meets the threshold.

        Passes if unique count is at or above threshold.
        """
        unique_count = values.nunique()

        # Test passes if unique count is at or above threshold
        passed = unique_count >= test.threshold

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=0,  # This is a count check, not per-sample
            total_samples=total_samples,
            threshold=test.threshold,
            actual_value=float(unique_count),
            details={
                "check_type": "unique_count",
                "unique_count": unique_count,
                "total_samples": total_samples,
            },
            metric=test.metric,
        )

    def _check_unique_ratio(
        self,
        test: TestParams,
        values: pd.Series,
        total_samples: int,
    ) -> TestResult:
        """Check if the ratio of unique values to total meets the threshold.

        Passes if unique ratio is at or above threshold.
        """
        unique_count = values.nunique()
        unique_ratio = unique_count / total_samples if total_samples > 0 else 0.0

        # Test passes if unique ratio is at or above threshold
        passed = unique_ratio >= test.threshold

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=0,  # This is a ratio check, not per-sample
            total_samples=total_samples,
            threshold=test.threshold,
            actual_value=round(unique_ratio, 4),
            details={
                "check_type": "unique_ratio",
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4),
            },
            metric=test.metric,
        )
