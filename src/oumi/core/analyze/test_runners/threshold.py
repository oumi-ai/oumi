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

"""Threshold test runner - compares metrics against threshold values."""

import operator
from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams


class ThresholdTestRunner(BaseTestRunner):
    """Runner for threshold-based tests.

    Compares a metric column against a threshold value using an operator.
    The test fails if too many samples exceed/fall below the threshold.

    Example config:
        - id: short_messages
          type: threshold
          metric: "length__word_count"
          operator: "<"
          value: 10
          max_percentage: 5.0
          severity: medium
    """

    # Mapping of operator strings to functions
    OPERATORS = {
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the threshold test.

        Args:
            test: Test configuration with metric, operator, value.
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
                test, "No metric specified for threshold test"
            )

        error = self.check_column_exists(df, test.metric, test)
        if error:
            return error

        # Get operator function
        if test.operator is None or test.operator not in self.OPERATORS:
            return self.create_error_result(test, f"Invalid operator: {test.operator}")
        op_func = self.OPERATORS[test.operator]

        # Get values and apply comparison
        values = df[test.metric].dropna()
        total_samples = len(values)

        if total_samples == 0:
            return self.create_error_result(
                test, f"No non-null values in column '{test.metric}'"
            )

        # Apply the comparison
        try:
            mask = op_func(values, test.value)
        except TypeError as e:
            return self.create_error_result(
                test, f"Cannot compare {test.metric} with {test.value}: {e}"
            )

        affected_count = mask.sum()
        affected_percentage = (affected_count / total_samples) * 100

        # Get indices of affected samples
        affected_indices = values[mask].index.tolist()

        # Determine pass/fail based on max_percentage
        if test.max_percentage is not None:
            passed = affected_percentage <= test.max_percentage
            threshold = test.max_percentage
        elif test.min_percentage is not None:
            passed = affected_percentage >= test.min_percentage
            threshold = test.min_percentage
        else:
            # No percentage constraint - just report
            passed = affected_count == 0
            threshold = 0.0

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=int(affected_count),
            total_samples=total_samples,
            threshold=threshold,
            actual_value=round(affected_percentage, 2),
            details={
                "operator": test.operator,
                "comparison_value": test.value,
                "metric_min": round(float(values.min()), 2),
                "metric_max": round(float(values.max()), 2),
                "metric_mean": round(float(values.mean()), 2),
            },
            sample_indices=affected_indices,
            metric=test.metric,
        )
