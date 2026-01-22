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

"""Percentage test runner - checks what percentage of samples match a condition."""

from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams


class PercentageTestRunner(BaseTestRunner):
    """Runner for percentage-based tests.

    Evaluates a condition on a metric column and checks if the percentage
    of matching samples is within bounds.

    Example config:
        - id: no_pii
          type: percentage
          metric: "quality__has_pii"
          condition: "== True"
          max_percentage: 1.0
          severity: high
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the percentage test.

        Args:
            test: Test configuration with metric, condition, max/min_percentage.
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
                test, "No metric specified for percentage test"
            )

        error = self.check_column_exists(df, test.metric, test)
        if error:
            return error

        if test.condition is None:
            return self.create_error_result(
                test, "No condition specified for percentage test"
            )

        # Get values
        values = df[test.metric]
        total_samples = len(values)

        if total_samples == 0:
            return self.create_error_result(
                test, f"No values in column '{test.metric}'"
            )

        # Evaluate condition using pandas eval
        try:
            # Build expression: column_name condition
            # e.g., "quality__has_pii == True" becomes evaluation on the series
            condition = test.condition.strip()

            # Handle common condition patterns
            if condition.startswith("=="):
                compare_value = self._parse_value(condition[2:].strip())
                mask = values == compare_value
            elif condition.startswith("!="):
                compare_value = self._parse_value(condition[2:].strip())
                mask = values != compare_value
            elif condition.startswith(">="):
                compare_value = self._parse_value(condition[2:].strip())
                mask = values >= compare_value
            elif condition.startswith("<="):
                compare_value = self._parse_value(condition[2:].strip())
                mask = values <= compare_value
            elif condition.startswith(">"):
                compare_value = self._parse_value(condition[1:].strip())
                mask = values > compare_value
            elif condition.startswith("<"):
                compare_value = self._parse_value(condition[1:].strip())
                mask = values < compare_value
            else:
                # Try to use pandas eval for complex conditions
                # Build a temporary DataFrame for evaluation
                temp_df = pd.DataFrame({"value": values})
                mask = temp_df.eval(f"value {condition}")

        except Exception as e:
            return self.create_error_result(
                test, f"Failed to evaluate condition '{test.condition}': {e}"
            )

        affected_count = mask.sum()
        affected_percentage = (affected_count / total_samples) * 100

        # Get indices of affected samples
        affected_indices = df[mask].index.tolist()

        # Determine pass/fail based on percentage bounds
        if test.max_percentage is not None:
            passed = affected_percentage <= test.max_percentage
            threshold = test.max_percentage
            bound_type = "max"
        elif test.min_percentage is not None:
            passed = affected_percentage >= test.min_percentage
            threshold = test.min_percentage
            bound_type = "min"
        else:
            # Should not happen due to validation, but handle gracefully
            passed = affected_count == 0
            threshold = 0.0
            bound_type = "max"

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=int(affected_count),
            total_samples=total_samples,
            threshold=threshold,
            actual_value=round(affected_percentage, 2),
            details={
                "condition": test.condition,
                "bound_type": bound_type,
                "matching_count": int(affected_count),
                "non_matching_count": total_samples - int(affected_count),
            },
            sample_indices=affected_indices,
            metric=test.metric,
        )

    def _parse_value(self, value_str: str) -> Any:
        """Parse a value string into the appropriate Python type.

        Args:
            value_str: String representation of a value.

        Returns:
            Parsed value (bool, int, float, or string).
        """
        value_str = value_str.strip()

        # Handle boolean
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

        # Handle None/null
        if value_str.lower() in ("none", "null"):
            return None

        # Handle numbers
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass

        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")
        ):
            return value_str[1:-1]

        # Return as-is (string)
        return value_str
