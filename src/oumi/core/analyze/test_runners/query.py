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

"""Query test runner - executes pandas query expressions."""

from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams


class QueryTestRunner(BaseTestRunner):
    """Runner for pandas query expression tests.

    Executes a pandas query expression and checks if the percentage
    of matching samples is within bounds. This is the most flexible
    test type, allowing complex multi-column conditions.

    Example config:
        - id: low_quality_responses
          type: query
          expression: "role == 'assistant' and instruct_reward__score < 0.3"
          max_percentage: 10.0
          severity: high
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the query test.

        Args:
            test: Test configuration with expression, max/min_percentage.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary (unused).

        Returns:
            TestResult indicating pass/fail.
        """
        df = self.get_dataframe(test, message_df, conversation_df)

        if test.expression is None:
            return self.create_error_result(
                test, "No expression specified for query test"
            )

        total_samples = len(df)

        if total_samples == 0:
            return self.create_error_result(test, "DataFrame is empty")

        # Execute the pandas query
        try:
            matching_df = df.query(test.expression)
        except Exception as e:
            return self.create_error_result(
                test,
                f"Failed to execute query '{test.expression}': {e}. "
                f"Available columns: {list(df.columns)[:10]}...",
            )

        affected_count = len(matching_df)
        affected_percentage = (affected_count / total_samples) * 100

        # Get indices of affected samples
        affected_indices = matching_df.index.tolist()

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
            # Default: pass if none match
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
                "expression": test.expression,
                "bound_type": bound_type,
                "matching_count": int(affected_count),
            },
            sample_indices=affected_indices,
        )
