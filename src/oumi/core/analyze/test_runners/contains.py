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

"""Contains test runner - checks for substring presence in text."""

from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams, TestType


class ContainsTestRunner(BaseTestRunner):
    """Runner for contains-based tests.

    Checks if text contains specific substrings. Supports three modes:
    - contains: single substring
    - contains-any: any of multiple substrings
    - contains-all: all of multiple substrings

    Example configs:
        - id: placeholder_text
          type: contains-any
          field: "text_content"
          values: ["[Name]", "[Company]", "[DATE]"]
          max_percentage: 1.0
          severity: medium

        - id: has_code
          type: contains
          field: "text_content"
          value: "```"
          min_percentage: 50.0
          negate: true
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the contains test.

        Args:
            test: Test configuration with field, value/values, max/min_percentage.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary (unused).

        Returns:
            TestResult indicating pass/fail.
        """
        df = self.get_dataframe(test, message_df, conversation_df)

        # Check field exists
        if test.text_field is None:
            return self.create_error_result(
                test, "No field specified for contains test"
            )

        error = self.check_column_exists(df, test.text_field, test)
        if error:
            return error

        # Get substrings to search for
        if test.values:
            substrings = test.values
        elif test.value is not None:
            substrings = [str(test.value)]
        else:
            return self.create_error_result(
                test, "No value or values specified for contains test"
            )

        # Get text values
        values = df[test.text_field].fillna("").astype(str)
        if not test.case_sensitive:
            values = values.str.lower()
            substrings = [s.lower() for s in substrings]

        total_samples = len(values)

        if total_samples == 0:
            return self.create_error_result(
                test, f"No values in column '{test.text_field}'"
            )

        # Apply contains logic based on test type
        if test.type == TestType.CONTAINS.value:
            # Single substring (use first one)
            mask = values.str.contains(substrings[0], regex=False, na=False)
            match_mode = "single"
        elif test.type == TestType.CONTAINS_ANY.value:
            # Any of the substrings
            mask = pd.Series([False] * total_samples, index=values.index)
            for substring in substrings:
                mask = mask | values.str.contains(substring, regex=False, na=False)
            match_mode = "any"
        elif test.type == TestType.CONTAINS_ALL.value:
            # All of the substrings
            mask = pd.Series([True] * total_samples, index=values.index)
            for substring in substrings:
                mask = mask & values.str.contains(substring, regex=False, na=False)
            match_mode = "all"
        else:
            return self.create_error_result(
                test, f"Unknown contains test type: {test.type}"
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
                "substrings": substrings,
                "match_mode": match_mode,
                "case_sensitive": test.case_sensitive,
                "bound_type": bound_type,
            },
            sample_indices=affected_indices,
            metric=test.text_field,
        )
