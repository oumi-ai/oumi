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

"""Regex test runner - matches patterns against text content."""

import re
from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import TestParams


class RegexTestRunner(BaseTestRunner):
    """Runner for regex pattern matching tests.

    Searches for a regex pattern in a text field and checks if the
    match rate is within bounds.

    Example config:
        - id: special_token_leakage
          type: regex
          field: "text_content"
          pattern: "<\\|(?:endoftext|im_start|im_end)\\|>"
          max_percentage: 0
          severity: high
    """

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the regex test.

        Args:
            test: Test configuration with field, pattern, max_percentage.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary (unused).

        Returns:
            TestResult indicating pass/fail.
        """
        df = self.get_dataframe(test, message_df, conversation_df)

        # Check field exists
        if test.text_field is None:
            return self.create_error_result(test, "No field specified for regex test")

        error = self.check_column_exists(df, test.text_field, test)
        if error:
            return error

        if test.pattern is None:
            return self.create_error_result(test, "No pattern specified for regex test")

        # Compile regex pattern
        try:
            flags = 0 if test.case_sensitive else re.IGNORECASE
            pattern = re.compile(test.pattern, flags)
        except re.error as e:
            return self.create_error_result(
                test, f"Invalid regex pattern '{test.pattern}': {e}"
            )

        # Get text values
        values = df[test.text_field].fillna("").astype(str)
        total_samples = len(values)

        if total_samples == 0:
            return self.create_error_result(
                test, f"No values in column '{test.text_field}'"
            )

        # Apply regex search
        mask = values.str.contains(pattern, regex=True, na=False)
        affected_count = mask.sum()
        affected_percentage = (affected_count / total_samples) * 100

        # Get indices of affected samples
        affected_indices = df[mask].index.tolist()

        # Get sample matches for details
        sample_matches = []
        for idx in affected_indices[:5]:  # Limit to 5 examples
            text = values.loc[idx]
            match = pattern.search(text)
            if match:
                sample_matches.append(
                    {
                        "index": int(idx),
                        "match": match.group(),
                        "position": match.start(),
                    }
                )

        # Determine pass/fail based on max_percentage
        threshold = test.max_percentage if test.max_percentage is not None else 0.0
        passed = affected_percentage <= threshold

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=int(affected_count),
            total_samples=total_samples,
            threshold=threshold,
            actual_value=round(affected_percentage, 2),
            details={
                "pattern": test.pattern,
                "case_sensitive": test.case_sensitive,
                "sample_matches": sample_matches,
            },
            sample_indices=affected_indices,
            metric=test.text_field,
        )
