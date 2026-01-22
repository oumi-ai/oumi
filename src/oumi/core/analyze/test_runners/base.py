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

"""Base class for test runners."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.configs.params.test_params import TestParams


class BaseTestRunner(ABC):
    """Abstract base class for test runners.

    Each test runner knows how to execute a specific type of test
    (threshold, percentage, regex, etc.) on analysis DataFrames.
    """

    # Maximum number of sample indices to include in results
    MAX_SAMPLE_INDICES = 20

    @abstractmethod
    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the test and return a result.

        Args:
            test: Test configuration parameters.
            message_df: DataFrame with message-level analysis results.
            conversation_df: DataFrame with conversation-level analysis results.
            summary: Analysis summary dictionary.

        Returns:
            TestResult containing the test outcome.
        """
        pass

    def get_dataframe(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Get the appropriate DataFrame based on test scope.

        Args:
            test: Test configuration with scope setting.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.

        Returns:
            The DataFrame matching the test's scope.
        """
        if test.scope == "conversation":
            return conversation_df
        return message_df

    def create_error_result(
        self,
        test: TestParams,
        error_message: str,
    ) -> TestResult:
        """Create a TestResult for an error condition.

        Args:
            test: Test configuration.
            error_message: Description of the error.

        Returns:
            TestResult with error flag set.
        """
        return TestResult(
            test_id=test.id,
            test_type=test.type,
            passed=False,
            severity=test.severity,
            title=test.get_title(),
            description=test.get_description(),
            affected_samples=0,
            total_samples=0,
            scope=test.scope,
            error=error_message,
        )

    def create_result(
        self,
        test: TestParams,
        passed: bool,
        affected_samples: int,
        total_samples: int,
        threshold: float | None = None,
        actual_value: float | None = None,
        details: dict[str, Any] | None = None,
        sample_indices: list[int] | None = None,
        metric: str | None = None,
    ) -> TestResult:
        """Create a TestResult with common fields populated.

        Args:
            test: Test configuration.
            passed: Whether the test passed.
            affected_samples: Number of samples that triggered the condition.
            total_samples: Total samples checked.
            threshold: The threshold that was checked.
            actual_value: The actual computed value.
            details: Additional test-specific details.
            sample_indices: Indices of affected samples.
            metric: The metric column that was checked.

        Returns:
            Populated TestResult.
        """
        # Apply negation if configured
        if test.negate:
            passed = not passed

        return TestResult(
            test_id=test.id,
            test_type=test.type,
            passed=passed,
            severity=test.severity,
            title=test.get_title(),
            description=test.get_description(),
            affected_samples=affected_samples,
            total_samples=total_samples,
            threshold=threshold,
            actual_value=actual_value,
            details=details or {},
            sample_indices=(sample_indices or [])[: self.MAX_SAMPLE_INDICES],
            scope=test.scope,
            metric=metric,
        )

    def check_column_exists(
        self,
        df: pd.DataFrame,
        column: str,
        test: TestParams,
    ) -> TestResult | None:
        """Check if a column exists in the DataFrame.

        Args:
            df: DataFrame to check.
            column: Column name to look for.
            test: Test configuration (for error reporting).

        Returns:
            None if column exists, TestResult with error if not.
        """
        if column not in df.columns:
            return self.create_error_result(
                test,
                f"Column '{column}' not found in {test.scope} DataFrame. "
                f"Available columns: {list(df.columns)[:10]}...",
            )
        return None
