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

"""Test engine for running user-defined tests on analysis results.

This module provides the main TestEngine class that orchestrates running
all configured tests on dataset analysis results.

Example:
    >>> from oumi.core.analyze.test_engine import TestEngine
    >>> from oumi.core.configs.params.test_params import TestParams
    >>>
    >>> tests = [
    ...     TestParams(
    ...         id="no_pii",
    ...         type="percentage",
    ...         metric="quality__has_pii",
    ...         condition="== True",
    ...         max_percentage=1.0,
    ...         severity="high",
    ...     )
    ... ]
    >>> engine = TestEngine(tests)
    >>> results = engine.run_tests(message_df, conversation_df, summary)
"""

from typing import Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult, TestSummary
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.analyze.test_runners.composite import CompositeTestRunner
from oumi.core.analyze.test_runners.contains import ContainsTestRunner
from oumi.core.analyze.test_runners.distribution import DistributionTestRunner
from oumi.core.analyze.test_runners.outliers import OutliersTestRunner
from oumi.core.analyze.test_runners.percentage import PercentageTestRunner
from oumi.core.analyze.test_runners.python_runner import PythonTestRunner
from oumi.core.analyze.test_runners.query import QueryTestRunner
from oumi.core.analyze.test_runners.regex import RegexTestRunner
from oumi.core.analyze.test_runners.threshold import ThresholdTestRunner
from oumi.core.configs.params.test_params import TestParams, TestType
from oumi.utils.logging import logger


class TestEngine:
    """Engine for running user-defined tests on analysis results.

    The TestEngine takes a list of test configurations and runs them
    against analysis DataFrames, returning structured results.

    Attributes:
        tests: List of test configurations to run.
    """

    def __init__(self, tests: list[TestParams]):
        """Initialize the test engine.

        Args:
            tests: List of TestParams configurations.
        """
        self.tests = tests
        self._runners = self._build_runners()

    def _build_runners(self) -> dict[str, BaseTestRunner]:
        """Build test runner instances for each test type.

        Returns:
            Dictionary mapping test type strings to runner instances.
        """
        # Create composite runner with reference to this engine
        composite_runner = CompositeTestRunner(test_engine=self)

        return {
            TestType.THRESHOLD.value: ThresholdTestRunner(),
            TestType.PERCENTAGE.value: PercentageTestRunner(),
            TestType.DISTRIBUTION.value: DistributionTestRunner(),
            TestType.REGEX.value: RegexTestRunner(),
            TestType.CONTAINS.value: ContainsTestRunner(),
            TestType.CONTAINS_ANY.value: ContainsTestRunner(),
            TestType.CONTAINS_ALL.value: ContainsTestRunner(),
            TestType.QUERY.value: QueryTestRunner(),
            TestType.OUTLIERS.value: OutliersTestRunner(),
            TestType.COMPOSITE.value: composite_runner,
            TestType.PYTHON.value: PythonTestRunner(),
        }

    def run_tests(
        self,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestSummary:
        """Run all configured tests and return a summary.

        Args:
            message_df: DataFrame with message-level analysis results.
            conversation_df: DataFrame with conversation-level analysis results.
            summary: Analysis summary dictionary.

        Returns:
            TestSummary containing all test results and statistics.
        """
        results: list[TestResult] = []

        logger.info(f"Running {len(self.tests)} user-defined tests...")

        for test in self.tests:
            try:
                result = self.run_single_test(
                    test, message_df, conversation_df, summary
                )
                results.append(result)

                # Log result
                status = "PASSED" if result.passed else "FAILED"
                if result.error:
                    status = "ERROR"
                logger.debug(
                    f"  Test '{test.id}': {status} "
                    f"({result.affected_samples}/{result.total_samples} affected)"
                )

            except Exception as e:
                # Create error result for unexpected exceptions
                error_result = TestResult(
                    test_id=test.id,
                    test_type=test.type,
                    passed=False,
                    severity=test.severity,
                    title=test.get_title(),
                    description=test.get_description(),
                    affected_samples=0,
                    total_samples=0,
                    scope=test.scope,
                    error=f"Unexpected error: {e}",
                )
                results.append(error_result)
                logger.warning(f"  Test '{test.id}': ERROR - {e}")

        # Create summary
        test_summary = TestSummary.from_results(results)

        # Log summary
        logger.info(
            f"Test results: {test_summary.passed_tests}/{test_summary.total_tests} "
            f"passed ({test_summary.pass_rate}%)"
        )
        if test_summary.high_severity_failures > 0:
            logger.warning(
                f"  {test_summary.high_severity_failures} high severity failures"
            )

        return test_summary

    def run_single_test(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Run a single test and return the result.

        Args:
            test: Test configuration.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary.

        Returns:
            TestResult for this test.
        """
        # Get the appropriate runner for this test type
        runner = self._runners.get(test.type)

        if runner is None:
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
                error=f"Unknown test type: {test.type}",
            )

        # Run the test
        return runner.run(test, message_df, conversation_df, summary)

    def get_runner(self, test_type: str) -> BaseTestRunner | None:
        """Get the runner for a specific test type.

        Args:
            test_type: Test type string.

        Returns:
            Runner instance or None if not found.
        """
        return self._runners.get(test_type)
