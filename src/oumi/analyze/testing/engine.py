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

"""Test engine for validating typed analysis results.

This module provides a test engine that operates on typed Pydantic results
instead of DataFrames. Tests are pure validation - no computation allowed.
"""

import logging
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary
from oumi.core.configs.params.test_params import TestParams, TestType

logger = logging.getLogger(__name__)


OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class TestEngine:
    """Engine for running tests on typed analysis results.

    Tests operate on typed Pydantic results, not DataFrames. This ensures
    tests are pure validation with no computation - all metrics must be
    pre-computed by analyzers.

    Example:
        >>> from oumi.analyze.testing import TestEngine, TestParams, TestType
        >>>
        >>> tests = [
        ...     TestParams(
        ...         id="max_words",
        ...         type=TestType.THRESHOLD,
        ...         metric="LengthAnalyzer.total_words",
        ...         operator=">",
        ...         value=10000,
        ...         max_percentage=5.0,
        ...         severity=TestSeverity.MEDIUM,
        ...     ),
        ... ]
        >>> engine = TestEngine(tests)
        >>> summary = engine.run(results)
        >>> print(f"Pass rate: {summary.pass_rate}%")

    Args:
        tests: List of test configurations.
    """

    def __init__(self, tests: list[TestParams]):
        """Initialize the test engine with test configurations."""
        self.tests = tests

    def _create_error_result(self, test: TestParams, error: str) -> TestResult:
        """Create a TestResult for an error condition.

        Args:
            test: Test configuration.
            error: Error message.

        Returns:
            TestResult with passed=False and error set.
        """
        return TestResult(
            test_id=test.id,
            passed=False,
            severity=TestSeverity(test.severity),
            title=test.title or test.id,
            description=test.description or "",
            metric=test.metric or "",
            error=error,
        )

    def _calculate_percentage(self, count: int, total: int) -> float:
        """Calculate percentage, handling division by zero.

        Args:
            count: Numerator.
            total: Denominator.

        Returns:
            Percentage (0.0 to 100.0).
        """
        return 100.0 * count / total if total > 0 else 0.0

    def _build_test_result(
        self,
        test: TestParams,
        passed: bool,
        total_count: int,
        affected_indices: list[int],
        affected_pct: float,
        details: dict[str, Any],
        actual_value: float | None = None,
    ) -> TestResult:
        """Build a TestResult from common fields.

        Args:
            test: Test configuration.
            passed: Whether the test passed.
            total_count: Total number of values tested.
            affected_indices: Indices of affected samples.
            affected_pct: Percentage of affected samples.
            details: Test-specific details.
            actual_value: Actual metric value for single-value tests.

        Returns:
            TestResult instance.
        """
        return TestResult(
            test_id=test.id,
            passed=passed,
            severity=TestSeverity(test.severity),
            title=test.title or test.id,
            description=test.description or "",
            metric=test.metric or "",
            affected_count=len(affected_indices),
            total_count=total_count,
            affected_percentage=round(affected_pct, 2),
            threshold=test.max_percentage or test.min_percentage,
            actual_value=actual_value,
            sample_indices=affected_indices[:50],
            details=details,
        )

    def _get_actual_value(self, values: list[Any]) -> float | None:
        """Extract actual value for single-value metrics.

        Args:
            values: List of metric values.

        Returns:
            Float value if this is a single numeric value, None otherwise.
        """
        if len(values) == 1:
            val = values[0]
            if isinstance(val, int | float):
                return float(val)
            if isinstance(val, bool):
                return 1.0 if val else 0.0
        return None

    def run(
        self,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> TestSummary:
        """Run all tests on the analysis results.

        Args:
            results: Dictionary mapping analyzer names to results.

        Returns:
            TestSummary containing all test results.
        """
        test_results: list[TestResult] = []

        logger.info(f"Running {len(self.tests)} tests...")

        for test in self.tests:
            try:
                result = self._run_single_test(test, results)
                test_results.append(result)

                status = "PASSED" if result.passed else "FAILED"
                logger.debug(
                    f"  Test '{test.id}': {status} "
                    f"({result.affected_count}/{result.total_count} affected)"
                )
            except Exception as e:
                error_result = self._create_error_result(
                    test, f"Test execution failed: {e}"
                )
                test_results.append(error_result)
                logger.warning(f"  Test '{test.id}': ERROR - {e}")

        summary = TestSummary.from_results(test_results)

        logger.info(
            f"Test results: {summary.passed_tests}/{summary.total_tests} passed "
            f"({summary.pass_rate}%)"
        )
        if summary.high_severity_failures > 0:
            logger.warning(f"  {summary.high_severity_failures} high severity failures")

        return summary

    def _run_single_test(
        self,
        test: TestParams,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> TestResult:
        """Run a single test.

        Args:
            test: Test configuration.
            results: Analysis results.

        Returns:
            TestResult for this test.
        """
        if not test.metric:
            return self._create_error_result(test, "Test requires 'metric' field")

        values = self._extract_metric_values(test.metric, results)

        if not values:
            return self._create_error_result(
                test, f"Metric '{test.metric}' not found in results"
            )

        if test.type == TestType.THRESHOLD:
            return self._run_threshold_test(test, values)
        else:
            return self._create_error_result(test, f"Unknown test type: {test.type}")

    def _normalize_analyzer_name(
        self,
        analyzer_name: str,
        available_names: set[str],
    ) -> str | None:
        """Try to find a matching analyzer name from available results.

        Handles common variations like:
        - LengthAnalyzer -> length
        - DifficultyJudgeAnalyzer -> difficulty_judge
        - length -> LengthAnalyzer

        Args:
            analyzer_name: The analyzer name from the metric path.
            available_names: Set of available analyzer names in results.

        Returns:
            Matching analyzer name or None if not found.
        """
        # Direct match
        if analyzer_name in available_names:
            return analyzer_name

        # Try lowercase
        lower_name = analyzer_name.lower()
        for name in available_names:
            if name.lower() == lower_name:
                return name

        # Try removing "Analyzer" suffix and converting to snake_case
        if analyzer_name.endswith("Analyzer"):
            base_name = analyzer_name[:-8]  # Remove "Analyzer"
            # Convert CamelCase to snake_case
            snake_name = ""
            for i, char in enumerate(base_name):
                if char.isupper() and i > 0:
                    snake_name += "_"
                snake_name += char.lower()

            for name in available_names:
                if name.lower() == snake_name or name.lower() == base_name.lower():
                    return name

        # Try adding "Analyzer" suffix
        class_name = analyzer_name.title().replace("_", "") + "Analyzer"
        for name in available_names:
            if name == class_name:
                return name

        return None

    def _extract_metric_values(
        self,
        metric: str,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> list[Any]:
        """Extract metric values from results.

        Metric format: "AnalyzerName.field_name" or "AnalyzerName.nested.field"

        Args:
            metric: Metric path string.
            results: Analysis results.

        Returns:
            List of values for the metric.
        """
        parts = metric.split(".")
        if len(parts) < 2:
            return []

        analyzer_name = parts[0]
        field_path = parts[1:]

        # Try to find a matching analyzer name (handles variations)
        matched_name = self._normalize_analyzer_name(analyzer_name, set(results.keys()))
        if matched_name is None:
            return []

        analyzer_results = results[matched_name]

        if isinstance(analyzer_results, BaseModel):
            value = self._get_nested_value(analyzer_results, field_path)
            return [value] if value is not None else []

        values = []
        for result in analyzer_results:
            value = self._get_nested_value(result, field_path)
            if value is not None:
                values.append(value)

        return values

    def _get_nested_value(
        self,
        obj: Any,
        field_path: list[str],
    ) -> Any | None:
        """Get a nested field value from a Pydantic model or dict.

        Args:
            obj: Pydantic model instance or dict.
            field_path: List of field names to traverse.

        Returns:
            Field value or None if not found.
        """
        current: Any = obj
        for i, field in enumerate(field_path):
            if isinstance(current, BaseModel):
                # Pydantic model - check if field exists in model_fields
                if field in type(current).model_fields:
                    current = getattr(current, field)
                else:
                    # Check for CustomMetricResult with values dict
                    values = getattr(current, "values", None)
                    if isinstance(values, dict):
                        return self._traverse_dict(values, field_path[i:])
                    return None
            elif isinstance(current, dict):
                if field in current:
                    current = current[field]
                else:
                    return None
            else:
                raise TypeError(
                    f"Cannot traverse type {type(current).__name__}. "
                    f"Expected BaseModel or dict, got {current!r}"
                )
        return current

    def _traverse_dict(self, d: dict, path: list[str]) -> Any | None:
        """Traverse a dict using a field path."""
        current: Any = d
        for field in path:
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return None
        return current

    def _run_threshold_test(
        self,
        test: TestParams,
        values: list[Any],
    ) -> TestResult:
        """Run a threshold test.

        Args:
            test: Test configuration.
            values: Metric values to test.

        Returns:
            TestResult.
        """
        if test.operator is None or test.value is None:
            return self._create_error_result(
                test, "Threshold test requires 'operator' and 'value'"
            )

        op_func = OPERATORS.get(test.operator)
        if op_func is None:
            return self._create_error_result(test, f"Unknown operator: {test.operator}")

        matching_indices = []
        non_matching_indices = []
        matching_reasons: dict[int, str] = {}
        non_matching_reasons: dict[int, str] = {}

        for i, value in enumerate(values):
            try:
                if op_func(value, test.value):
                    matching_indices.append(i)
                    matching_reasons[i] = (
                        f"{value} satisfies {test.operator} {test.value}"
                    )
                else:
                    non_matching_indices.append(i)
                    non_matching_reasons[i] = (
                        f"{value} does not satisfy {test.operator} {test.value}"
                    )
            except (TypeError, ValueError):
                non_matching_indices.append(i)
                non_matching_reasons[i] = f"Cannot evaluate: {value}"

        total_count = len(values)
        matching_count = len(matching_indices)
        non_matching_count = len(non_matching_indices)
        matching_pct = self._calculate_percentage(matching_count, total_count)
        non_matching_pct = self._calculate_percentage(non_matching_count, total_count)

        # Determine pass/fail based on percentage thresholds.
        # For threshold tests, matching samples are considered "affected" (flagged).
        passed = True
        affected_indices = []
        affected_pct = 0.0
        failure_reasons: dict[int, str] = {}

        if test.max_percentage is not None and matching_pct > test.max_percentage:
            passed = False
            affected_indices = matching_indices
            affected_pct = matching_pct
            failure_reasons = matching_reasons

        if test.min_percentage is not None and matching_pct < test.min_percentage:
            passed = False
            # If max also failed, keep matching; otherwise use non-matching as
            # the affected set because they prevent meeting the minimum.
            if not affected_indices:
                affected_indices = non_matching_indices
                affected_pct = non_matching_pct
                failure_reasons = non_matching_reasons

        # Default case: no percentage thresholds.
        # Matching samples are flagged, and the test passes when no samples match.
        if test.max_percentage is None and test.min_percentage is None:
            passed = matching_count == 0
            affected_indices = matching_indices
            affected_pct = matching_pct
            failure_reasons = matching_reasons

        return self._build_test_result(
            test=test,
            passed=passed,
            total_count=total_count,
            affected_indices=affected_indices,
            affected_pct=affected_pct,
            actual_value=self._get_actual_value(values),
            details={
                "operator": test.operator,
                "value": test.value,
                "max_percentage": test.max_percentage,
                "min_percentage": test.min_percentage,
                "matching_count": matching_count,
                "matching_percentage": round(matching_pct, 2),
                "failure_reasons": {
                    k: v for k, v in list(failure_reasons.items())[:50]
                },
            },
        )
