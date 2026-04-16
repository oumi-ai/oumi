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

"""Test engine for validating typed analysis results."""

import logging
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary
from oumi.core.configs.params.test_params import TestParams, TestType

logger = logging.getLogger(__name__)


# Maximum number of sample indices to include in test results
MAX_SAMPLE_INDICES = 50
# Maximum number of failure reasons to include in test details
MAX_FAILURE_REASONS = 50

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

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, tests: list[TestParams]):
        """Initialize the test engine."""
        self.tests = tests

    def _create_error_result(self, test: TestParams, error: str) -> TestResult:
        """Create a TestResult for an error condition."""
        return TestResult(
            test_id=test.id,
            passed=False,
            severity=TestSeverity(test.severity),
            title=test.title or test.id,
            description=test.description or "",
            metric=test.metric or "",
            error=error,
        )

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
        """Run a single test and return its result."""
        if not test.metric:
            return self._create_error_result(test, "Test requires 'metric' field")
        indexed_values = self._extract_metric_values(test.metric, results)

        if not indexed_values:
            return self._create_error_result(
                test, f"Metric '{test.metric}' not found in results"
            )

        if test.type == TestType.THRESHOLD:
            return self._run_threshold_test(test, indexed_values)
        else:
            return self._create_error_result(test, f"Unknown test type: {test.type}")

    def _extract_metric_values(
        self,
        metric: str,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> list[tuple[int, Any]]:
        """Extract values for a metric path like "AnalyzerName.field_name"."""
        parts = metric.split(".")
        if len(parts) < 2:
            return []

        analyzer_name = parts[0]
        field_path = parts[1:]

        if analyzer_name not in results:
            return []

        analyzer_results = results[analyzer_name]

        if isinstance(analyzer_results, BaseModel):
            value = self._get_nested_value(analyzer_results, field_path)
            return [(0, value)] if value is not None else []

        indexed_values = []
        for i, result in enumerate(analyzer_results):
            value = self._get_nested_value(result, field_path)
            if value is not None:
                indexed_values.append((i, value))

        return indexed_values

    def _get_nested_value(self, obj: Any, field_path: list[str]) -> Any:
        """Get a nested field value from a Pydantic model or dict."""
        current: Any = obj
        for i, field in enumerate(field_path):
            if isinstance(current, BaseModel):
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
        indexed_values: list[tuple[int, Any]],
    ) -> TestResult:
        """Run a threshold test.

        Semantics depend on max_percentage vs min_percentage:
        - max_percentage: at most X% can match (matching = problematic)
        - min_percentage: at least X% must match (non-matching = problematic)
        - neither: all samples must match
        """
        if test.operator is None or test.value is None:
            return self._create_error_result(
                test, "Threshold test requires 'operator' and 'value'"
            )

        op_func = OPERATORS.get(test.operator)
        if op_func is None:
            return self._create_error_result(test, f"Unknown operator: {test.operator}")

        matching_indices: list[int] = []
        non_matching_indices: list[int] = []
        matching_reasons: dict[int, str] = {}
        non_matching_reasons: dict[int, str] = {}

        for orig_idx, value in indexed_values:
            try:
                if op_func(value, test.value):
                    matching_indices.append(orig_idx)
                    matching_reasons[orig_idx] = f"{value} {test.operator} {test.value}"
                else:
                    non_matching_indices.append(orig_idx)
                    non_matching_reasons[orig_idx] = (
                        f"{value} does not satisfy {test.operator} {test.value}"
                    )
            except (TypeError, ValueError):
                non_matching_indices.append(orig_idx)
                non_matching_reasons[orig_idx] = f"Cannot evaluate: {value}"

        total_count = len(indexed_values)
        matching_count = len(matching_indices)
        non_matching_count = len(non_matching_indices)
        matching_pct = 100.0 * matching_count / total_count if total_count > 0 else 0.0
        non_matching_pct = (
            100.0 * non_matching_count / total_count if total_count > 0 else 0.0
        )

        passed = True
        if test.max_percentage is not None:
            if matching_pct > test.max_percentage:
                passed = False
            affected_indices = matching_indices
            affected_count = matching_count
            affected_pct = matching_pct
            failure_reasons = matching_reasons
        elif test.min_percentage is not None:
            if matching_pct < test.min_percentage:
                passed = False
            affected_indices = non_matching_indices
            affected_count = non_matching_count
            affected_pct = non_matching_pct
            failure_reasons = non_matching_reasons
        else:
            passed = non_matching_count == 0
            affected_indices = non_matching_indices
            affected_count = non_matching_count
            affected_pct = non_matching_pct
            failure_reasons = non_matching_reasons

        actual_value = None
        if total_count == 1:
            val = indexed_values[0][1]
            if isinstance(val, (int, float)):
                actual_value = float(val)

        threshold = (
            test.max_percentage
            if test.max_percentage is not None
            else test.min_percentage
        )

        return TestResult(
            test_id=test.id,
            passed=passed,
            severity=TestSeverity(test.severity),
            title=test.title or test.id,
            description=test.description or "",
            metric=test.metric or "",
            affected_count=affected_count,
            total_count=total_count,
            affected_percentage=round(affected_pct, 2),
            threshold=threshold,
            actual_value=actual_value,
            sample_indices=affected_indices[:MAX_SAMPLE_INDICES],
            all_affected_indices=affected_indices,
            details={
                "operator": test.operator,
                "value": test.value,
                "max_percentage": test.max_percentage,
                "min_percentage": test.min_percentage,
                "matching_count": matching_count,
                "matching_percentage": round(matching_pct, 2),
                "failure_reasons": {
                    k: v for k, v in list(failure_reasons.items())[:MAX_FAILURE_REASONS]
                },
            },
        )
