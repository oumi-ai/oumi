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


# Maximum number of sample indices to include in test results
MAX_SAMPLE_INDICES = 50
# Maximum number of failure reasons to include in test details
MAX_FAILURE_REASONS = 50

# Operator mapping
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
        """Initialize the test engine.

        Args:
            tests: List of test configurations.
        """
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
        """Run a single test.

        Args:
            test: Test configuration.
            results: Analysis results.

        Returns:
            TestResult for this test.
        """
        # Extract values for the metric (with original indices preserved)
        if not test.metric:
            return self._create_error_result(test, "Test requires 'metric' field")
        indexed_values = self._extract_metric_values(test.metric, results)

        if not indexed_values:
            return self._create_error_result(
                test, f"Metric '{test.metric}' not found in results"
            )

        # Run appropriate test type
        if test.type == TestType.THRESHOLD:
            return self._run_threshold_test(test, indexed_values)
        else:
            return self._create_error_result(test, f"Unknown test type: {test.type}")

    def _extract_metric_values(
        self,
        metric: str,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> list[tuple[int, Any]]:
        """Extract metric values from results with their original indices.

        Metric format: "AnalyzerName.field_name" or "AnalyzerName.nested.field"

        Args:
            metric: Metric path string.
            results: Analysis results.

        Returns:
            List of (original_index, value) tuples. None values are filtered
            out, but original indices are preserved so that sample_indices in
            test results map back to the correct conversations.
        """
        parts = metric.split(".")
        if len(parts) < 2:
            return []

        analyzer_name = parts[0]
        field_path = parts[1:]

        if analyzer_name not in results:
            return []

        analyzer_results = results[analyzer_name]

        # Handle single result (dataset-level) vs list (per-conversation)
        if isinstance(analyzer_results, BaseModel):
            value = self._get_nested_value(analyzer_results, field_path)
            return [(0, value)] if value is not None else []

        # List of results -- preserve original index
        indexed_values = []
        for i, result in enumerate(analyzer_results):
            value = self._get_nested_value(result, field_path)
            if value is not None:
                indexed_values.append((i, value))

        return indexed_values

    def _get_nested_value(self, obj: Any, field_path: list[str]) -> Any:
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
                return None
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

        The semantics depend on max_percentage vs min_percentage:

        - max_percentage: "At most X% can match the condition"
          Samples MATCHING the condition are problematic.
          Example: "At most 10% can have total_tokens > 4096"

        - min_percentage: "At least X% must match the condition"
          Samples NOT matching the condition are problematic.
          Example: "At least 80% must have quality_score > 0.5"

        - Neither set: ALL samples must match the condition.

        Args:
            test: Test configuration.
            indexed_values: List of (original_index, value) tuples.

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

        # Evaluate the condition for each value
        matching_indices = []  # Original indices of samples that MATCH
        non_matching_indices = []  # Original indices of samples that DON'T match
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

        # Determine pass/fail and which samples are "affected" (problematic)
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

        # For single-value (dataset-level) metrics, include the actual value
        actual_value = None
        if total_count == 1:
            val = indexed_values[0][1]
            if isinstance(val, (int, float)):
                actual_value = float(val)

        # Use `is not None` to avoid treating 0.0 as falsy
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
