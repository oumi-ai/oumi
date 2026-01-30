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
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of tests that can be run."""

    THRESHOLD = "threshold"
    """Check if metric exceeds a threshold."""

    PERCENTAGE = "percentage"
    """Check percentage of samples matching a condition."""

    RANGE = "range"
    """Check if metric is within a range."""


@dataclass
class TestConfig:
    """Configuration for a single test.

    Attributes:
        id: Unique identifier for the test.
        type: Type of test to run.
        metric: Path to the metric field (e.g., "LengthAnalyzer.total_words").
        severity: Severity level if test fails.
        title: Human-readable title.
        description: Description of what the test checks.
        operator: Comparison operator for threshold tests.
        value: Value to compare against.
        condition: Condition string for percentage tests.
        max_percentage: Maximum allowed percentage for percentage tests.
        min_percentage: Minimum required percentage for percentage tests.
        min_value: Minimum value for range tests.
        max_value: Maximum value for range tests.
    """

    id: str
    type: TestType
    metric: str
    severity: TestSeverity = TestSeverity.MEDIUM
    title: str = ""
    description: str = ""

    # Threshold test parameters
    operator: str | None = None  # "<", ">", "<=", ">=", "==", "!="
    value: float | int | str | None = None

    # Percentage test parameters
    condition: str | None = None  # e.g., "== True", "> 0.5"
    max_percentage: float | None = None
    min_percentage: float | None = None

    # Range test parameters
    min_value: float | None = None
    max_value: float | None = None


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
        >>> from oumi.analyze.testing import TestEngine, TestConfig, TestType
        >>>
        >>> tests = [
        ...     TestConfig(
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

    def __init__(self, tests: list[TestConfig]):
        """Initialize the test engine.

        Args:
            tests: List of test configurations.
        """
        self.tests = tests

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
                # Create error result
                error_result = TestResult(
                    test_id=test.id,
                    passed=False,
                    severity=test.severity,
                    title=test.title or test.id,
                    description=test.description,
                    error=f"Test execution failed: {e}",
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
        test: TestConfig,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> TestResult:
        """Run a single test.

        Args:
            test: Test configuration.
            results: Analysis results.

        Returns:
            TestResult for this test.
        """
        # Extract values for the metric
        values = self._extract_metric_values(test.metric, results)

        if not values:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                description=test.description,
                metric=test.metric or "",
                error=f"Metric '{test.metric}' not found in results",
            )

        # Run appropriate test type
        if test.type == TestType.THRESHOLD:
            return self._run_threshold_test(test, values)
        elif test.type == TestType.PERCENTAGE:
            return self._run_percentage_test(test, values)
        elif test.type == TestType.RANGE:
            return self._run_range_test(test, values)
        else:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error=f"Unknown test type: {test.type}",
            )

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

        if analyzer_name not in results:
            return []

        analyzer_results = results[analyzer_name]

        # Handle single result (dataset-level) vs list (per-conversation)
        if isinstance(analyzer_results, BaseModel):
            value = self._get_nested_value(analyzer_results, field_path)
            return [value] if value is not None else []

        # List of results
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
        """Get a nested field value from a Pydantic model or CustomMetricResult.

        Handles both:
        - Regular analyzer results: result.field_name
        - Custom metric results: result.values["field_name"]

        Args:
            obj: Pydantic model instance or dict-wrapped object.
            field_path: List of field names to traverse.

        Returns:
            Field value or None if not found.
        """
        current: Any = obj
        for i, field in enumerate(field_path):
            if hasattr(current, field):
                current = getattr(current, field)
            # Check if this is a CustomMetricResult with values dict
            elif hasattr(current, "values"):
                values_attr = getattr(current, "values", None)
                if isinstance(values_attr, dict):
                    # Try to get the remaining path from values dict
                    remaining_path = field_path[i:]
                    temp: Any = values_attr
                    for f in remaining_path:
                        if isinstance(temp, dict) and f in temp:
                            temp = temp[f]
                        else:
                            return None
                    return temp
            else:
                return None
        return current

    def _run_threshold_test(
        self,
        test: TestConfig,
        values: list[Any],
    ) -> TestResult:
        """Run a threshold test.

        The semantics depend on max_percentage vs min_percentage:

        - max_percentage: "At most X% can match the condition"
          Samples MATCHING the condition are problematic.
          Example: "At most 10% can have total_tokens > 4096"
          → Samples with > 4096 tokens are the issues.

        - min_percentage: "At least X% must match the condition"
          Samples NOT matching the condition are problematic.
          Example: "At least 80% must have quality_score > 0.5"
          → Samples with <= 0.5 score are the issues.

        - Neither set: ALL samples must match the condition.

        Args:
            test: Test configuration.
            values: Metric values to test.

        Returns:
            TestResult.
        """
        if test.operator is None or test.value is None:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error="Threshold test requires 'operator' and 'value'",
            )

        op_func = OPERATORS.get(test.operator)
        if op_func is None:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error=f"Unknown operator: {test.operator}",
            )

        # Evaluate the condition for each value
        matching_indices = []  # Samples that MATCH the condition
        non_matching_indices = []  # Samples that DON'T match the condition
        matching_reasons: dict[int, str] = {}
        non_matching_reasons: dict[int, str] = {}

        for i, value in enumerate(values):
            try:
                if op_func(value, test.value):
                    # Value matches the condition (e.g., total_tokens > 4096)
                    matching_indices.append(i)
                    matching_reasons[i] = (
                        f"{value} {test.operator} {test.value}"
                    )
                else:
                    # Value does NOT match the condition
                    non_matching_indices.append(i)
                    non_matching_reasons[i] = (
                        f"{value} does not satisfy {test.operator} {test.value}"
                    )
            except (TypeError, ValueError):
                # Can't evaluate - treat as non-matching
                non_matching_indices.append(i)
                non_matching_reasons[i] = f"Cannot evaluate: {value}"

        total_count = len(values)
        matching_count = len(matching_indices)
        non_matching_count = len(non_matching_indices)
        matching_pct = 100.0 * matching_count / total_count if total_count > 0 else 0.0
        non_matching_pct = 100.0 * non_matching_count / total_count if total_count > 0 else 0.0

        # Determine pass/fail and which samples are "affected" (problematic)
        # The semantics depend on whether max_percentage or min_percentage is used:
        #
        # max_percentage: "At most X% can match the condition"
        #   - Matching samples are problematic (they exceed the threshold)
        #   - Example: "At most 10% can have tokens > 4096"
        #
        # min_percentage: "At least X% must match the condition"
        #   - Non-matching samples are problematic (they don't meet the requirement)
        #   - Example: "At least 80% must have quality_score > 0.5"
        #
        # Neither set: ALL samples must match (any non-matching are problematic)

        passed = True
        if test.max_percentage is not None:
            # max_percentage: matching samples are the problematic ones
            if matching_pct > test.max_percentage:
                passed = False
            affected_indices = matching_indices
            affected_count = matching_count
            affected_pct = matching_pct
            failure_reasons = matching_reasons
        elif test.min_percentage is not None:
            # min_percentage: non-matching samples are the problematic ones
            if matching_pct < test.min_percentage:
                passed = False
            affected_indices = non_matching_indices
            affected_count = non_matching_count
            affected_pct = non_matching_pct
            failure_reasons = non_matching_reasons
        else:
            # Neither set: ALL must match, non-matching are problematic
            passed = non_matching_count == 0
            affected_indices = non_matching_indices
            affected_count = non_matching_count
            affected_pct = non_matching_pct
            failure_reasons = non_matching_reasons

        return TestResult(
            test_id=test.id,
            passed=passed,
            severity=test.severity,
            title=test.title or test.id,
            description=test.description,
            metric=test.metric or "",
            affected_count=affected_count,
            total_count=total_count,
            affected_percentage=round(affected_pct, 2),
            threshold=test.max_percentage or test.min_percentage,
            sample_indices=affected_indices[:50],  # Limit to first 50
            details={
                "operator": test.operator,
                "value": test.value,
                "max_percentage": test.max_percentage,
                "min_percentage": test.min_percentage,
                "matching_count": matching_count,
                "matching_percentage": round(matching_pct, 2),
                "failure_reasons": {k: v for k, v in list(failure_reasons.items())[:50]},
            },
        )

    def _run_percentage_test(
        self,
        test: TestConfig,
        values: list[Any],
    ) -> TestResult:
        """Run a percentage test.

        Checks what percentage of values match a condition.

        Args:
            test: Test configuration.
            values: Metric values to test.

        Returns:
            TestResult.
        """
        if test.condition is None:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error="Percentage test requires 'condition'",
            )

        # Parse condition (e.g., "== True", "> 0.5")
        match = re.match(r"([<>=!]+)\s*(.+)", test.condition.strip())
        if not match:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error=f"Invalid condition format: {test.condition}",
            )

        op_str, value_str = match.groups()
        op_func = OPERATORS.get(op_str)
        if op_func is None:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error=f"Unknown operator in condition: {op_str}",
            )

        # Parse value
        try:
            if value_str.lower() == "true":
                compare_value: Any = True
            elif value_str.lower() == "false":
                compare_value = False
            elif value_str.lower() == "none":
                compare_value = None
            else:
                compare_value = float(value_str)
        except ValueError:
            compare_value = value_str

        # Count matches and non-matches
        matching_indices = []
        non_matching_indices = []
        failure_reasons: dict[int, str] = {}
        for i, value in enumerate(values):
            try:
                if op_func(value, compare_value):
                    matching_indices.append(i)
                else:
                    non_matching_indices.append(i)
                    failure_reasons[i] = f"{value} does not match {test.condition}"
            except (TypeError, ValueError):
                non_matching_indices.append(i)
                failure_reasons[i] = f"Cannot evaluate: {value}"

        matching_count = len(matching_indices)
        total_count = len(values)
        matching_pct = 100.0 * matching_count / total_count if total_count > 0 else 0.0

        # Check against thresholds
        passed = True
        if test.max_percentage is not None and matching_pct > test.max_percentage:
            passed = False
        if test.min_percentage is not None and matching_pct < test.min_percentage:
            passed = False

        # For min_percentage tests, affected samples are those that don't match
        # For max_percentage tests, affected samples are those that do match
        if test.min_percentage is not None and not passed:
            affected_indices = non_matching_indices
            affected_count = len(non_matching_indices)
            affected_pct = 100.0 * affected_count / total_count if total_count > 0 else 0.0
        else:
            affected_indices = matching_indices
            affected_count = matching_count
            affected_pct = matching_pct

        return TestResult(
            test_id=test.id,
            passed=passed,
            severity=test.severity,
            title=test.title or test.id,
            description=test.description,
            metric=test.metric or "",
            affected_count=affected_count,
            total_count=total_count,
            affected_percentage=round(affected_pct, 2),
            threshold=test.max_percentage or test.min_percentage,
            sample_indices=affected_indices[:50],
            details={
                "condition": test.condition,
                "matching_count": matching_count,
                "matching_percentage": round(matching_pct, 2),
                "failure_reasons": {k: v for k, v in list(failure_reasons.items())[:50]},
            },
        )

    def _run_range_test(
        self,
        test: TestConfig,
        values: list[Any],
    ) -> TestResult:
        """Run a range test.

        Checks what percentage of values fall outside a range.

        Args:
            test: Test configuration.
            values: Metric values to test.

        Returns:
            TestResult.
        """
        if test.min_value is None and test.max_value is None:
            return TestResult(
                test_id=test.id,
                passed=False,
                severity=test.severity,
                title=test.title or test.id,
                metric=test.metric or "",
                error="Range test requires 'min_value' and/or 'max_value'",
            )

        # Find values outside range
        affected_indices = []
        for i, value in enumerate(values):
            try:
                outside_range = False
                if test.min_value is not None and value < test.min_value:
                    outside_range = True
                if test.max_value is not None and value > test.max_value:
                    outside_range = True
                if outside_range:
                    affected_indices.append(i)
            except (TypeError, ValueError):
                pass

        affected_count = len(affected_indices)
        total_count = len(values)
        affected_pct = 100.0 * affected_count / total_count if total_count > 0 else 0.0

        # Default: no values should be outside range
        max_pct = test.max_percentage if test.max_percentage is not None else 0.0
        passed = affected_pct <= max_pct

        return TestResult(
            test_id=test.id,
            passed=passed,
            severity=test.severity,
            title=test.title or test.id,
            description=test.description,
            metric=test.metric or "",
            affected_count=affected_count,
            total_count=total_count,
            affected_percentage=round(affected_pct, 2),
            threshold=max_pct,
            sample_indices=affected_indices[:50],
            details={
                "min_value": test.min_value,
                "max_value": test.max_value,
            },
        )
