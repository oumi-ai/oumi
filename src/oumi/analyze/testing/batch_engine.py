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

"""Batch-aware test engine for validating analysis results incrementally.

Processes results one batch at a time with constant memory, accumulating
only lightweight counters and affected conversation IDs. Call
``process_batch()`` for each batch, then ``finalize()`` to compute the
final TestSummary.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from oumi.analyze.testing.engine import (
    MAX_FAILURE_REASONS,
    MAX_SAMPLE_INDICES,
    OPERATORS,
)
from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary
from oumi.core.configs.params.test_params import TestParams, TestType

logger = logging.getLogger(__name__)


@dataclass
class _TestAccumulator:
    """Per-test state accumulated across batches."""

    test: TestParams
    matching_count: int = 0
    non_matching_count: int = 0
    total_count: int = 0
    matching_conversation_ids: list[str | None] = field(default_factory=list)
    non_matching_conversation_ids: list[str | None] = field(default_factory=list)
    matching_reasons: dict[int, str] = field(default_factory=dict)
    non_matching_reasons: dict[int, str] = field(default_factory=dict)
    error: str | None = None


class BatchTestEngine:
    """Engine for running tests on analysis results incrementally.

    Unlike ``TestEngine`` which requires the full dataset in memory,
    ``BatchTestEngine`` processes one batch at a time and accumulates
    only counters and affected conversation IDs.

    Example:
        >>> engine = BatchTestEngine(tests)
        >>> for batch_results, batch_ids in batches:
        ...     engine.process_batch(batch_results, batch_ids)
        >>> summary = engine.finalize()

    Args:
        tests: List of test configurations.
    """

    def __init__(self, tests: list[TestParams]):
        """Initialize the batch test engine with test configurations."""
        self.tests = tests
        self._accumulators: dict[str, _TestAccumulator] = {}
        for test in tests:
            self._accumulators[test.id] = _TestAccumulator(test=test)

    def process_batch(
        self,
        results: dict[str, list[BaseModel] | BaseModel],
        conversation_ids: list[str | None],
    ) -> None:
        """Process one batch of analysis results.

        Args:
            results: Analyzer results for this batch (same format as
                ``TestEngine.run()``).
            conversation_ids: Conversation IDs for each item in this batch,
                aligned by index with the per-conversation result lists.
        """
        for test in self.tests:
            acc = self._accumulators[test.id]
            if acc.error:
                continue

            try:
                self._process_test_batch(acc, test, results, conversation_ids)
            except Exception as e:
                acc.error = f"Test execution failed: {e}"
                logger.warning(f"  Test '{test.id}': ERROR - {e}")

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

    def finalize(self) -> TestSummary:
        """Compute final test results from accumulated batch data.

        Returns:
            TestSummary with pass/fail for each test.
        """
        test_results: list[TestResult] = []

        for test in self.tests:
            acc = self._accumulators[test.id]

            if acc.error:
                test_results.append(self._create_error_result(test, acc.error))
                continue

            if acc.total_count == 0 and test.metric:
                test_results.append(
                    self._create_error_result(
                        test, f"Metric '{test.metric}' not found in results"
                    )
                )
                continue

            test_results.append(self._build_final_result(acc))

        summary = TestSummary.from_results(test_results)

        logger.info(
            f"Test results: {summary.passed_tests}/{summary.total_tests} passed "
            f"({summary.pass_rate}%)"
        )
        if summary.high_severity_failures > 0:
            logger.warning(f"  {summary.high_severity_failures} high severity failures")

        return summary

    def get_affected_conversation_ids(self) -> dict[str, list[str | None]]:
        """Return affected conversation IDs per test.

        Call after ``finalize()`` to get the full mapping for persistence
        (e.g. ``test_affected_rows.json``).
        """
        result: dict[str, list[str | None]] = {}
        for test in self.tests:
            acc = self._accumulators[test.id]
            if acc.error:
                result[test.id] = []
                continue
            affected_ids = self._get_affected_ids(acc)
            result[test.id] = affected_ids
        return result

    def _process_test_batch(
        self,
        acc: _TestAccumulator,
        test: TestParams,
        results: dict[str, list[BaseModel] | BaseModel],
        conversation_ids: list[str | None],
    ) -> None:
        """Process a single test against one batch of results."""
        if not test.metric:
            acc.error = "Test requires 'metric' field"
            return

        values = self._extract_metric_values(test.metric, results)
        if not values:
            return

        if test.type != TestType.THRESHOLD:
            acc.error = f"Unknown test type: {test.type}"
            return

        if test.operator is None or test.value is None:
            acc.error = "Threshold test requires 'operator' and 'value'"
            return

        op_func = OPERATORS.get(test.operator)
        if op_func is None:
            acc.error = f"Unknown operator: {test.operator}"
            return

        for orig_idx, value in values:
            conv_id = (
                conversation_ids[orig_idx] if orig_idx < len(conversation_ids) else None
            )
            try:
                if op_func(value, test.value):
                    match_pos = acc.matching_count
                    acc.matching_count += 1
                    acc.matching_conversation_ids.append(conv_id)
                    if len(acc.matching_reasons) < MAX_FAILURE_REASONS:
                        acc.matching_reasons[match_pos] = (
                            f"Flagged: {test.metric} {test.operator} {test.value}"
                            f" (value={value})"
                        )
                else:
                    non_match_pos = acc.non_matching_count
                    acc.non_matching_count += 1
                    acc.non_matching_conversation_ids.append(conv_id)
                    if len(acc.non_matching_reasons) < MAX_FAILURE_REASONS:
                        acc.non_matching_reasons[non_match_pos] = (
                            f"Not flagged: {test.metric} {test.operator} {test.value}"
                            f" (value={value})"
                        )
            except (TypeError, ValueError):
                non_match_pos = acc.non_matching_count
                acc.non_matching_count += 1
                acc.non_matching_conversation_ids.append(conv_id)
                if len(acc.non_matching_reasons) < MAX_FAILURE_REASONS:
                    acc.non_matching_reasons[non_match_pos] = (
                        f"Cannot evaluate: {value}"
                    )

        acc.total_count += len(values)

    def _determine_outcome(
        self, acc: _TestAccumulator
    ) -> tuple[bool, list[str | None], float, dict[int, str]]:
        """Determine pass/fail and select the affected set.

        Returns:
            (passed, affected_ids, affected_pct, failure_reasons)
        """
        test = acc.test
        total_count = acc.total_count
        matching_count = acc.matching_count

        if total_count > 0:
            matching_pct = 100.0 * matching_count / total_count
            non_matching_pct = 100.0 * acc.non_matching_count / total_count
        else:
            matching_pct = 0.0
            non_matching_pct = 0.0

        passed = True
        affected_ids: list[str | None] = []
        affected_pct = 0.0
        failure_reasons: dict[int, str] = {}

        if test.max_percentage is not None and matching_pct > test.max_percentage:
            passed = False
            affected_ids = acc.matching_conversation_ids
            affected_pct = matching_pct
            failure_reasons = acc.matching_reasons

        if test.min_percentage is not None and matching_pct < test.min_percentage:
            passed = False
            if not affected_ids:
                affected_ids = acc.non_matching_conversation_ids
                affected_pct = non_matching_pct
                failure_reasons = acc.non_matching_reasons

        if test.max_percentage is None and test.min_percentage is None:
            passed = matching_count == 0
            affected_ids = acc.matching_conversation_ids
            affected_pct = matching_pct
            failure_reasons = acc.matching_reasons

        return passed, affected_ids, affected_pct, failure_reasons

    def _build_final_result(self, acc: _TestAccumulator) -> TestResult:
        """Build the final TestResult from accumulated data."""
        test = acc.test
        passed, affected_ids, affected_pct, failure_reasons = self._determine_outcome(
            acc
        )

        return TestResult(
            test_id=test.id,
            passed=passed,
            severity=TestSeverity(test.severity),
            title=test.title or test.id,
            description=test.description or "",
            metric=test.metric or "",
            affected_count=len(affected_ids),
            total_count=acc.total_count,
            affected_percentage=round(affected_pct, 2),
            threshold=(
                test.max_percentage
                if test.max_percentage is not None
                else test.min_percentage
            ),
            actual_value=None,
            sample_indices=[],  # Not meaningful for batch mode
            all_affected_indices=[],  # Not meaningful for batch mode
            details={
                "operator": test.operator,
                "value": test.value,
                "max_percentage": test.max_percentage,
                "min_percentage": test.min_percentage,
                "matching_count": acc.matching_count,
                "matching_percentage": round(
                    100.0 * acc.matching_count / acc.total_count
                    if acc.total_count > 0
                    else 0.0,
                    2,
                ),
                "failure_reasons": {
                    str(k): v
                    for k, v in list(failure_reasons.items())[:MAX_FAILURE_REASONS]
                },
                "sample_conversation_ids": affected_ids[:MAX_SAMPLE_INDICES],
            },
        )

    def _get_affected_ids(self, acc: _TestAccumulator) -> list[str | None]:
        """Return all affected conversation IDs based on test outcome."""
        _, affected_ids, _, _ = self._determine_outcome(acc)
        return affected_ids

    def _extract_metric_values(
        self,
        metric: str,
        results: dict[str, list[BaseModel] | BaseModel],
    ) -> list[tuple[int, Any]]:
        """Extract values for a metric path like 'analyzer_name.field_name'.

        Returns a list of (original_index, value) tuples so callers can
        correctly align values with conversation IDs even when some items
        have ``None`` metric values.
        """
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

        values = []
        for idx, result in enumerate(analyzer_results):
            value = self._get_nested_value(result, field_path)
            if value is not None:
                values.append((idx, value))

        return values

    def _get_nested_value(self, obj: Any, field_path: list[str]) -> Any:
        """Get a nested field value from a Pydantic model or dict."""
        current: Any = obj
        for i, field_name in enumerate(field_path):
            if isinstance(current, BaseModel):
                if field_name in type(current).model_fields:
                    current = getattr(current, field_name)
                else:
                    values = getattr(current, "values", None)
                    if isinstance(values, dict):
                        return self._traverse_dict(values, field_path[i:])
                    return None
            elif isinstance(current, dict):
                if field_name in current:
                    current = current[field_name]
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
        for field_name in path:
            if isinstance(current, dict) and field_name in current:
                current = current[field_name]
            else:
                return None
        return current
