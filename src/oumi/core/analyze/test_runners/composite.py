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

"""Composite test runner - combines multiple tests with logical operators."""

from typing import TYPE_CHECKING, Any

import pandas as pd

from oumi.core.analyze.test_result import TestResult
from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.configs.params.test_params import CompositeOperator, TestParams

if TYPE_CHECKING:
    from oumi.core.analyze.test_engine import TestEngine


class CompositeTestRunner(BaseTestRunner):
    """Runner for composite tests that combine multiple sub-tests.

    Supports logical operators:
    - any: Pass if ANY sub-test passes
    - all: Pass only if ALL sub-tests pass
    - integer N: Pass if at least N sub-tests pass

    Example config:
        - id: critical_issues
          type: composite
          composite_operator: any
          severity: high
          title: "Critical safety or privacy issues"
          tests:
            - type: percentage
              metric: "quality__has_pii"
              condition: "== True"
              max_percentage: 1.0
            - type: percentage
              metric: "safety__is_unsafe"
              condition: "== True"
              max_percentage: 0.5
    """

    def __init__(self, test_engine: "TestEngine | None" = None):
        """Initialize with optional reference to parent test engine.

        Args:
            test_engine: Parent TestEngine for running sub-tests.
        """
        self._test_engine = test_engine

    def set_test_engine(self, test_engine: "TestEngine") -> None:
        """Set the test engine for running sub-tests.

        Args:
            test_engine: TestEngine instance.
        """
        self._test_engine = test_engine

    def run(
        self,
        test: TestParams,
        message_df: pd.DataFrame,
        conversation_df: pd.DataFrame,
        summary: dict[str, Any],
    ) -> TestResult:
        """Execute the composite test.

        Args:
            test: Test configuration with tests list and composite_operator.
            message_df: Message-level DataFrame.
            conversation_df: Conversation-level DataFrame.
            summary: Analysis summary.

        Returns:
            TestResult indicating pass/fail based on sub-test results.
        """
        if self._test_engine is None:
            return self.create_error_result(
                test, "CompositeTestRunner requires a TestEngine instance"
            )

        if not test.tests:
            return self.create_error_result(
                test, "No sub-tests specified for composite test"
            )

        # Run all sub-tests
        sub_results: list[TestResult] = []
        for i, sub_test_dict in enumerate(test.tests):
            # Create TestParams from dict
            sub_test = self._create_sub_test(sub_test_dict, test.id, i)
            if sub_test is None:
                sub_results.append(
                    self.create_error_result(
                        test, f"Invalid sub-test configuration at index {i}"
                    )
                )
                continue

            # Run the sub-test using the test engine
            sub_result = self._test_engine.run_single_test(
                sub_test, message_df, conversation_df, summary
            )
            sub_results.append(sub_result)

        # Determine overall pass/fail based on operator
        passed, details = self._evaluate_composite(test, sub_results)

        # Aggregate affected samples (union of all sub-test affected samples)
        all_affected_indices = set()
        total_affected = 0
        for result in sub_results:
            total_affected += result.affected_samples
            all_affected_indices.update(result.sample_indices)

        # Get total samples from first sub-test (they should all be the same scope)
        total_samples = sub_results[0].total_samples if sub_results else 0

        return self.create_result(
            test=test,
            passed=passed,
            affected_samples=total_affected,
            total_samples=total_samples,
            details={
                "composite_operator": test.composite_operator,
                "sub_tests_count": len(sub_results),
                "sub_tests_passed": sum(1 for r in sub_results if r.passed),
                "sub_tests_failed": sum(1 for r in sub_results if not r.passed),
                "sub_results": [r.to_dict() for r in sub_results],
                **details,
            },
            sample_indices=list(all_affected_indices),
        )

    def _create_sub_test(
        self,
        sub_test_dict: dict[str, Any],
        parent_id: str,
        index: int,
    ) -> TestParams | None:
        """Create a TestParams from a sub-test dictionary.

        Args:
            sub_test_dict: Dictionary with sub-test configuration.
            parent_id: ID of the parent composite test.
            index: Index of this sub-test.

        Returns:
            TestParams instance or None if invalid.
        """
        try:
            # Generate ID if not provided
            if "id" not in sub_test_dict:
                sub_test_dict["id"] = f"{parent_id}_sub_{index}"

            # Inherit severity from parent if not specified
            if "severity" not in sub_test_dict:
                sub_test_dict["severity"] = "medium"

            return TestParams(**sub_test_dict)
        except Exception:
            return None

    def _evaluate_composite(
        self,
        test: TestParams,
        sub_results: list[TestResult],
    ) -> tuple[bool, dict[str, Any]]:
        """Evaluate composite test result based on operator.

        Args:
            test: Composite test configuration.
            sub_results: List of sub-test results.

        Returns:
            Tuple of (passed, details_dict).
        """
        passed_count = sum(1 for r in sub_results if r.passed)
        failed_count = len(sub_results) - passed_count
        total_count = len(sub_results)

        operator = test.composite_operator.lower()

        if operator == CompositeOperator.ANY.value:
            # Pass if ANY sub-test passes
            passed = passed_count > 0
            logic = "any"
        elif operator == CompositeOperator.ALL.value:
            # Pass only if ALL sub-tests pass
            passed = passed_count == total_count
            logic = "all"
        else:
            # Interpret as integer: pass if at least N sub-tests pass
            try:
                min_required = int(operator)
                passed = passed_count >= min_required
                logic = f"at_least_{min_required}"
            except ValueError:
                # Invalid operator, fail safely
                passed = False
                logic = "invalid"

        return passed, {
            "logic": logic,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "total_count": total_count,
        }
