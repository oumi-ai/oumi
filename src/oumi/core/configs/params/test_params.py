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

"""Test configuration parameters for dataset analysis.

This module provides dataclasses for configuring user-defined tests that run
on dataset analysis results. Inspired by promptfoo's declarative assertion system.

Example:
    >>> from oumi.core.configs.params.test_params import TestParams
    >>> test = TestParams(
    ...     id="no_pii",
    ...     type="percentage",
    ...     metric="quality__has_pii",
    ...     condition="== True",
    ...     max_percentage=1.0,
    ...     severity="high",
    ...     title="PII detected in dataset",
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from oumi.core.configs.params.base_params import BaseParams


class TestType(str, Enum):
    """Types of tests that can be run on analysis results."""

    THRESHOLD = "threshold"
    PERCENTAGE = "percentage"
    DISTRIBUTION = "distribution"
    REGEX = "regex"
    CONTAINS = "contains"
    CONTAINS_ANY = "contains-any"
    CONTAINS_ALL = "contains-all"
    QUERY = "query"
    OUTLIERS = "outliers"
    COMPOSITE = "composite"
    PYTHON = "python"


class TestSeverity(str, Enum):
    """Severity levels for test failures."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestScope(str, Enum):
    """Scope at which a test operates."""

    MESSAGE = "message"
    CONVERSATION = "conversation"


class DistributionCheck(str, Enum):
    """Types of distribution checks."""

    MAX_FRACTION = "max_fraction"
    DOMINANT_FRACTION = "dominant_fraction"
    ENTROPY = "entropy"
    UNIQUE_COUNT = "unique_count"
    UNIQUE_RATIO = "unique_ratio"


class CompositeOperator(str, Enum):
    """Operators for combining tests in composite tests."""

    ANY = "any"
    ALL = "all"


@dataclass
class TestParams(BaseParams):
    """Configuration for a single test on analysis results.

    This is a flexible dataclass that supports all test types. Fields are
    optional based on the test type being configured. Validation is performed
    in __finalize_and_validate__ based on the test type.

    Attributes:
        id: Unique identifier for this test.
        type: The type of test (threshold, percentage, regex, etc.).
        severity: How severe a failure of this test is (high, medium, low).
        title: Human-readable title for the test (shown in reports).
        description: Detailed description of what this test checks.
        scope: Whether to run on message or conversation DataFrame.
        negate: If True, invert the test logic (pass becomes fail).

        # Metric-based test fields (threshold, percentage, outliers)
        metric: Column name to check (e.g., "length__token_count").
        operator: Comparison operator for threshold tests (<, >, <=, >=, ==, !=).
        value: Value to compare against for threshold tests.
        condition: Condition string for percentage tests (e.g., "== True", "> 0.5").
        max_percentage: Maximum percentage of samples that can match/fail.
        min_percentage: Minimum percentage of samples that must match.
        std_threshold: Standard deviations for outlier detection.

        # Text-based test fields (regex, contains)
        field: Column name containing text to search (e.g., "text_content").
        pattern: Regex pattern for regex tests.
        values: List of substrings for contains-any/contains-all tests.
        case_sensitive: Whether text matching is case-sensitive.

        # Distribution test fields
        check: Type of distribution check (max_fraction, entropy, etc.).
        threshold: Threshold value for distribution checks.

        # Query test fields
        expression: Pandas query expression string.

        # Composite test fields
        tests: List of sub-test configurations for composite tests.
        composite_operator: How to combine sub-tests (any, all, or min count).

        # Python test fields
        function: Python function code as a string.
    """

    id: str = ""
    type: str = ""
    severity: str = "medium"
    title: str | None = None
    description: str | None = None
    scope: str = "message"
    negate: bool = False
    metric: str | None = None
    operator: str | None = None
    value: float | int | str | None = None
    condition: str | None = None
    max_percentage: float | None = None
    min_percentage: float | None = None
    std_threshold: float = 3.0
    text_field: str | None = None
    pattern: str | None = None
    values: list[str] | None = None
    case_sensitive: bool = False
    check: str | None = None
    threshold: float | None = None
    expression: str | None = None
    tests: list[dict[str, Any]] = field(default_factory=list)
    composite_operator: str = "any"
    function: str | None = None

    def __finalize_and_validate__(self) -> None:
        """Validate test configuration based on test type."""
        if not self.id:
            raise ValueError("Test 'id' is required.")

        if not self.type:
            raise ValueError(f"Test 'type' is required for test '{self.id}'.")

        # Validate test type
        valid_types = [t.value for t in TestType]
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid test type '{self.type}' for test '{self.id}'. "
                f"Valid types: {valid_types}"
            )

        # Validate severity
        valid_severities = [s.value for s in TestSeverity]
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity '{self.severity}' for test '{self.id}'. "
                f"Valid severities: {valid_severities}"
            )

        # Validate scope
        valid_scopes = [s.value for s in TestScope]
        if self.scope not in valid_scopes:
            raise ValueError(
                f"Invalid scope '{self.scope}' for test '{self.id}'. "
                f"Valid scopes: {valid_scopes}"
            )

        # Type-specific validation
        self._validate_by_type()

    def _validate_by_type(self) -> None:
        """Validate fields based on test type."""
        if self.type == TestType.THRESHOLD.value:
            self._validate_threshold_test()
        elif self.type == TestType.PERCENTAGE.value:
            self._validate_percentage_test()
        elif self.type == TestType.DISTRIBUTION.value:
            self._validate_distribution_test()
        elif self.type in (TestType.REGEX.value,):
            self._validate_regex_test()
        elif self.type in (
            TestType.CONTAINS.value,
            TestType.CONTAINS_ANY.value,
            TestType.CONTAINS_ALL.value,
        ):
            self._validate_contains_test()
        elif self.type == TestType.QUERY.value:
            self._validate_query_test()
        elif self.type == TestType.OUTLIERS.value:
            self._validate_outliers_test()
        elif self.type == TestType.COMPOSITE.value:
            self._validate_composite_test()
        elif self.type == TestType.PYTHON.value:
            self._validate_python_test()

    def _validate_threshold_test(self) -> None:
        """Validate threshold test configuration."""
        if not self.metric:
            raise ValueError(
                f"Test '{self.id}': 'metric' is required for threshold tests."
            )
        if not self.operator:
            raise ValueError(
                f"Test '{self.id}': 'operator' is required for threshold tests."
            )
        if self.value is None:
            raise ValueError(
                f"Test '{self.id}': 'value' is required for threshold tests."
            )

        valid_operators = ["<", ">", "<=", ">=", "==", "!="]
        if self.operator not in valid_operators:
            raise ValueError(
                f"Test '{self.id}': Invalid operator '{self.operator}'. "
                f"Valid operators: {valid_operators}"
            )

    def _validate_percentage_test(self) -> None:
        """Validate percentage test configuration."""
        if not self.metric:
            raise ValueError(
                f"Test '{self.id}': 'metric' is required for percentage tests."
            )
        if not self.condition:
            raise ValueError(
                f"Test '{self.id}': 'condition' is required for percentage tests."
            )
        if self.max_percentage is None and self.min_percentage is None:
            raise ValueError(
                f"Test '{self.id}': Either 'max_percentage' or 'min_percentage' "
                "is required for percentage tests."
            )

    def _validate_distribution_test(self) -> None:
        """Validate distribution test configuration."""
        if not self.metric:
            raise ValueError(
                f"Test '{self.id}': 'metric' is required for distribution tests."
            )
        if not self.check:
            raise ValueError(
                f"Test '{self.id}': 'check' is required for distribution tests."
            )
        if self.threshold is None:
            raise ValueError(
                f"Test '{self.id}': 'threshold' is required for distribution tests."
            )

        valid_checks = [c.value for c in DistributionCheck]
        if self.check not in valid_checks:
            raise ValueError(
                f"Test '{self.id}': Invalid distribution check '{self.check}'. "
                f"Valid checks: {valid_checks}"
            )

    def _validate_regex_test(self) -> None:
        """Validate regex test configuration."""
        if not self.text_field:
            raise ValueError(
                f"Test '{self.id}': 'field' (text_field) is required for regex tests."
            )
        if not self.pattern:
            raise ValueError(
                f"Test '{self.id}': 'pattern' is required for regex tests."
            )

    def _validate_contains_test(self) -> None:
        """Validate contains test configuration."""
        if not self.text_field:
            raise ValueError(
                f"Test '{self.id}': 'field' (text_field) is required for "
                "contains tests."
            )

        if self.type == TestType.CONTAINS.value:
            if self.value is None and not self.values:
                raise ValueError(
                    f"Test '{self.id}': 'value' or 'values' is required for "
                    "contains tests."
                )
        else:  # contains-any or contains-all
            if not self.values:
                raise ValueError(
                    f"Test '{self.id}': 'values' is required for {self.type} tests."
                )

    def _validate_query_test(self) -> None:
        """Validate query test configuration."""
        if not self.expression:
            raise ValueError(
                f"Test '{self.id}': 'expression' is required for query tests."
            )

    def _validate_outliers_test(self) -> None:
        """Validate outliers test configuration."""
        if not self.metric:
            raise ValueError(
                f"Test '{self.id}': 'metric' is required for outliers tests."
            )
        if self.std_threshold <= 0:
            raise ValueError(f"Test '{self.id}': 'std_threshold' must be positive.")

    def _validate_composite_test(self) -> None:
        """Validate composite test configuration."""
        if not self.tests:
            raise ValueError(
                f"Test '{self.id}': 'tests' is required for composite tests."
            )

        valid_operators = [o.value for o in CompositeOperator]
        # Also allow integer values for "at least N must pass"
        if self.composite_operator not in valid_operators:
            try:
                int(self.composite_operator)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Test '{self.id}': Invalid composite_operator "
                    f"'{self.composite_operator}'. "
                    f"Valid operators: {valid_operators} or an integer."
                )

    def _validate_python_test(self) -> None:
        """Validate python test configuration."""
        if not self.function:
            raise ValueError(
                f"Test '{self.id}': 'function' is required for python tests."
            )

    def get_title(self) -> str:
        """Get the display title for this test."""
        if self.title:
            return self.title
        return self.id.replace("_", " ").title()

    def get_description(self) -> str:
        """Get the description for this test."""
        if self.description:
            return self.description
        return f"Test of type '{self.type}'."
