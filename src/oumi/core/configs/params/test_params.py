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
    """Compare a metric against a threshold value."""

    PERCENTAGE = "percentage"
    """Check the percentage of samples matching a condition."""

    DISTRIBUTION = "distribution"
    """Check distribution properties of a metric."""

    REGEX = "regex"
    """Match a regex pattern against text content."""

    CONTAINS = "contains"
    """Check if text contains a specific substring."""

    CONTAINS_ANY = "contains-any"
    """Check if text contains any of multiple substrings."""

    CONTAINS_ALL = "contains-all"
    """Check if text contains all of multiple substrings."""

    QUERY = "query"
    """Execute a pandas query expression."""

    OUTLIERS = "outliers"
    """Detect statistical outliers in a metric."""

    COMPOSITE = "composite"
    """Combine multiple tests with logical operators."""

    PYTHON = "python"
    """Execute a custom Python function."""


class TestSeverity(str, Enum):
    """Severity levels for test failures."""

    HIGH = "high"
    """Critical issue that should be addressed before training."""

    MEDIUM = "medium"
    """Significant issue that may affect training quality."""

    LOW = "low"
    """Minor issue or informational finding."""


class TestScope(str, Enum):
    """Scope at which a test operates."""

    MESSAGE = "message"
    """Test runs on message-level DataFrame."""

    CONVERSATION = "conversation"
    """Test runs on conversation-level DataFrame."""


class DistributionCheck(str, Enum):
    """Types of distribution checks."""

    MAX_FRACTION = "max_fraction"
    """Check if any value exceeds a fraction threshold."""

    DOMINANT_FRACTION = "dominant_fraction"
    """Check if the most common value exceeds a fraction threshold."""

    ENTROPY = "entropy"
    """Check if distribution entropy is within bounds."""

    UNIQUE_COUNT = "unique_count"
    """Check the number of unique values."""

    UNIQUE_RATIO = "unique_ratio"
    """Check the ratio of unique values to total."""


class CompositeOperator(str, Enum):
    """Operators for combining tests in composite tests."""

    ANY = "any"
    """Pass if any sub-test passes (OR logic)."""

    ALL = "all"
    """Pass only if all sub-tests pass (AND logic)."""


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

    # Required fields
    id: str = ""
    """Unique identifier for this test."""

    type: str = ""
    """Test type (threshold, percentage, regex, query, composite, python, etc.)."""

    # Common optional fields
    severity: str = "medium"
    """Severity level: high, medium, or low."""

    title: str | None = None
    """Human-readable title for reports."""

    description: str | None = None
    """Detailed description of the test."""

    scope: str = "message"
    """Scope: message or conversation."""

    negate: bool = False
    """If True, invert the test logic."""

    # Metric-based test fields
    metric: str | None = None
    """Column name to check (e.g., 'length__token_count')."""

    operator: str | None = None
    """Comparison operator: <, >, <=, >=, ==, !=."""

    value: float | int | str | None = None
    """Value to compare against."""

    condition: str | None = None
    """Condition string for percentage tests (e.g., '== True')."""

    max_percentage: float | None = None
    """Maximum percentage of samples that can match/fail."""

    min_percentage: float | None = None
    """Minimum percentage of samples that must match."""

    std_threshold: float = 3.0
    """Standard deviations for outlier detection."""

    # Text-based test fields
    text_field: str | None = None
    """Column name containing text to search (alias: 'field' in YAML)."""

    pattern: str | None = None
    """Regex pattern for regex tests."""

    values: list[str] | None = None
    """List of substrings for contains-any/contains-all tests."""

    case_sensitive: bool = False
    """Whether text matching is case-sensitive."""

    # Distribution test fields
    check: str | None = None
    """Type of distribution check (max_fraction, dominant_fraction, entropy, etc.)."""

    threshold: float | None = None
    """Threshold value for distribution checks."""

    # Query test fields
    expression: str | None = None
    """Pandas query expression string."""

    # Composite test fields
    tests: list[dict[str, Any]] = field(default_factory=list)
    """List of sub-test configurations for composite tests."""

    composite_operator: str = "any"
    """How to combine sub-tests: any, all, or integer (min count)."""

    # Python test fields
    function: str | None = None
    """Python function code as a string."""

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
                f"Test '{self.id}': 'field' (text_field) is required for contains tests."
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
        """Get the display title for this test.

        Returns:
            The title if set, otherwise a generated title from the id.
        """
        if self.title:
            return self.title
        # Convert id to title case: "no_pii" -> "No Pii"
        return self.id.replace("_", " ").title()

    def get_description(self) -> str:
        """Get the description for this test.

        Returns:
            The description if set, otherwise a generated description.
        """
        if self.description:
            return self.description

        # Generate description based on test type
        if self.type == TestType.THRESHOLD.value:
            return (
                f"Check that {self.metric} {self.operator} {self.value} "
                f"for samples in {self.scope} scope."
            )
        elif self.type == TestType.PERCENTAGE.value:
            bound = f"max {self.max_percentage}%" if self.max_percentage else ""
            if self.min_percentage:
                bound = f"min {self.min_percentage}%"
            return f"Check that {bound} of samples have {self.metric} {self.condition}."
        elif self.type == TestType.QUERY.value:
            return f"Check samples matching query: {self.expression}"
        elif self.type == TestType.REGEX.value:
            return f"Check for pattern '{self.pattern}' in {self.text_field}."
        elif self.type in (
            TestType.CONTAINS.value,
            TestType.CONTAINS_ANY.value,
            TestType.CONTAINS_ALL.value,
        ):
            vals = self.values or [self.value]
            return f"Check for substrings {vals} in {self.text_field}."
        elif self.type == TestType.OUTLIERS.value:
            return (
                f"Check for outliers in {self.metric} "
                f"({self.std_threshold} std threshold)."
            )
        elif self.type == TestType.DISTRIBUTION.value:
            return f"Check distribution of {self.metric} using {self.check}."
        elif self.type == TestType.COMPOSITE.value:
            return f"Composite test with {len(self.tests)} sub-tests."
        elif self.type == TestType.PYTHON.value:
            return "Custom Python function test."
        else:
            return f"Test of type '{self.type}'."
