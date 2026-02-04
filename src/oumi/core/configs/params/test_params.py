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


# Declarative validation configuration
TEST_VALIDATIONS = {
    "threshold": {
        "required": ["metric", "operator", "value"],
        "valid_values": {"operator": ["<", ">", "<=", ">=", "==", "!="]},
    },
    "percentage": {
        "required": ["metric", "condition"],
        "either_required": [["max_percentage", "min_percentage"]],
    },
    "distribution": {
        "required": ["metric", "check", "threshold"],
        "valid_enums": {"check": "DistributionCheck"},
    },
    "regex": {
        "required": ["text_field", "pattern"],
    },
    "contains": {
        "required": ["text_field"],
        "custom": lambda self: (self.value is not None or self.values)
        or "requires 'value' or 'values'",
    },
    "contains-any": {
        "required": ["text_field", "values"],
    },
    "contains-all": {
        "required": ["text_field", "values"],
    },
    "query": {
        "required": ["expression"],
    },
    "outliers": {
        "required": ["metric"],
        "custom": lambda self: (self.std_threshold > 0)
        or "'std_threshold' must be positive",
    },
    "composite": {
        "required": ["tests"],
        "custom": lambda self: (
            self.composite_operator in ["any", "all"]
            or _try_parse_int(self.composite_operator)
        )
        or f"Invalid composite_operator '{self.composite_operator}'",
    },
    "python": {
        "required": ["function"],
    },
}


def _try_parse_int(value: str) -> bool:
    """Try to parse a string as an integer."""
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False


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

        self._validate_enum_field("type", TestType, "test type")
        self._validate_enum_field("severity", TestSeverity, "severity")
        self._validate_enum_field("scope", TestScope, "scope")

        self._validate_by_type()

    def _validate_enum_field(
        self, field_name: str, enum_class: Any, label: str
    ) -> None:
        """Validate that a field matches an enum value.

        Args:
            field_name: Name of the field to validate.
            enum_class: Enum class to validate against.
            label: Human-readable label for error messages.
        """
        value = getattr(self, field_name)
        valid_values = [e.value for e in enum_class]
        if value not in valid_values:
            raise ValueError(
                f"Invalid {label} '{value}' for test '{self.id}'. "
                f"Valid values: {valid_values}"
            )

    def _validate_by_type(self) -> None:
        """Validate fields based on test type using declarative rules."""
        validation_rules = TEST_VALIDATIONS.get(self.type)
        if not validation_rules:
            return

        # Check required fields
        for field_name in validation_rules.get("required", []):
            value = getattr(self, field_name)
            if value is None or (isinstance(value, str) and not value):
                raise ValueError(
                    f"Test '{self.id}': '{field_name}' is required for "
                    f"{self.type} tests."
                )

        # Check either_required (at least one must be set)
        for field_group in validation_rules.get("either_required", []):
            if not any(getattr(self, f) is not None for f in field_group):
                fields_str = "' or '".join(field_group)
                raise ValueError(
                    f"Test '{self.id}': Either '{fields_str}' "
                    f"is required for {self.type} tests."
                )

        # Check valid_values (field must be in allowed list)
        for field_name, valid_values in validation_rules.get(
            "valid_values", {}
        ).items():
            value = getattr(self, field_name)
            if value and value not in valid_values:
                raise ValueError(
                    f"Test '{self.id}': Invalid {field_name} '{value}'. "
                    f"Valid values: {valid_values}"
                )

        # Check valid_enums (field must be in enum)
        for field_name, enum_name in validation_rules.get("valid_enums", {}).items():
            value = getattr(self, field_name)
            if value:
                enum_class = globals()[enum_name]
                valid_values = [e.value for e in enum_class]
                if value not in valid_values:
                    raise ValueError(
                        f"Test '{self.id}': Invalid {field_name} '{value}'. "
                        f"Valid values: {valid_values}"
                    )

        # Run custom validation if provided
        custom_validator = validation_rules.get("custom")
        if custom_validator:
            result = custom_validator(self)
            if isinstance(result, str):
                raise ValueError(f"Test '{self.id}': {result}")

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
