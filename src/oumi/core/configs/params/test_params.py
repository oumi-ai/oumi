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

"""Test configuration parameters for the typed analyzer framework.

This module provides configuration classes for defining tests on analysis results.
"""

from dataclasses import dataclass, field
from enum import Enum


class TestType(str, Enum):
    """Type of test to run."""

    THRESHOLD = "threshold"
    PERCENTAGE = "percentage"
    RANGE = "range"


class TestSeverity(str, Enum):
    """Severity level of test failures."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestScope(str, Enum):
    """Scope at which test is evaluated."""

    MESSAGE = "message"
    CONVERSATION = "conversation"
    DATASET = "dataset"


class CompositeOperator(str, Enum):
    """Operators for combining multiple conditions."""

    AND = "and"
    OR = "or"


@dataclass
class DistributionCheck:
    """Configuration for distribution-based checks.

    Attributes:
        metric: Path to the metric to check.
        expected_mean: Expected mean value.
        expected_std: Expected standard deviation.
        tolerance: Tolerance for deviation from expected values.
    """

    metric: str
    expected_mean: float | None = None
    expected_std: float | None = None
    tolerance: float = 0.1


@dataclass
class TestConfig:
    """Configuration for a single test.

    Attributes:
        id: Unique identifier for the test.
        type: Type of test (threshold, percentage, range).
        metric: Path to the metric to test.
        severity: Severity level of failures.
        title: Human-readable title.
        description: Description of what the test checks.
        operator: Comparison operator for threshold tests.
        value: Value to compare against.
        condition: Condition string for percentage tests.
        max_percentage: Maximum allowed percentage.
        min_percentage: Minimum required percentage.
        min_value: Minimum value for range tests.
        max_value: Maximum value for range tests.
    """

    id: str
    type: TestType
    metric: str
    severity: TestSeverity = TestSeverity.MEDIUM
    title: str = ""
    description: str = ""
    operator: str | None = None
    value: float | int | str | None = None
    condition: str | None = None
    max_percentage: float | None = None
    min_percentage: float | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class TestParams:
    """Parameters for configuring test execution.

    Attributes:
        tests: List of test configurations.
        fail_fast: Whether to stop on first failure.
        verbose: Whether to show detailed output.
    """

    tests: list[TestConfig] = field(default_factory=list)
    fail_fast: bool = False
    verbose: bool = True
