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

"""Test result models for the test engine."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TestSeverity(str, Enum):
    """Severity levels for test failures."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestResult(BaseModel):
    """Result of a single test execution.

    Attributes:
        test_id: Unique identifier for the test.
        passed: Whether the test passed.
        severity: Severity level of the test.
        title: Human-readable title.
        description: Description of what the test checks.
        affected_count: Number of samples that failed the test.
        total_count: Total number of samples tested.
        affected_percentage: Percentage of samples affected.
        threshold: The configured threshold for the test.
        actual_value: The actual computed value (for threshold tests).
        sample_indices: Indices of affected samples (limited).
        error: Error message if test execution failed.
        details: Additional details about the test result.
    """

    test_id: str
    passed: bool
    severity: TestSeverity = TestSeverity.MEDIUM
    title: str = ""
    description: str = ""
    affected_count: int = 0
    total_count: int = 0
    affected_percentage: float = 0.0
    threshold: float | None = None
    actual_value: float | None = None
    sample_indices: list[int] = Field(default_factory=list)
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()


class TestSummary(BaseModel):
    """Summary of all test results.

    Attributes:
        results: List of individual test results.
        total_tests: Total number of tests run.
        passed_tests: Number of tests that passed.
        failed_tests: Number of tests that failed.
        error_tests: Number of tests that had errors.
        pass_rate: Percentage of tests that passed.
        high_severity_failures: Number of high severity failures.
        medium_severity_failures: Number of medium severity failures.
        low_severity_failures: Number of low severity failures.
    """

    results: list[TestResult] = Field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    error_tests: int = 0
    pass_rate: float = 0.0
    high_severity_failures: int = 0
    medium_severity_failures: int = 0
    low_severity_failures: int = 0

    @classmethod
    def from_results(cls, results: list[TestResult]) -> "TestSummary":
        """Create a summary from a list of test results.

        Args:
            results: List of test results.

        Returns:
            TestSummary with computed statistics.
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed and not r.error)
        errors = sum(1 for r in results if r.error)
        failed = total - passed - errors

        high_failures = sum(
            1 for r in results if not r.passed and r.severity == TestSeverity.HIGH
        )
        medium_failures = sum(
            1 for r in results if not r.passed and r.severity == TestSeverity.MEDIUM
        )
        low_failures = sum(
            1 for r in results if not r.passed and r.severity == TestSeverity.LOW
        )

        return cls(
            results=results,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            pass_rate=round(100.0 * passed / total, 1) if total > 0 else 0.0,
            high_severity_failures=high_failures,
            medium_severity_failures=medium_failures,
            low_severity_failures=low_failures,
        )

    def get_passed_results(self) -> list[TestResult]:
        """Get all passed test results."""
        return [r for r in self.results if r.passed]

    def get_failed_results(self) -> list[TestResult]:
        """Get all failed test results."""
        return [r for r in self.results if not r.passed and not r.error]

    def get_error_results(self) -> list[TestResult]:
        """Get all test results with errors."""
        return [r for r in self.results if r.error]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return self.model_dump()
