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

"""Test result data model for dataset analysis.

This module provides the data model for representing the results of running
user-defined tests on dataset analysis results.

Example:
    >>> from oumi.core.analyze.test_result import TestResult
    >>> result = TestResult(
    ...     test_id="no_pii",
    ...     test_type="percentage",
    ...     passed=True,
    ...     severity="high",
    ...     title="PII detected in dataset",
    ...     description="Check that max 1% of samples have PII.",
    ...     affected_samples=5,
    ...     total_samples=1000,
    ...     threshold=1.0,
    ...     actual_value=0.5,
    ... )
    >>> print(result.to_dict())
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestResult:
    """Result of running a single test on analysis data.

    Attributes:
        test_id: Unique identifier from the test configuration.
        test_type: Type of test that was run (threshold, percentage, etc.).
        passed: Whether the test passed (True) or failed (False).
        severity: Severity level of this test (high, medium, low).
        title: Human-readable title of the test.
        description: Detailed description of what was checked.
        affected_samples: Number of samples that triggered the test condition.
        total_samples: Total number of samples that were checked.
        affected_percentage: Percentage of samples that triggered the condition.
        threshold: The threshold value that was checked against (if applicable).
        actual_value: The actual computed value (if applicable).
        details: Additional test-specific details.
        sample_indices: Indices of affected samples (limited to max 20 for display).
        scope: The scope at which this test was run (message or conversation).
        metric: The metric column that was checked (if applicable).
        error: Error message if the test failed to run.
    """

    # Core identification
    test_id: str
    """Unique identifier from the test configuration."""

    test_type: str
    """Type of test (threshold, percentage, regex, query, etc.)."""

    # Result
    passed: bool
    """Whether the test passed."""

    # Metadata
    severity: str
    """Severity level: high, medium, or low."""

    title: str
    """Human-readable title for display."""

    description: str
    """Detailed description of what was checked."""

    # Metrics
    affected_samples: int = 0
    """Number of samples that triggered the test condition."""

    total_samples: int = 0
    """Total number of samples checked."""

    affected_percentage: float = 0.0
    """Percentage of samples affected (0-100)."""

    # Thresholds and values
    threshold: float | None = None
    """The threshold that was checked against."""

    actual_value: float | None = None
    """The actual computed value."""

    # Additional context
    details: dict[str, Any] = field(default_factory=dict)
    """Test-specific additional details."""

    sample_indices: list[int] = field(default_factory=list)
    """Indices of affected samples (max 20 for display)."""

    scope: str = "message"
    """Scope: message or conversation."""

    metric: str | None = None
    """The metric column that was checked."""

    error: str | None = None
    """Error message if the test failed to execute."""

    # Maximum number of sample indices to include in output
    MAX_SAMPLE_INDICES: int = field(default=20, repr=False)

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        # Compute affected percentage if not set
        if self.total_samples > 0 and self.affected_percentage == 0.0:
            self.affected_percentage = round(
                (self.affected_samples / self.total_samples) * 100, 2
            )

        # Limit sample indices
        if len(self.sample_indices) > self.MAX_SAMPLE_INDICES:
            self.sample_indices = self.sample_indices[: self.MAX_SAMPLE_INDICES]

    def to_dict(self) -> dict[str, Any]:
        """Convert the test result to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the test result.
        """
        result = {
            "test_id": self.test_id,
            "test_type": self.test_type,
            # Convert numpy.bool_ to Python bool for JSON compatibility
            "passed": bool(self.passed),
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "affected_samples": int(self.affected_samples),
            "total_samples": int(self.total_samples),
            "affected_percentage": float(self.affected_percentage),
            "scope": self.scope,
        }

        # Include optional fields only if they have values
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.actual_value is not None:
            result["actual_value"] = self.actual_value
        if self.metric:
            result["metric"] = self.metric
        if self.details:
            result["details"] = self.details
        if self.sample_indices:
            result["sample_indices"] = self.sample_indices[: self.MAX_SAMPLE_INDICES]
        if self.error:
            result["error"] = self.error

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestResult":
        """Create a TestResult from a dictionary.

        Args:
            data: Dictionary containing test result data.

        Returns:
            TestResult instance.
        """
        return cls(
            test_id=data.get("test_id", ""),
            test_type=data.get("test_type", ""),
            passed=data.get("passed", False),
            severity=data.get("severity", "medium"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            affected_samples=data.get("affected_samples", 0),
            total_samples=data.get("total_samples", 0),
            affected_percentage=data.get("affected_percentage", 0.0),
            threshold=data.get("threshold"),
            actual_value=data.get("actual_value"),
            details=data.get("details", {}),
            sample_indices=data.get("sample_indices", []),
            scope=data.get("scope", "message"),
            metric=data.get("metric"),
            error=data.get("error"),
        )

    @property
    def status(self) -> str:
        """Get a human-readable status string.

        Returns:
            'PASSED', 'FAILED', or 'ERROR'.
        """
        if self.error:
            return "ERROR"
        return "PASSED" if self.passed else "FAILED"

    @property
    def status_emoji(self) -> str:
        """Get a status emoji for display.

        Returns:
            Emoji representing the status.
        """
        if self.error:
            return "⚠️"
        return "✅" if self.passed else "❌"

    def __str__(self) -> str:
        """Return a string representation of the test result."""
        return (
            f"{self.status_emoji} [{self.severity.upper()}] {self.title}: "
            f"{self.status} ({self.affected_samples}/{self.total_samples} affected)"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"TestResult(test_id='{self.test_id}', passed={self.passed}, "
            f"severity='{self.severity}', affected={self.affected_samples}/"
            f"{self.total_samples})"
        )


@dataclass
class TestSummary:
    """Summary of all test results for a dataset analysis run.

    Attributes:
        total_tests: Total number of tests that were run.
        passed_tests: Number of tests that passed.
        failed_tests: Number of tests that failed.
        error_tests: Number of tests that encountered errors.
        high_severity_failures: Number of high severity test failures.
        medium_severity_failures: Number of medium severity test failures.
        low_severity_failures: Number of low severity test failures.
        results: List of all individual test results.
    """

    total_tests: int = 0
    """Total number of tests run."""

    passed_tests: int = 0
    """Number of tests that passed."""

    failed_tests: int = 0
    """Number of tests that failed."""

    error_tests: int = 0
    """Number of tests that encountered errors."""

    high_severity_failures: int = 0
    """Number of high severity failures."""

    medium_severity_failures: int = 0
    """Number of medium severity failures."""

    low_severity_failures: int = 0
    """Number of low severity failures."""

    results: list[TestResult] = field(default_factory=list)
    """All individual test results."""

    @classmethod
    def from_results(cls, results: list[TestResult]) -> "TestSummary":
        """Create a TestSummary from a list of TestResults.

        Args:
            results: List of TestResult objects.

        Returns:
            TestSummary instance with computed statistics.
        """
        summary = cls(results=results)
        summary.total_tests = len(results)

        for result in results:
            if result.error:
                summary.error_tests += 1
            elif result.passed:
                summary.passed_tests += 1
            else:
                summary.failed_tests += 1
                # Count by severity
                if result.severity == "high":
                    summary.high_severity_failures += 1
                elif result.severity == "medium":
                    summary.medium_severity_failures += 1
                else:
                    summary.low_severity_failures += 1

        return summary

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to a dictionary.

        Returns:
            Dictionary representation of the summary.
        """
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "error_tests": self.error_tests,
            "high_severity_failures": self.high_severity_failures,
            "medium_severity_failures": self.medium_severity_failures,
            "low_severity_failures": self.low_severity_failures,
            "pass_rate": self.pass_rate,
            "results": [r.to_dict() for r in self.results],
        }

    @property
    def pass_rate(self) -> float:
        """Calculate the test pass rate.

        Returns:
            Pass rate as a percentage (0-100).
        """
        if self.total_tests == 0:
            return 100.0
        return round((self.passed_tests / self.total_tests) * 100, 2)

    @property
    def has_critical_failures(self) -> bool:
        """Check if there are any high severity failures.

        Returns:
            True if there are high severity failures.
        """
        return self.high_severity_failures > 0

    @property
    def all_passed(self) -> bool:
        """Check if all tests passed.

        Returns:
            True if all tests passed with no errors.
        """
        return self.failed_tests == 0 and self.error_tests == 0

    def get_failed_results(self) -> list[TestResult]:
        """Get only the failed test results.

        Returns:
            List of TestResult objects that failed.
        """
        return [r for r in self.results if not r.passed and not r.error]

    def get_passed_results(self) -> list[TestResult]:
        """Get only the passed test results.

        Returns:
            List of TestResult objects that passed.
        """
        return [r for r in self.results if r.passed]

    def get_error_results(self) -> list[TestResult]:
        """Get only the test results with errors.

        Returns:
            List of TestResult objects that had errors.
        """
        return [r for r in self.results if r.error]

    def get_results_by_severity(self, severity: str) -> list[TestResult]:
        """Get test results filtered by severity.

        Args:
            severity: Severity level to filter by (high, medium, low).

        Returns:
            List of TestResult objects with the specified severity.
        """
        return [r for r in self.results if r.severity == severity]

    def __str__(self) -> str:
        """Return a string summary."""
        return (
            f"TestSummary: {self.passed_tests}/{self.total_tests} passed "
            f"({self.pass_rate}%), "
            f"{self.high_severity_failures} high severity failures"
        )
