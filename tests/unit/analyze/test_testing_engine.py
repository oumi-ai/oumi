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

"""Tests for the TestEngine."""

from typing import Any

import pytest
from pydantic import BaseModel

from oumi.analyze.testing.engine import TestConfig, TestEngine, TestType
from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary

# -----------------------------------------------------------------------------
# Helper models for test results
# -----------------------------------------------------------------------------


class MockMetrics(BaseModel):
    """Mock metrics model for testing."""

    total_tokens: int = 0
    quality_score: float = 0.0
    has_issue: bool = False


class SparseMetrics(BaseModel):
    """Metrics model with optional field for testing None-filtering."""

    value: int | None = None


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_results() -> dict[str, Any]:
    """Create sample analysis results."""
    return {
        "Length": [
            MockMetrics(total_tokens=100, quality_score=0.8, has_issue=False),
            MockMetrics(total_tokens=200, quality_score=0.6, has_issue=True),
            MockMetrics(total_tokens=5000, quality_score=0.3, has_issue=True),
            MockMetrics(total_tokens=150, quality_score=0.9, has_issue=False),
        ],
    }


@pytest.fixture
def single_result() -> dict[str, Any]:
    """Create a single (dataset-level) result."""
    return {
        "Stats": MockMetrics(total_tokens=500, quality_score=0.75, has_issue=False),
    }


# -----------------------------------------------------------------------------
# TestConfig Tests
# -----------------------------------------------------------------------------


def test_test_config_creation():
    """Test basic TestConfig creation."""
    config = TestConfig(
        id="test_1",
        type=TestType.THRESHOLD,
        metric="Length.total_tokens",
        operator=">",
        value=4096,
    )
    assert config.id == "test_1"
    assert config.type == TestType.THRESHOLD
    assert config.metric == "Length.total_tokens"
    assert config.severity == TestSeverity.MEDIUM


def test_test_config_with_severity():
    """Test TestConfig with custom severity."""
    config = TestConfig(
        id="critical_test",
        type=TestType.THRESHOLD,
        metric="Length.total_tokens",
        severity=TestSeverity.HIGH,
        operator=">",
        value=10000,
    )
    assert config.severity == TestSeverity.HIGH


def test_test_type_enum():
    """Test TestType enum values."""
    assert TestType.THRESHOLD.value == "threshold"
    assert TestType.PERCENTAGE.value == "percentage"
    assert TestType.RANGE.value == "range"


# -----------------------------------------------------------------------------
# TestEngine Initialization Tests
# -----------------------------------------------------------------------------


def test_engine_initialization():
    """Test TestEngine initialization."""
    tests = [
        TestConfig(
            id="t1",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    assert len(engine.tests) == 1


def test_engine_empty_tests():
    """Test TestEngine with no tests."""
    engine = TestEngine([])
    summary = engine.run({})
    assert summary.total_tests == 0
    assert summary.pass_rate == 0.0


# -----------------------------------------------------------------------------
# Threshold Test Tests
# -----------------------------------------------------------------------------


def test_threshold_test_pass(sample_results):
    """Test threshold test that passes."""
    tests = [
        TestConfig(
            id="max_tokens",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value=4096,
            max_percentage=50.0,
            title="Token limit check",
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.total_tests == 1
    assert summary.passed_tests == 1
    result = summary.results[0]
    assert result.passed is True
    assert result.total_count == 4
    # Only 1 out of 4 exceeds 4096 (the 5000 one)
    assert result.affected_count == 1


def test_threshold_test_fail(sample_results):
    """Test threshold test that fails."""
    tests = [
        TestConfig(
            id="max_tokens",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value=4096,
            max_percentage=0.0,  # No samples allowed above threshold
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.failed_tests == 1
    result = summary.results[0]
    assert result.passed is False
    assert result.affected_count == 1


def test_threshold_test_all_must_match(sample_results):
    """Test threshold test with no percentage (all must match)."""
    tests = [
        TestConfig(
            id="all_low",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator="<",
            value=1000,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 5000 > 1000, so one doesn't match -> fail
    assert summary.results[0].passed is False
    assert summary.results[0].affected_count == 1


def test_threshold_test_min_percentage(sample_results):
    """Test threshold test with min_percentage."""
    tests = [
        TestConfig(
            id="min_quality",
            type=TestType.THRESHOLD,
            metric="Length.quality_score",
            operator=">",
            value=0.5,
            min_percentage=50.0,
            title="Minimum quality check",
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 3 out of 4 have quality_score > 0.5 (75%) >= 50%
    assert summary.results[0].passed is True


def test_threshold_test_min_percentage_fail(sample_results):
    """Test threshold test with min_percentage that fails."""
    tests = [
        TestConfig(
            id="min_quality",
            type=TestType.THRESHOLD,
            metric="Length.quality_score",
            operator=">",
            value=0.5,
            min_percentage=90.0,  # Need 90% but only 75% match
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.results[0].passed is False


def test_threshold_missing_operator():
    """Test threshold test with missing operator returns error."""
    tests = [
        TestConfig(
            id="bad_test",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    result = engine.run({"Length": [MockMetrics(total_tokens=50)]})  # type: ignore[arg-type]

    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "operator" in result.results[0].error.lower()


def test_threshold_unknown_operator():
    """Test threshold test with unknown operator."""
    tests = [
        TestConfig(
            id="bad_op",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator="~=",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    result = engine.run({"Length": [MockMetrics(total_tokens=50)]})  # type: ignore[arg-type]

    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "unknown operator" in result.results[0].error.lower()


def test_threshold_operators():
    """Test all supported operators."""
    metrics = [MockMetrics(total_tokens=100)]

    for op, value, should_match in [
        ("<", 200, True),
        ("<", 50, False),
        (">", 50, True),
        (">", 200, False),
        ("<=", 100, True),
        ("<=", 99, False),
        (">=", 100, True),
        (">=", 101, False),
        ("==", 100, True),
        ("==", 99, False),
        ("!=", 99, True),
        ("!=", 100, False),
    ]:
        tests = [
            TestConfig(
                id=f"op_{op}_{value}",
                type=TestType.THRESHOLD,
                metric="Length.total_tokens",
                operator=op,
                value=value,
            )
        ]
        engine = TestEngine(tests)
        result = engine.run({"Length": metrics})  # type: ignore[arg-type]
        assert result.results[0].passed is should_match, (
            f"Operator {op} with value {value}: expected passed={should_match}"
        )


def test_threshold_single_value_actual(single_result):
    """Test that single-value results include actual_value."""
    tests = [
        TestConfig(
            id="check",
            type=TestType.THRESHOLD,
            metric="Stats.total_tokens",
            operator=">",
            value=1000,
            max_percentage=100.0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(single_result)

    result = summary.results[0]
    assert result.actual_value is not None
    assert result.actual_value == 500.0


# -----------------------------------------------------------------------------
# Percentage Test Tests
# -----------------------------------------------------------------------------


def test_percentage_test_pass(sample_results):
    """Test percentage test that passes."""
    tests = [
        TestConfig(
            id="issue_rate",
            type=TestType.PERCENTAGE,
            metric="Length.has_issue",
            condition="== True",
            max_percentage=60.0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 2 out of 4 have has_issue=True (50%) <= 60%
    assert summary.results[0].passed is True


def test_percentage_test_fail(sample_results):
    """Test percentage test that fails."""
    tests = [
        TestConfig(
            id="issue_rate",
            type=TestType.PERCENTAGE,
            metric="Length.has_issue",
            condition="== True",
            max_percentage=25.0,  # 50% > 25%
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.results[0].passed is False


def test_percentage_test_fail_has_failure_reasons(sample_results):
    """Test that max_percentage failure includes failure_reasons."""
    tests = [
        TestConfig(
            id="issue_rate",
            type=TestType.PERCENTAGE,
            metric="Length.has_issue",
            condition="== True",
            max_percentage=25.0,  # 50% > 25% -> fail
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    result = summary.results[0]
    assert result.passed is False
    reasons = result.details.get("failure_reasons", {})
    # Affected samples are the matching ones (has_issue=True at idx 1, 2)
    assert len(reasons) > 0, "failure_reasons should not be empty"


def test_percentage_test_min_percentage(sample_results):
    """Test percentage test with min_percentage."""
    tests = [
        TestConfig(
            id="min_no_issue",
            type=TestType.PERCENTAGE,
            metric="Length.has_issue",
            condition="== False",
            min_percentage=40.0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 2 out of 4 have has_issue=False (50%) >= 40%
    assert summary.results[0].passed is True


def test_percentage_test_numeric_condition(sample_results):
    """Test percentage test with numeric condition."""
    tests = [
        TestConfig(
            id="quality_check",
            type=TestType.PERCENTAGE,
            metric="Length.quality_score",
            condition="> 0.5",
            min_percentage=50.0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 3 out of 4 have quality_score > 0.5 (75%) >= 50%
    assert summary.results[0].passed is True


def test_percentage_test_missing_condition():
    """Test percentage test with missing condition."""
    tests = [
        TestConfig(
            id="bad_test",
            type=TestType.PERCENTAGE,
            metric="Length.has_issue",
            max_percentage=50.0,
        )
    ]
    engine = TestEngine(tests)
    result = engine.run({"Length": [MockMetrics()]})  # type: ignore[arg-type]

    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "condition" in result.results[0].error.lower()


def test_percentage_test_invalid_condition():
    """Test percentage test with invalid condition format."""
    tests = [
        TestConfig(
            id="bad_cond",
            type=TestType.PERCENTAGE,
            metric="Length.has_issue",
            condition="invalid",
            max_percentage=50.0,
        )
    ]
    engine = TestEngine(tests)
    result = engine.run({"Length": [MockMetrics()]})  # type: ignore[arg-type]

    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "invalid condition" in result.results[0].error.lower()


# -----------------------------------------------------------------------------
# Range Test Tests
# -----------------------------------------------------------------------------


def test_range_test_pass(sample_results):
    """Test range test that passes."""
    tests = [
        TestConfig(
            id="token_range",
            type=TestType.RANGE,
            metric="Length.total_tokens",
            min_value=0,
            max_value=10000,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.results[0].passed is True
    assert summary.results[0].affected_count == 0


def test_range_test_fail(sample_results):
    """Test range test that fails."""
    tests = [
        TestConfig(
            id="token_range",
            type=TestType.RANGE,
            metric="Length.total_tokens",
            min_value=100,
            max_value=300,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 5000 is outside the range
    assert summary.results[0].passed is False
    assert summary.results[0].affected_count >= 1


def test_range_test_with_max_percentage(sample_results):
    """Test range test with max_percentage tolerance."""
    tests = [
        TestConfig(
            id="token_range",
            type=TestType.RANGE,
            metric="Length.total_tokens",
            min_value=100,
            max_value=300,
            max_percentage=50.0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 5000 is outside range (1 out of 4 = 25%) <= 50%
    assert summary.results[0].passed is True


def test_range_test_min_only():
    """Test range test with only min_value."""
    tests = [
        TestConfig(
            id="min_check",
            type=TestType.RANGE,
            metric="Length.total_tokens",
            min_value=50,
        )
    ]
    engine = TestEngine(tests)
    result = engine.run(
        {"Length": [MockMetrics(total_tokens=100), MockMetrics(total_tokens=10)]}
    )

    # 10 < 50 is outside range
    assert result.results[0].passed is False
    assert result.results[0].affected_count == 1


def test_range_test_max_only():
    """Test range test with only max_value."""
    tests = [
        TestConfig(
            id="max_check",
            type=TestType.RANGE,
            metric="Length.total_tokens",
            max_value=500,
        )
    ]
    engine = TestEngine(tests)
    result = engine.run(
        {"Length": [MockMetrics(total_tokens=100), MockMetrics(total_tokens=200)]}
    )

    assert result.results[0].passed is True
    assert result.results[0].affected_count == 0


def test_range_test_missing_values():
    """Test range test with no min or max value."""
    tests = [
        TestConfig(
            id="bad_range",
            type=TestType.RANGE,
            metric="Length.total_tokens",
        )
    ]
    engine = TestEngine(tests)
    result = engine.run({"Length": [MockMetrics(total_tokens=100)]})  # type: ignore[arg-type]

    assert result.results[0].passed is False
    assert result.results[0].error is not None
    assert "min_value" in result.results[0].error.lower()


# -----------------------------------------------------------------------------
# Metric Extraction Tests
# -----------------------------------------------------------------------------


def test_metric_not_found():
    """Test handling of missing metric."""
    tests = [
        TestConfig(
            id="missing",
            type=TestType.THRESHOLD,
            metric="Missing.field",
            operator=">",
            value=0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run({"Length": [MockMetrics()]})  # type: ignore[arg-type]

    assert summary.results[0].passed is False
    assert summary.results[0].error is not None
    assert "not found" in summary.results[0].error.lower()


def test_invalid_metric_path():
    """Test handling of invalid metric path (no dot)."""
    tests = [
        TestConfig(
            id="bad_path",
            type=TestType.THRESHOLD,
            metric="nodot",
            operator=">",
            value=0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run({"Length": [MockMetrics()]})  # type: ignore[arg-type]

    assert summary.results[0].passed is False
    assert summary.results[0].error is not None
    assert "not found" in summary.results[0].error.lower()


# -----------------------------------------------------------------------------
# TestSummary Tests
# -----------------------------------------------------------------------------


def test_summary_pass_rate(sample_results):
    """Test summary pass rate calculation."""
    tests = [
        TestConfig(
            id="pass_test",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value=4096,
            max_percentage=50.0,
        ),
        TestConfig(
            id="fail_test",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value=4096,
            max_percentage=0.0,
        ),
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.total_tests == 2
    assert summary.passed_tests == 1
    assert summary.failed_tests == 1
    assert summary.pass_rate == 50.0


def test_summary_severity_counts():
    """Test summary severity counting."""
    results = [
        TestResult(test_id="t1", passed=False, severity=TestSeverity.HIGH),
        TestResult(test_id="t2", passed=False, severity=TestSeverity.HIGH),
        TestResult(test_id="t3", passed=False, severity=TestSeverity.MEDIUM),
        TestResult(test_id="t4", passed=False, severity=TestSeverity.LOW),
        TestResult(test_id="t5", passed=True, severity=TestSeverity.HIGH),
    ]
    summary = TestSummary.from_results(results)

    assert summary.high_severity_failures == 2
    assert summary.medium_severity_failures == 1
    assert summary.low_severity_failures == 1
    assert summary.passed_tests == 1


def test_summary_accessors():
    """Test TestSummary accessor methods."""
    results = [
        TestResult(test_id="pass1", passed=True),
        TestResult(test_id="fail1", passed=False),
        TestResult(test_id="err1", passed=False, error="Something broke"),
    ]
    summary = TestSummary.from_results(results)

    assert len(summary.get_passed_results()) == 1
    assert len(summary.get_failed_results()) == 1
    assert len(summary.get_error_results()) == 1


def test_summary_to_dict():
    """Test TestSummary serialization."""
    results = [TestResult(test_id="t1", passed=True)]
    summary = TestSummary.from_results(results)

    d = summary.to_dict()
    assert d["total_tests"] == 1
    assert d["passed_tests"] == 1
    assert "results" in d


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


def test_test_execution_error():
    """Test that test execution errors are caught gracefully."""
    tests = [
        TestConfig(
            id="error_test",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value="not_a_number",  # Will cause comparison error
        )
    ]
    # Pass values that will cause comparison issues
    engine = TestEngine(tests)
    summary = engine.run({"Length": [MockMetrics(total_tokens=100)]})

    # Should not raise, but result depends on comparison behavior
    assert summary.total_tests == 1


def test_empty_results():
    """Test running tests with empty results."""
    tests = [
        TestConfig(
            id="empty",
            type=TestType.THRESHOLD,
            metric="Length.total_tokens",
            operator=">",
            value=0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run({})

    assert summary.results[0].passed is False
    assert summary.results[0].error is not None
    assert "not found" in summary.results[0].error.lower()


# -----------------------------------------------------------------------------
# Index Preservation Tests
# -----------------------------------------------------------------------------


def test_sample_indices_preserved_with_none_values():
    """Test sample_indices use original indices when Nones filtered.

    When some results have None for a metric field, those entries are skipped.
    The remaining entries' indices should still map back to the correct
    original conversation positions, not positions in the filtered list.
    """
    # Conversations 0, 2, 4 have values; 1, 3 have None (will be filtered)
    results: dict[str, Any] = {
        "Analyzer": [
            SparseMetrics(value=100),  # index 0
            SparseMetrics(value=None),  # index 1 — filtered out
            SparseMetrics(value=5000),  # index 2
            SparseMetrics(value=None),  # index 3 — filtered out
            SparseMetrics(value=200),  # index 4
        ]
    }

    tests = [
        TestConfig(
            id="high_value",
            type=TestType.THRESHOLD,
            metric="Analyzer.value",
            operator=">",
            value=4096,
            max_percentage=0.0,  # Fail if any match
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(results)

    result = summary.results[0]
    assert result.passed is False
    assert result.affected_count == 1
    assert result.total_count == 3  # Only non-None values counted
    # The affected sample should be original index 2, not filtered-list index 1
    assert result.sample_indices == [2]


def test_range_indices_preserved_with_none_values():
    """Test that range test sample_indices use original indices."""
    results: dict[str, Any] = {
        "Analyzer": [
            SparseMetrics(value=50),  # index 0 — in range
            SparseMetrics(value=None),  # index 1 — filtered out
            SparseMetrics(value=None),  # index 2 — filtered out
            SparseMetrics(value=999),  # index 3 — outside range
            SparseMetrics(value=100),  # index 4 — in range
        ]
    }

    tests = [
        TestConfig(
            id="range_check",
            type=TestType.RANGE,
            metric="Analyzer.value",
            min_value=0,
            max_value=500,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(results)

    result = summary.results[0]
    assert result.passed is False
    assert result.affected_count == 1
    # The out-of-range sample is original index 3
    assert result.sample_indices == [3]


def test_percentage_indices_preserved_with_none_values():
    """Test that percentage test sample_indices use original indices."""
    results: dict[str, Any] = {
        "Analyzer": [
            MockMetrics(has_issue=True),  # index 0
            MockMetrics(has_issue=False),  # index 1
            MockMetrics(has_issue=True),  # index 2
        ]
    }

    tests = [
        TestConfig(
            id="issue_check",
            type=TestType.PERCENTAGE,
            metric="Analyzer.has_issue",
            condition="== True",
            max_percentage=0.0,  # No issues allowed
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(results)

    result = summary.results[0]
    assert result.passed is False
    # Affected (matching) indices should be 0 and 2
    assert result.sample_indices == [0, 2]
