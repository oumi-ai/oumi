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

"""Tests for the testing engine module."""

import pytest
from pydantic import BaseModel

from oumi.analyze.testing.engine import TestConfig, TestEngine, TestType
from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary

# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


class SampleMetrics(BaseModel):
    """Sample metrics for testing."""

    total_tokens: int
    total_chars: int
    is_valid: bool = True


class NestedMetrics(BaseModel):
    """Metrics with nested structure."""

    values: dict


@pytest.fixture
def sample_results() -> dict[str, list[BaseModel]]:
    """Create sample analysis results for testing."""
    return {
        "LengthAnalyzer": [
            SampleMetrics(total_tokens=100, total_chars=500),
            SampleMetrics(total_tokens=200, total_chars=1000),
            SampleMetrics(total_tokens=50, total_chars=250),
            SampleMetrics(total_tokens=150, total_chars=750),
        ]
    }


@pytest.fixture
def mixed_results() -> dict[str, list[BaseModel]]:
    """Create results with mixed valid/invalid flags."""
    return {
        "QualityAnalyzer": [
            SampleMetrics(total_tokens=100, total_chars=500, is_valid=True),
            SampleMetrics(total_tokens=200, total_chars=1000, is_valid=False),
            SampleMetrics(total_tokens=50, total_chars=250, is_valid=True),
            SampleMetrics(total_tokens=150, total_chars=750, is_valid=False),
        ]
    }


# -----------------------------------------------------------------------------
# Tests: TestConfig
# -----------------------------------------------------------------------------


def test_test_config_creation():
    """Test creating a TestConfig."""
    config = TestConfig(
        id="test_1",
        type=TestType.THRESHOLD,
        metric="LengthAnalyzer.total_tokens",
        operator=">",
        value=100,
    )
    assert config.id == "test_1"
    assert config.type == TestType.THRESHOLD
    assert config.metric == "LengthAnalyzer.total_tokens"


def test_test_config_defaults():
    """Test TestConfig default values."""
    config = TestConfig(
        id="test_1",
        type=TestType.THRESHOLD,
        metric="test",
    )
    assert config.severity == TestSeverity.MEDIUM
    assert config.title == ""
    assert config.operator is None
    assert config.value is None


# -----------------------------------------------------------------------------
# Tests: TestEngine Initialization
# -----------------------------------------------------------------------------


def test_engine_initialization():
    """Test TestEngine initialization."""
    tests = [
        TestConfig(id="t1", type=TestType.THRESHOLD, metric="m"),
        TestConfig(
            id="t2", type=TestType.THRESHOLD, metric="n", operator="<=", value=100
        ),
    ]
    engine = TestEngine(tests)
    assert len(engine.tests) == 2


def test_engine_empty_tests():
    """Test TestEngine with empty tests list."""
    engine = TestEngine([])
    assert len(engine.tests) == 0


# -----------------------------------------------------------------------------
# Tests: Threshold Tests
# -----------------------------------------------------------------------------


def test_threshold_test_all_pass(sample_results):
    """Test threshold where all values pass."""
    tests = [
        TestConfig(
            id="min_tokens",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator=">=",
            value=50,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.passed_tests == 1
    assert summary.failed_tests == 0


def test_threshold_test_some_fail(sample_results):
    """Test threshold where some values fail."""
    tests = [
        TestConfig(
            id="max_tokens",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator="<",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 3 out of 4 are >= 100, so only 1 passes (50 < 100)
    assert summary.failed_tests == 1


def test_threshold_test_with_max_percentage(sample_results):
    """Test threshold with max_percentage tolerance."""
    tests = [
        TestConfig(
            id="high_tokens",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator=">",
            value=100,
            max_percentage=50.0,  # Allow up to 50% to exceed 100
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 2 out of 4 (50%) have tokens > 100, which equals max_percentage
    assert summary.passed_tests == 1


def test_threshold_test_with_min_percentage(sample_results):
    """Test threshold with min_percentage requirement."""
    tests = [
        TestConfig(
            id="most_valid",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator=">=",
            value=50,
            min_percentage=100.0,  # Require all to pass
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # All 4 have tokens >= 50, so 100% pass
    assert summary.passed_tests == 1


def test_threshold_test_both_min_and_max_percentage(sample_results):
    """Test threshold with both min and max percentage."""
    tests = [
        TestConfig(
            id="bounded_tokens",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator=">",
            value=75,
            min_percentage=25.0,  # At least 25% must exceed
            max_percentage=75.0,  # No more than 75% can exceed
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    # 3 out of 4 (75%) have tokens > 75, which is within bounds
    assert summary.passed_tests == 1


def test_threshold_test_missing_operator():
    """Test threshold test returns error with missing operator."""
    tests = [
        TestConfig(
            id="missing_op",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(
        {"LengthAnalyzer": [SampleMetrics(total_tokens=50, total_chars=100)]}
    )

    assert summary.error_tests == 1
    assert summary.results[0].error is not None


def test_threshold_test_unknown_operator():
    """Test threshold test returns error with unknown operator."""
    tests = [
        TestConfig(
            id="bad_op",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator="~=",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(
        {"LengthAnalyzer": [SampleMetrics(total_tokens=50, total_chars=100)]}
    )

    assert summary.error_tests == 1
    assert "Unknown operator" in summary.results[0].error


# -----------------------------------------------------------------------------
# Tests: Metric Extraction
# -----------------------------------------------------------------------------


def test_extract_metric_not_found():
    """Test that missing metric returns error."""
    tests = [
        TestConfig(
            id="missing_metric",
            type=TestType.THRESHOLD,
            metric="NonExistent.field",
            operator=">",
            value=0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(
        {"LengthAnalyzer": [SampleMetrics(total_tokens=50, total_chars=100)]}
    )

    assert summary.error_tests == 1
    assert "not found" in summary.results[0].error


def test_extract_metric_invalid_format():
    """Test that invalid metric format returns error."""
    tests = [
        TestConfig(
            id="bad_format",
            type=TestType.THRESHOLD,
            metric="no_dot_separator",
            operator=">",
            value=0,
        )
    ]
    engine = TestEngine(tests)
    summary = engine.run(
        {"LengthAnalyzer": [SampleMetrics(total_tokens=50, total_chars=100)]}
    )

    assert summary.error_tests == 1


def test_extract_metric_single_result():
    """Test extracting metric from single result (not list)."""
    tests = [
        TestConfig(
            id="single_result",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator="==",
            value=100,
        )
    ]
    engine = TestEngine(tests)
    # Single result, not a list
    summary = engine.run(
        {"LengthAnalyzer": SampleMetrics(total_tokens=100, total_chars=500)}
    )

    assert summary.passed_tests == 1


# -----------------------------------------------------------------------------
# Tests: Nested Value Traversal
# -----------------------------------------------------------------------------


def test_get_nested_value_from_dict():
    """Test traversing nested dicts."""
    engine = TestEngine([])
    result = engine._traverse_dict({"a": {"b": {"c": 42}}}, ["a", "b", "c"])
    assert result == 42


def test_get_nested_value_dict_missing_key():
    """Test traversing dict with missing key returns None."""
    engine = TestEngine([])
    result = engine._traverse_dict({"a": {"b": 1}}, ["a", "x"])
    assert result is None


def test_get_nested_value_unsupported_type():
    """Test that unsupported types raise TypeError."""
    engine = TestEngine([])

    # Create a non-BaseModel, non-dict object
    class CustomObj:
        pass

    with pytest.raises(TypeError, match="Cannot traverse type"):
        engine._get_nested_value(CustomObj(), ["field"])


# -----------------------------------------------------------------------------
# Tests: TestSummary
# -----------------------------------------------------------------------------


def test_test_summary_from_results():
    """Test creating TestSummary from results."""
    results = [
        TestResult(test_id="t1", passed=True),
        TestResult(test_id="t2", passed=False, severity=TestSeverity.HIGH),
        TestResult(test_id="t3", passed=False, severity=TestSeverity.LOW),
        TestResult(test_id="t4", passed=False, error="Some error"),
    ]
    summary = TestSummary.from_results(results)

    assert summary.total_tests == 4
    assert summary.passed_tests == 1
    assert summary.failed_tests == 2
    assert summary.error_tests == 1
    assert summary.high_severity_failures == 1
    assert summary.low_severity_failures == 1
    assert summary.pass_rate == 25.0


def test_test_summary_empty_results():
    """Test TestSummary with empty results."""
    summary = TestSummary.from_results([])

    assert summary.total_tests == 0
    assert summary.pass_rate == 0.0


def test_test_summary_get_methods():
    """Test TestSummary getter methods."""
    results = [
        TestResult(test_id="t1", passed=True),
        TestResult(test_id="t2", passed=False),
        TestResult(test_id="t3", passed=False, error="Error"),
    ]
    summary = TestSummary.from_results(results)

    assert len(summary.get_passed_results()) == 1
    assert len(summary.get_failed_results()) == 1
    assert len(summary.get_error_results()) == 1


# -----------------------------------------------------------------------------
# Tests: TestResult
# -----------------------------------------------------------------------------


def test_test_result_to_dict():
    """Test TestResult.to_dict() method."""
    result = TestResult(
        test_id="test_1",
        passed=True,
        severity=TestSeverity.HIGH,
        affected_count=5,
        total_count=100,
    )
    data = result.to_dict()

    assert data["test_id"] == "test_1"
    assert data["passed"] is True
    assert data["severity"] == "high"
    assert data["affected_count"] == 5


# -----------------------------------------------------------------------------
# Tests: Engine Run
# -----------------------------------------------------------------------------


def test_engine_run_multiple_tests(sample_results):
    """Test running multiple tests."""
    tests = [
        TestConfig(
            id="test_1",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_tokens",
            operator=">=",
            value=50,
        ),
        TestConfig(
            id="test_2",
            type=TestType.THRESHOLD,
            metric="LengthAnalyzer.total_chars",
            operator="<=",
            value=2000,
        ),
    ]
    engine = TestEngine(tests)
    summary = engine.run(sample_results)

    assert summary.total_tests == 2
    assert summary.passed_tests == 2


def test_engine_handles_test_exception():
    """Test that engine handles exceptions gracefully."""
    tests = [
        TestConfig(
            id="error_test",
            type=TestType.THRESHOLD,
            metric="Bad.metric",
            operator=">",
            value=0,
        )
    ]
    engine = TestEngine(tests)

    # Should not raise, should return error result
    summary = engine.run({})

    assert summary.total_tests == 1
    assert summary.results[0].error is not None
