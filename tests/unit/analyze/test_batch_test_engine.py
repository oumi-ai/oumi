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

"""Tests for the batch test engine module."""

import pytest
from pydantic import BaseModel

from oumi.analyze.testing.batch_engine import BatchTestEngine
from oumi.core.configs.params.test_params import TestParams, TestType

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
    """Create sample analysis results for testing (single batch)."""
    return {
        "length": [
            SampleMetrics(total_tokens=100, total_chars=500),
            SampleMetrics(total_tokens=200, total_chars=1000),
            SampleMetrics(total_tokens=50, total_chars=250),
            SampleMetrics(total_tokens=150, total_chars=750),
        ]
    }


@pytest.fixture
def sample_conversation_ids() -> list[str | None]:
    """Conversation IDs aligned with sample_results."""
    return ["conv_1", "conv_2", "conv_3", "conv_4"]


# -----------------------------------------------------------------------------
# Tests: Initialization
# -----------------------------------------------------------------------------


def test_engine_initialization():
    """Test BatchTestEngine initialization."""
    tests = [
        TestParams(id="t1", type=TestType.THRESHOLD, metric="m"),
        TestParams(
            id="t2", type=TestType.THRESHOLD, metric="n", operator="<=", value=100
        ),
    ]
    engine = BatchTestEngine(tests)
    assert len(engine.tests) == 2


def test_engine_empty_tests():
    """Test BatchTestEngine with empty tests list."""
    engine = BatchTestEngine([])
    summary = engine.finalize()
    assert summary.total_tests == 0


# -----------------------------------------------------------------------------
# Tests: Single Batch (equivalent to TestEngine behavior)
# -----------------------------------------------------------------------------


def test_threshold_all_pass(sample_results, sample_conversation_ids):
    """Test threshold where no values are flagged."""
    tests = [
        TestParams(
            id="min_tokens",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    assert summary.passed_tests == 1
    assert summary.failed_tests == 0


def test_threshold_some_fail(sample_results, sample_conversation_ids):
    """Test threshold where some values are flagged."""
    tests = [
        TestParams(
            id="max_tokens",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator="<",
            value=100,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    # 1 out of 4 values is < 100
    assert summary.failed_tests == 1


def test_threshold_with_max_percentage(sample_results, sample_conversation_ids):
    """Test threshold with max_percentage tolerance."""
    tests = [
        TestParams(
            id="high_tokens",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
            max_percentage=50.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    # 2 out of 4 (50%) have tokens > 100, equals max_percentage
    assert summary.passed_tests == 1


def test_threshold_with_min_percentage(sample_results, sample_conversation_ids):
    """Test threshold with min_percentage requirement."""
    tests = [
        TestParams(
            id="most_valid",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">=",
            value=50,
            min_percentage=100.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    # All 4 have tokens >= 50
    assert summary.passed_tests == 1


def test_threshold_both_min_and_max_percentage(sample_results, sample_conversation_ids):
    """Test threshold with both min and max percentage."""
    tests = [
        TestParams(
            id="bounded_tokens",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=75,
            min_percentage=25.0,
            max_percentage=75.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    # 3 out of 4 (75%) have tokens > 75
    assert summary.passed_tests == 1


# -----------------------------------------------------------------------------
# Tests: Multi-Batch Processing
# -----------------------------------------------------------------------------


def test_multi_batch_accumulation():
    """Test that results accumulate correctly across batches."""
    tests = [
        TestParams(
            id="token_check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
        )
    ]
    engine = BatchTestEngine(tests)

    # Batch 1: 1 matching (200 > 100)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=50, total_chars=100)]},
        ["conv_1"],
    )
    # Batch 2: 1 matching (200 > 100)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=200, total_chars=400)]},
        ["conv_2"],
    )

    summary = engine.finalize()
    result = summary.results[0]

    # 1 out of 2 matched
    assert result.total_count == 2
    assert result.details["matching_count"] == 1
    assert result.details["matching_percentage"] == 50.0


def test_multi_batch_conversation_ids_tracked():
    """Test that conversation IDs are tracked across batches."""
    tests = [
        TestParams(
            id="token_check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
        )
    ]
    engine = BatchTestEngine(tests)

    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=200, total_chars=400)]},
        ["conv_1"],
    )
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=300, total_chars=600)]},
        ["conv_2"],
    )

    affected = engine.get_affected_conversation_ids()
    assert "conv_1" in affected["token_check"]
    assert "conv_2" in affected["token_check"]


def test_multi_batch_percentage_computed_over_total():
    """Test that percentages are computed over the total across all batches."""
    tests = [
        TestParams(
            id="check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
            max_percentage=30.0,
        )
    ]
    engine = BatchTestEngine(tests)

    # Batch 1: 2 items, 1 matching (50%)
    engine.process_batch(
        {
            "length": [
                SampleMetrics(total_tokens=200, total_chars=400),
                SampleMetrics(total_tokens=50, total_chars=100),
            ]
        },
        ["conv_1", "conv_2"],
    )
    # Batch 2: 2 items, 0 matching
    engine.process_batch(
        {
            "length": [
                SampleMetrics(total_tokens=80, total_chars=160),
                SampleMetrics(total_tokens=90, total_chars=180),
            ]
        },
        ["conv_3", "conv_4"],
    )

    summary = engine.finalize()
    result = summary.results[0]

    # 1 matching out of 4 total = 25%, which is under 30% max
    assert result.passed is True
    assert result.details["matching_count"] == 1
    assert result.details["matching_percentage"] == 25.0


# -----------------------------------------------------------------------------
# Tests: Error Handling
# -----------------------------------------------------------------------------


def test_missing_operator():
    """Test error when operator is missing."""
    tests = [
        TestParams(
            id="missing_op",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            value=100,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=50, total_chars=100)]},
        ["conv_1"],
    )
    summary = engine.finalize()

    assert summary.error_tests == 1
    assert summary.results[0].error is not None


def test_unknown_operator():
    """Test error with unknown operator."""
    tests = [
        TestParams(
            id="bad_op",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator="~=",
            value=100,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=50, total_chars=100)]},
        ["conv_1"],
    )
    summary = engine.finalize()

    assert summary.error_tests == 1
    assert summary.results[0].error is not None
    assert "Unknown operator" in summary.results[0].error


def test_missing_metric():
    """Test error when metric not found in results."""
    tests = [
        TestParams(
            id="missing",
            type=TestType.THRESHOLD,
            metric="NonExistent.field",
            operator=">",
            value=0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=50, total_chars=100)]},
        ["conv_1"],
    )
    summary = engine.finalize()

    assert summary.error_tests == 1
    assert summary.results[0].error is not None
    assert "not found" in summary.results[0].error


def test_error_persists_across_batches():
    """Test that once a test errors, subsequent batches are skipped."""
    tests = [
        TestParams(
            id="bad_metric",
            type=TestType.THRESHOLD,
            metric="NonExistent.field",
            operator=">",
            value=0,
        )
    ]
    engine = BatchTestEngine(tests)

    # First batch triggers error
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=50, total_chars=100)]},
        ["conv_1"],
    )
    # Second batch should be skipped
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=200, total_chars=400)]},
        ["conv_2"],
    )

    summary = engine.finalize()
    assert summary.error_tests == 1


def test_empty_first_batch_does_not_block_later_batches():
    """Test that an empty first batch does not permanently error the test."""
    tests = [
        TestParams(
            id="resilient",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
        )
    ]
    engine = BatchTestEngine(tests)

    # First batch has no data for the metric
    engine.process_batch({}, ["conv_1"])

    # Second batch has valid data
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=200, total_chars=400)]},
        ["conv_2"],
    )

    summary = engine.finalize()
    result = summary.results[0]

    assert result.error is None
    assert result.total_count == 1
    assert result.details["matching_count"] == 1


# -----------------------------------------------------------------------------
# Tests: Metric Extraction
# -----------------------------------------------------------------------------


def test_single_result_not_list():
    """Test extracting metric from single result (not list)."""
    tests = [
        TestParams(
            id="single",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": SampleMetrics(total_tokens=100, total_chars=500)},
        ["conv_1"],
    )
    summary = engine.finalize()

    assert summary.passed_tests == 1


def test_actual_value_single_item():
    """Test that actual_value is populated for single-item metrics."""
    tests = [
        TestParams(
            id="single_val",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=42, total_chars=200)]},
        ["conv_1"],
    )
    summary = engine.finalize()

    assert summary.results[0].actual_value == 42.0


def test_actual_value_none_for_multiple_items(
    sample_results, sample_conversation_ids
):
    """Test that actual_value is None when there are multiple items."""
    tests = [
        TestParams(
            id="multi_val",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    assert summary.results[0].actual_value is None


def test_actual_value_none_for_multi_batch():
    """Test that actual_value is None when total items span multiple batches."""
    tests = [
        TestParams(
            id="multi_batch_val",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=42, total_chars=200)]},
        ["conv_1"],
    )
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=99, total_chars=400)]},
        ["conv_2"],
    )
    summary = engine.finalize()

    # Two total items across batches → actual_value should be None
    assert summary.results[0].actual_value is None


def test_nested_dict_values():
    """Test extracting metric from nested dict via values field."""
    tests = [
        TestParams(
            id="nested",
            type=TestType.THRESHOLD,
            metric="custom.score",
            operator=">",
            value=0.5,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"custom": [NestedMetrics(values={"score": 0.8})]},
        ["conv_1"],
    )
    summary = engine.finalize()

    # 0.8 > 0.5 → matching, no percentage thresholds → test fails (matching > 0)
    assert summary.failed_tests == 1


# -----------------------------------------------------------------------------
# Tests: Affected Conversation IDs
# -----------------------------------------------------------------------------


def test_affected_ids_max_percentage_exceeded():
    """Test affected IDs when max_percentage is exceeded."""
    tests = [
        TestParams(
            id="check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
            max_percentage=10.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {
            "length": [
                SampleMetrics(total_tokens=200, total_chars=400),
                SampleMetrics(total_tokens=50, total_chars=100),
            ]
        },
        ["conv_match", "conv_no_match"],
    )

    affected = engine.get_affected_conversation_ids()
    # Matching IDs are the affected ones when max_percentage is exceeded
    assert "conv_match" in affected["check"]
    assert "conv_no_match" not in affected["check"]


def test_affected_ids_min_percentage_not_met():
    """Test affected IDs when min_percentage is not met."""
    tests = [
        TestParams(
            id="check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
            min_percentage=90.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {
            "length": [
                SampleMetrics(total_tokens=200, total_chars=400),
                SampleMetrics(total_tokens=50, total_chars=100),
            ]
        },
        ["conv_match", "conv_no_match"],
    )

    affected = engine.get_affected_conversation_ids()
    # Non-matching IDs are affected when min_percentage not met
    assert "conv_no_match" in affected["check"]
    assert "conv_match" not in affected["check"]


def test_affected_ids_empty_when_passing():
    """Test affected IDs are empty when test passes."""
    tests = [
        TestParams(
            id="check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
            max_percentage=50.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(
        {"length": [SampleMetrics(total_tokens=50, total_chars=100)]},
        ["conv_1"],
    )

    affected = engine.get_affected_conversation_ids()
    assert affected["check"] == []


# -----------------------------------------------------------------------------
# Tests: Multiple Tests
# -----------------------------------------------------------------------------


def test_multiple_tests(sample_results, sample_conversation_ids):
    """Test running multiple tests in batch mode."""
    tests = [
        TestParams(
            id="test_1",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=1000,
        ),
        TestParams(
            id="test_2",
            type=TestType.THRESHOLD,
            metric="length.total_chars",
            operator=">",
            value=2000,
        ),
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    assert summary.total_tests == 2
    assert summary.passed_tests == 2


# -----------------------------------------------------------------------------
# Tests: Result Details
# -----------------------------------------------------------------------------


def test_result_details_populated(sample_results, sample_conversation_ids):
    """Test that result details contain expected fields."""
    tests = [
        TestParams(
            id="detail_check",
            type=TestType.THRESHOLD,
            metric="length.total_tokens",
            operator=">",
            value=100,
            max_percentage=80.0,
        )
    ]
    engine = BatchTestEngine(tests)
    engine.process_batch(sample_results, sample_conversation_ids)
    summary = engine.finalize()

    result = summary.results[0]
    assert result.details["operator"] == ">"
    assert result.details["value"] == 100
    assert result.details["max_percentage"] == 80.0
    assert "matching_count" in result.details
    assert "matching_percentage" in result.details
    assert "failure_reasons" in result.details
    assert "sample_conversation_ids" in result.details
