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

"""Tests for test_params module."""

import pytest

from oumi.core.configs.params.test_params import (
    CompositeOperator,
    DistributionCheck,
    TestParams,
    TestScope,
    TestSeverity,
    TestType,
)

# -----------------------------------------------------------------------------
# Tests: Enums
# -----------------------------------------------------------------------------


def test_test_type_enum_values():
    """Test TestType enum has expected values."""
    assert TestType.THRESHOLD == "threshold"
    assert TestType.PERCENTAGE == "percentage"
    assert TestType.DISTRIBUTION == "distribution"
    assert TestType.REGEX == "regex"
    assert TestType.COMPOSITE == "composite"


def test_test_severity_enum_values():
    """Test TestSeverity enum has expected values."""
    assert TestSeverity.HIGH == "high"
    assert TestSeverity.MEDIUM == "medium"
    assert TestSeverity.LOW == "low"


def test_test_scope_enum_values():
    """Test TestScope enum has expected values."""
    assert TestScope.MESSAGE == "message"
    assert TestScope.CONVERSATION == "conversation"


def test_distribution_check_enum_values():
    """Test DistributionCheck enum has expected values."""
    assert DistributionCheck.MAX_FRACTION == "max_fraction"
    assert DistributionCheck.ENTROPY == "entropy"
    assert DistributionCheck.UNIQUE_COUNT == "unique_count"


def test_composite_operator_enum_values():
    """Test CompositeOperator enum has expected values."""
    assert CompositeOperator.ANY == "any"
    assert CompositeOperator.ALL == "all"


# -----------------------------------------------------------------------------
# Tests: TestParams Creation
# -----------------------------------------------------------------------------


def test_test_params_minimal():
    """Test creating TestParams with minimal required fields."""
    params = TestParams(
        id="test_1",
        type="threshold",
        metric="length__total_tokens",
        operator=">",
        value=100,
    )
    assert params.id == "test_1"
    assert params.type == "threshold"
    assert params.metric == "length__total_tokens"


def test_test_params_defaults():
    """Test TestParams default values."""
    params = TestParams(id="t", type="threshold", metric="m", operator=">", value=0)
    assert params.severity == "medium"
    assert params.scope == "message"
    assert params.negate is False
    assert params.case_sensitive is False
    assert params.std_threshold == 3.0


def test_test_params_all_fields():
    """Test TestParams with all fields set."""
    params = TestParams(
        id="comprehensive_test",
        type="threshold",
        severity="high",
        title="My Test",
        description="A comprehensive test",
        scope="conversation",
        negate=True,
        metric="quality__score",
        operator=">=",
        value=0.8,
        max_percentage=5.0,
    )
    assert params.id == "comprehensive_test"
    assert params.severity == "high"
    assert params.title == "My Test"
    assert params.negate is True


# -----------------------------------------------------------------------------
# Tests: Validation - Required ID and Type
# -----------------------------------------------------------------------------


def test_validation_missing_id():
    """Test validation fails without id."""
    params = TestParams(type="threshold", metric="m", operator=">", value=0)
    with pytest.raises(ValueError, match="'id' is required"):
        params.finalize_and_validate()


def test_validation_missing_type():
    """Test validation fails without type."""
    params = TestParams(id="test_1", metric="m", operator=">", value=0)
    with pytest.raises(ValueError, match="'type' is required"):
        params.finalize_and_validate()


def test_validation_invalid_type():
    """Test validation fails with invalid type."""
    params = TestParams(id="test_1", type="invalid_type", metric="m")
    with pytest.raises(ValueError, match="Invalid test type"):
        params.finalize_and_validate()


def test_validation_invalid_severity():
    """Test validation fails with invalid severity."""
    params = TestParams(
        id="test_1",
        type="threshold",
        severity="critical",
        metric="m",
        operator=">",
        value=0,
    )
    with pytest.raises(ValueError, match="Invalid severity"):
        params.finalize_and_validate()


def test_validation_invalid_scope():
    """Test validation fails with invalid scope."""
    params = TestParams(
        id="test_1",
        type="threshold",
        scope="dataset",
        metric="m",
        operator=">",
        value=0,
    )
    with pytest.raises(ValueError, match="Invalid scope"):
        params.finalize_and_validate()


# -----------------------------------------------------------------------------
# Tests: Validation - Threshold Tests
# -----------------------------------------------------------------------------


def test_validation_threshold_valid():
    """Test valid threshold test configuration."""
    params = TestParams(
        id="threshold_test",
        type="threshold",
        metric="length__chars",
        operator="<=",
        value=1000,
    )
    params.finalize_and_validate()
    assert params.type == "threshold"


def test_validation_threshold_missing_metric():
    """Test threshold test requires metric."""
    params = TestParams(
        id="test_1",
        type="threshold",
        operator=">",
        value=100,
    )
    with pytest.raises(ValueError, match="'metric' is required"):
        params.finalize_and_validate()


def test_validation_threshold_missing_operator():
    """Test threshold test requires operator."""
    params = TestParams(
        id="test_1",
        type="threshold",
        metric="length__chars",
        value=100,
    )
    with pytest.raises(ValueError, match="'operator' is required"):
        params.finalize_and_validate()


def test_validation_threshold_missing_value():
    """Test threshold test requires value."""
    params = TestParams(
        id="test_1",
        type="threshold",
        metric="length__chars",
        operator=">",
    )
    with pytest.raises(ValueError, match="'value' is required"):
        params.finalize_and_validate()


def test_validation_threshold_invalid_operator():
    """Test threshold test rejects invalid operator."""
    params = TestParams(
        id="test_1",
        type="threshold",
        metric="length__chars",
        operator="~=",
        value=100,
    )
    with pytest.raises(ValueError, match="Invalid operator"):
        params.finalize_and_validate()


def test_validation_threshold_all_operators():
    """Test all valid operators are accepted."""
    operators = ["<", ">", "<=", ">=", "==", "!="]
    for op in operators:
        params = TestParams(
            id=f"test_{op}",
            type="threshold",
            metric="m",
            operator=op,
            value=0,
        )
        params.finalize_and_validate()
        assert params.operator == op


# -----------------------------------------------------------------------------
# Tests: Validation - Percentage Tests
# -----------------------------------------------------------------------------


def test_validation_percentage_valid_max():
    """Test valid percentage test with max_percentage."""
    params = TestParams(
        id="pct_test",
        type="percentage",
        metric="quality__has_pii",
        condition="== True",
        max_percentage=1.0,
    )
    params.finalize_and_validate()
    assert params.max_percentage == 1.0


def test_validation_percentage_valid_min():
    """Test valid percentage test with min_percentage."""
    params = TestParams(
        id="pct_test",
        type="percentage",
        metric="quality__is_valid",
        condition="== True",
        min_percentage=95.0,
    )
    params.finalize_and_validate()
    assert params.min_percentage == 95.0


def test_validation_percentage_missing_condition():
    """Test percentage test requires condition."""
    params = TestParams(
        id="test_1",
        type="percentage",
        metric="quality__valid",
        max_percentage=5.0,
    )
    with pytest.raises(ValueError, match="'condition' is required"):
        params.finalize_and_validate()


def test_validation_percentage_missing_percentage():
    """Test percentage test requires min or max percentage."""
    params = TestParams(
        id="test_1",
        type="percentage",
        metric="quality__valid",
        condition="== True",
    )
    with pytest.raises(ValueError, match="max_percentage.+min_percentage"):
        params.finalize_and_validate()


# -----------------------------------------------------------------------------
# Tests: Validation - Distribution Tests
# -----------------------------------------------------------------------------


def test_validation_distribution_valid():
    """Test valid distribution test."""
    params = TestParams(
        id="dist_test",
        type="distribution",
        metric="role",
        check="max_fraction",
        threshold=0.5,
    )
    params.finalize_and_validate()
    assert params.check == "max_fraction"


def test_validation_distribution_missing_check():
    """Test distribution test requires check."""
    params = TestParams(
        id="test_1",
        type="distribution",
        metric="role",
        threshold=0.5,
    )
    with pytest.raises(ValueError, match="'check' is required"):
        params.finalize_and_validate()


def test_validation_distribution_invalid_check():
    """Test distribution test rejects invalid check type."""
    params = TestParams(
        id="test_1",
        type="distribution",
        metric="role",
        check="invalid_check",
        threshold=0.5,
    )
    with pytest.raises(ValueError, match="Invalid check"):
        params.finalize_and_validate()


# -----------------------------------------------------------------------------
# Tests: Validation - Regex Tests
# -----------------------------------------------------------------------------


def test_validation_regex_valid():
    """Test valid regex test."""
    params = TestParams(
        id="regex_test",
        type="regex",
        text_field="content",
        pattern=r"\d{3}-\d{4}",
        max_percentage=0.0,
    )
    params.finalize_and_validate()
    assert params.pattern == r"\d{3}-\d{4}"


def test_validation_regex_missing_pattern():
    """Test regex test requires pattern."""
    params = TestParams(
        id="test_1",
        type="regex",
        text_field="content",
    )
    with pytest.raises(ValueError, match="'pattern' is required"):
        params.finalize_and_validate()


def test_validation_regex_missing_text_field():
    """Test regex test requires text_field."""
    params = TestParams(
        id="test_1",
        type="regex",
        pattern=r"\d+",
    )
    with pytest.raises(ValueError, match="'text_field' is required"):
        params.finalize_and_validate()


# -----------------------------------------------------------------------------
# Tests: Validation - Query Tests
# -----------------------------------------------------------------------------


def test_validation_query_valid():
    """Test valid query test."""
    params = TestParams(
        id="query_test",
        type="query",
        expression="length__chars > 100 and quality__valid == True",
    )
    params.finalize_and_validate()
    assert "length__chars" in params.expression


def test_validation_query_missing_expression():
    """Test query test requires expression."""
    params = TestParams(
        id="test_1",
        type="query",
    )
    with pytest.raises(ValueError, match="'expression' is required"):
        params.finalize_and_validate()


# -----------------------------------------------------------------------------
# Tests: Validation - Composite Tests
# -----------------------------------------------------------------------------


def test_validation_composite_valid():
    """Test valid composite test."""
    params = TestParams(
        id="composite_test",
        type="composite",
        tests=[
            {
                "id": "sub1",
                "type": "threshold",
                "metric": "m",
                "operator": ">",
                "value": 0,
            },
        ],
        composite_operator="all",
    )
    params.finalize_and_validate()
    assert len(params.tests) == 1


def test_validation_composite_empty_tests():
    """Test composite test with empty tests list.

    Note: Empty list passes validation since the check is for None/empty string.
    This behavior allows lazy initialization of tests list.
    """
    params = TestParams(
        id="test_1",
        type="composite",
        composite_operator="any",
        tests=[],  # Empty list is allowed
    )
    # Should not raise - empty list is not considered "missing"
    params.finalize_and_validate()
    assert params.tests == []


def test_validation_composite_numeric_operator():
    """Test composite test accepts numeric operator."""
    params = TestParams(
        id="composite_test",
        type="composite",
        tests=[
            {
                "id": "s1",
                "type": "threshold",
                "metric": "m",
                "operator": ">",
                "value": 0,
            }
        ],
        composite_operator="2",  # At least 2 must pass
    )
    params.finalize_and_validate()
    assert params.composite_operator == "2"


# -----------------------------------------------------------------------------
# Tests: Validation - Python Tests
# -----------------------------------------------------------------------------


def test_validation_python_valid():
    """Test valid python test."""
    params = TestParams(
        id="python_test",
        type="python",
        function="def check(row): return row['value'] > 0",
    )
    params.finalize_and_validate()
    assert "def check" in params.function


def test_validation_python_missing_function():
    """Test python test requires function."""
    params = TestParams(
        id="test_1",
        type="python",
    )
    with pytest.raises(ValueError, match="'function' is required"):
        params.finalize_and_validate()


# -----------------------------------------------------------------------------
# Tests: Helper Methods
# -----------------------------------------------------------------------------


def test_get_title_custom():
    """Test get_title returns custom title when set."""
    params = TestParams(
        id="test_1",
        type="threshold",
        title="My Custom Title",
        metric="m",
        operator=">",
        value=0,
    )
    assert params.get_title() == "My Custom Title"


def test_get_title_from_id():
    """Test get_title derives title from id when not set."""
    params = TestParams(
        id="check_token_count",
        type="threshold",
        metric="m",
        operator=">",
        value=0,
    )
    assert params.get_title() == "Check Token Count"


def test_get_description_custom():
    """Test get_description returns custom description."""
    params = TestParams(
        id="test_1",
        type="threshold",
        description="Custom description",
        metric="m",
        operator=">",
        value=0,
    )
    assert params.get_description() == "Custom description"


def test_get_description_default():
    """Test get_description returns default when not set."""
    params = TestParams(
        id="test_1",
        type="threshold",
        metric="m",
        operator=">",
        value=0,
    )
    assert "threshold" in params.get_description()
