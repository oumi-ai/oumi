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

import random

import pytest

from oumi.core.configs.params.synthesis_params import (
    AttributeCombination,
    GeneralSynthesisParams,
    PermutableAttribute,
    PermutableAttributeValue,
)
from oumi.core.synthesis.planner import DatasetPlanner


@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set up a fixed random seed for deterministic tests."""
    random.seed(42)
    yield
    random.seed()  # Reset the seed after the test


@pytest.fixture
def planner():
    return DatasetPlanner()


@pytest.fixture
def mock_permutable_attributes():
    class MockPermutableAttribute1(PermutableAttribute):
        def __init__(self):
            super().__init__(
                id="attr1",
                attribute="test1",
                description="First test attribute",
                possible_values=[
                    PermutableAttributeValue(
                        id="value1",
                        value="value1",
                        description="First value",
                        sample_rate=0.5,
                    ),
                    PermutableAttributeValue(
                        id="value2",
                        value="value2",
                        description="Second value",
                        sample_rate=0.5,
                    ),
                ],
            )

        def get_value_distribution(self):
            return {"value1": 0.5, "value2": 0.5}

    class MockPermutableAttribute2(PermutableAttribute):
        def __init__(self):
            super().__init__(
                id="attr2",
                attribute="test2",
                description="Second test attribute",
                possible_values=[
                    PermutableAttributeValue(
                        id="valueA",
                        value="valueA",
                        description="Value A",
                        sample_rate=0.5,
                    ),
                    PermutableAttributeValue(
                        id="valueB",
                        value="valueB",
                        description="Value B",
                        sample_rate=0.5,
                    ),
                ],
            )

        def get_value_distribution(self):
            return {"valueA": 0.5, "valueB": 0.5}

    return [MockPermutableAttribute1(), MockPermutableAttribute2()]


def test_plan_with_no_permutable_attributes(planner):
    params = GeneralSynthesisParams(permutable_attributes=None)
    result = planner.plan(params, sample_count=5)
    assert isinstance(result, list)
    assert len(result) == 0


def test_plan_with_empty_permutable_attributes(planner, mock_permutable_attributes):
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    params.permutable_attributes = []
    result = planner.plan(params, sample_count=5)
    assert isinstance(result, list)
    assert len(result) == 0


def test_plan_with_zero_samples(planner, mock_permutable_attributes):
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    result = planner.plan(params, sample_count=0)
    assert isinstance(result, list)
    assert len(result) == 0


def test_plan_with_negative_samples(planner, mock_permutable_attributes):
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    with pytest.raises(ValueError, match="Count must be positive"):
        planner.plan(params, sample_count=-1)


def test_plan_with_valid_permutable_attributes(planner, mock_permutable_attributes):
    """Test that the distribution matches expected values with a fixed seed."""
    params = GeneralSynthesisParams(permutable_attributes=mock_permutable_attributes)
    sample_count = 10
    result = planner.plan(params, sample_count=sample_count)

    # With seed 42, we know exactly what values we should get
    expected_values = [
        "value1",
        "value2",
        "value1",
        "value1",
        "value1",
        "value1",
        "value1",
        "value1",
        "value1",
        "value2",
    ]

    for i, sample in enumerate(result):
        value = list(sample.values())[0]
        assert value == expected_values[i], (
            f"Value at index {i} does not match expected value"
        )


def test_plan_with_valid_combination_sampling(planner, mock_permutable_attributes):
    """Test that combination sampling works with valid probabilities."""
    combination = {"attr1": "value1", "attr2": "valueA"}
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        combination_sampling=[
            AttributeCombination(combination=combination, sample_rate=0.8)
        ],
    )
    sample_count = 100
    result = planner.plan(params, sample_count=sample_count)

    # Count how many samples match our forced combination
    matching_samples = sum(
        1
        for sample in result
        if all(sample.get(attr) == val for attr, val in combination.items())
    )

    # With seed 42 and 0.8 probability, we expect 8 matches
    assert matching_samples == 78, (
        "Expected 78 samples to match the combination based on seed 42"
    )

    # Check that non-matching samples have the correct distribution
    non_matching_samples = [
        sample
        for sample in result
        if not all(sample.get(attr) == val for attr, val in combination.items())
    ]
    non_matching_attr_1_values = [
        sample.get("attr1") for sample in non_matching_samples
    ]
    non_matching_attr_2_values = [
        sample.get("attr2") for sample in non_matching_samples
    ]
    assert non_matching_attr_1_values.count("value1") == 8, (
        "Expected 8 non-matching samples with value1"
    )
    assert non_matching_attr_1_values.count("value2") == 14, (
        "Expected 14 non-matching samples with value2"
    )
    assert non_matching_attr_2_values.count("valueA") == 7, (
        "Expected 10 non-matching samples with valueA"
    )
    assert non_matching_attr_2_values.count("valueB") == 15, (
        "Expected 10 non-matching samples with valueB"
    )


def test_plan_with_multiple_combinations(planner, mock_permutable_attributes):
    """Test that multiple combinations are sampled correctly."""
    combinations = [
        AttributeCombination(
            combination={"attr1": "value1", "attr2": "valueA"}, sample_rate=0.3
        ),
        AttributeCombination(
            combination={"attr1": "value2", "attr2": "valueB"}, sample_rate=0.3
        ),
    ]
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        combination_sampling=combinations,
    )
    sample_count = 100
    result = planner.plan(params, sample_count=sample_count)

    # Count matches for each combination
    matches = [
        sum(
            1
            for sample in result
            if all(sample.get(attr) == val for attr, val in comb.combination.items())
        )
        for comb in combinations
    ]

    # With seed 42 and 0.3 probability each, we expect 26 and 29 matches
    expected_matches = [26, 29]
    for i, m in enumerate(matches):
        assert m == expected_matches[i], (
            f"Expected {expected_matches[i]} matches for combination {i}"
        )


def test_plan_resamples_on_combination_match(planner, mock_permutable_attributes):
    """Test that samples are redrawn if they accidentally match a combination."""
    # Set up a combination that would be very likely to occur randomly
    forbidden_combination = {"attr1": "value1", "attr2": "valueA"}
    params = GeneralSynthesisParams(
        permutable_attributes=mock_permutable_attributes,
        combination_sampling=[
            AttributeCombination(
                combination=forbidden_combination,
                sample_rate=0.0,  # Explicitly never sample this combination
            )
        ],
    )
    sample_count = 20
    result = planner.plan(params, sample_count=sample_count)

    # Check that none of the random samples match our forbidden combination
    matching_samples = sum(
        1
        for sample in result
        if all(sample.get(attr) == val for attr, val in forbidden_combination.items())
    )
    assert matching_samples == 0, (
        "Expected no samples to match the forbidden combination"
    )
