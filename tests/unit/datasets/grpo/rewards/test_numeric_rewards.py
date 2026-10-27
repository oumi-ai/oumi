import pytest

from oumi.datasets.grpo.rewards.numeric_rewards import (
    extract_number,
    numeric_match_reward,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("It costs $9", 9.0),
        ("The total is $8.00.", 8.0),
        ("About 1,234 dollars", 1234.0),
        ("2 + 3 = 5", 5.0),
        ("no number here", None),
    ],
)
def test_extract_number_reads_the_last_number(text, expected):
    assert extract_number(text) == expected


def test_decimal_point_does_not_split_the_number():
    # "$8.00" must be one number (8), not the two numbers 8 and 0.
    assert extract_number("The total is $8.00.") == 8.0


def test_reward_matches_the_expected_number():
    rewards = numeric_match_reward(
        [["The price is $3."], [{"role": "assistant", "content": "It costs 3"}]],
        [3, 3],
    )

    assert rewards == [1.0, 1.0]


def test_reward_rejects_a_different_number():
    assert numeric_match_reward([["It costs $4."]], [3]) == [0.0]


def test_reward_handles_a_scalar_answer():
    assert numeric_match_reward([["It costs $3."], ["It costs $3."]], 3) == [1.0, 1.0]


def test_reward_without_a_number_scores_zero():
    assert numeric_match_reward(["I don't know."], [3]) == [0.0]
