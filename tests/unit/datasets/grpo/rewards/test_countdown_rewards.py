import pytest

from oumi.datasets.grpo.rewards import countdown_reward


@pytest.mark.parametrize(
    "s,nums,target,reward",
    [
        # No valid answer: nothing to parse, so no format credit.
        ("foo bar 1", [], 1, 0),
        # Valid answer format, incorrect numbers.
        ("<answer>1 + 2</answer>", [1, 3], 2, 0.1),
        ("<answer>1 / 2</answer>", [1, 2, 3], 6, 0.1),
        # Invalid equation.
        ("<answer></answer>", [], 1, 0.1),
        ("<answer>1 foo 2 bar 3</answer>", [1, 2, 3], 1, 0.1),
        ("<answer>1.0 * 2.0 * 3.0</answer>", [1, 2, 3], 1, 0.1),
        # Incorrect answer.
        ("<answer>1 + 2 + 3</answer>", [1, 2, 3], 1, 0.1),
        ("<answer> (1 * 2) / 3</answer>", [1, 2, 3], 1, 0.1),
        # Correct answer.
        ("<answer> ( 3 - 2 ) * 1 </answer>", [1, 2, 3], 1, 1),
        ("<answer>(3-2)*1</answer>", [1, 2, 3], 1, 1),
    ],
)
def test_countdown_reward(s, nums, target, reward):
    ground_truth = {"target": target, "numbers": nums}
    assert countdown_reward("data_source", s, ground_truth, {}) == reward


def test_parsed_but_wrong_beats_unparsed():
    """The format-shaping signal GRPO needs: parseable-but-wrong must outscore garbage."""
    ground_truth = {"target": 1, "numbers": [1, 2, 3]}
    unparsed = countdown_reward("countdown", "no tags here", ground_truth, {})
    parsed_wrong = countdown_reward(
        "countdown", "<answer>1 + 2 + 3</answer>", ground_truth, {}
    )
    assert unparsed == 0
    assert parsed_wrong > unparsed
