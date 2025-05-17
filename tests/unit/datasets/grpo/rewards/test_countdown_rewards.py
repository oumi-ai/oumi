import pytest

from oumi.datasets.grpo.rewards import countdown_reward


@pytest.mark.parametrize(
    "s,nums,target,reward",
    [
        # No valid answer
        ("foo bar 1", [], 1, 0),
        # Valid answer format, incorrect numbers
        ("<answer>1 + 2</answer>", [1, 3], 2, 0),
        ("<answer>1 / 2</answer>", [1, 2, 3], 6, 0),
        # Invalid equation
        ("<answer></answer>", [], 1, 0),
        ("<answer>1 foo 2 bar 3</answer>", [1, 2, 3], 1, 0),
        ("<answer>1.0 * 2.0 * 3.0</answer>", [1, 2, 3], 1, 0),
        # Incorrect answer
        ("<answer>1 + 2 + 3</answer>", [1, 2, 3], 1, 0),
        ("<answer> (1 * 2) / 3</answer>", [1, 2, 3], 1, 0),
        # Correct answer
        ("<answer> ( 3 - 2 ) * 1 </answer>", [1, 2, 3], 1, 1),
        ("<answer>(3-2)*1</answer>", [1, 2, 3], 1, 1),
    ],
)
def test_compute_soft_target_token_length_reward(s, nums, target, reward):
    ground_truth = {"target": target, "numbers": nums}
    assert countdown_reward("data_source", s, ground_truth, {}) == reward
