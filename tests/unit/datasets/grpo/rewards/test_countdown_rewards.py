import pytest

from oumi.datasets.grpo.rewards import countdown_reward


@pytest.mark.parametrize(
    "s,nums,target,reward",
    [
        # No valid answer
        ("foo bar 1", [], 1, 0),
    ],
)
def test_compute_soft_target_token_length_reward(s, nums, target, reward):
    ground_truth = {"target": target, "numbers": nums}
    assert countdown_reward("data_source", s, ground_truth, {}) == reward
