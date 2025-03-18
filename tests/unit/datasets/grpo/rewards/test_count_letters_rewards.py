import pytest

from oumi.datasets.grpo.rewards import compute_letter_count_reward


@pytest.mark.parametrize(
    "s,target_count,reward",
    [
        ("foo bar 1", 1, 0),
        ("foo bar1", 1, 0),
        ("foo bar one", 1, -1),
        ("11 1", 1, 0),
        ("The number of 'r's in strawberry is 10.", 3, -7),
    ],
)
def test_compute_soft_target_token_length_reward(s, target_count, reward):
    assert compute_letter_count_reward(s, target_count=target_count) == reward
