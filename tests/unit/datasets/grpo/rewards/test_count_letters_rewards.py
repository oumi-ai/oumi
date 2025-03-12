from oumi.datasets.grpo.rewards import compute_letter_count_reward


def test_compute_soft_target_token_length_reward():
    assert compute_letter_count_reward("foo bar 1", target_count=1) == 0
    assert compute_letter_count_reward("foo bar1", target_count=1) == 0
    assert compute_letter_count_reward("foo bar one", target_count=1) == -1
    assert compute_letter_count_reward("11 1", target_count=1) == 0
    assert (
        compute_letter_count_reward(
            "The number of 'r's in strawberry is 10.", target_count=3
        )
        == -7
    )
