from oumi.datasets.grpo.rewards import compute_letter_count_reward


def test_compute_soft_target_token_length_reward_simple():
    assert compute_letter_count_reward("foo bar 1", target_count=1) == 0


def test_compute_soft_target_token_length_reward_no_space():
    assert compute_letter_count_reward("foo bar1", target_count=1) == 0


def test_compute_soft_target_token_length_reward_spelled_out_not_found():
    assert compute_letter_count_reward("foo bar one", target_count=1) == -1


def test_compute_soft_target_token_length_reward_two_numbers():
    assert compute_letter_count_reward("11 1", target_count=1) == 0


def test_compute_soft_target_token_length_reward_real_example():
    assert (
        compute_letter_count_reward(
            "The number of 'r's in strawberry is 10.", target_count=3
        )
        == -7
    )
