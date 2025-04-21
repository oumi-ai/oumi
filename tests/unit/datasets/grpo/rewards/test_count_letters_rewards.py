import pytest

from oumi.datasets.grpo.rewards import compute_letter_count_reward


@pytest.mark.parametrize(
    "s,target_count,reward",
    [
        # No valid answer
        ("foo bar 1", 1, -1),
        # Valid correct answer
        (r"\boxed{1}", 1, 0.1),
        # Valid correct answer
        (r"\boxed{+1}", 1, 0.1),
        # Valid incorrect answer
        (r"\boxed{4}", 1, -2.9),
        # Valid incorrect answer
        (r"\boxed{-1}", 1, -1.9),
        # Invalid answer
        (r"The answer is \boxed{one}", 0, 0),
        # Conflicting answers
        (r"\boxed{1} \boxed{2}", 1, -1),
        (r"The number of 'r's in strawberry is \boxed{10}.", 3, -6.9),
    ],
)
def test_compute_soft_target_token_length_reward(s, target_count, reward):
    assert compute_letter_count_reward(s, target_count=target_count) == reward
