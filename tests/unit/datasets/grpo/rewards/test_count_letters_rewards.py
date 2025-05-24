import pytest

from oumi.datasets.grpo.rewards import compute_letter_count_reward


@pytest.mark.parametrize(
    "s,target_count,reward",
    [
        # No valid answer
        ("foo bar 1", 1, -3),
        # Valid correct answer
        (r"\boxed{1}", 1, 0.0),
        # Valid correct answer
        (r"\boxed{+1}", 1, 0.0),
        # Valid incorrect answer
        (r"\boxed{4}", 1, -1.71429),
        # Valid incorrect answer
        (r"\boxed{-1}", 1, -1.6),
        # Invalid answer
        (r"The answer is \boxed{one}", 0, -3.0),
        # Conflicting answers
        (r"\boxed{1} \boxed{2}", 1, -3.0),
        # Valid incorrect answer
        (r"The number of 'r's in strawberry is \boxed{10}.", 3, -1.86667),
    ],
)
def test_compute_soft_target_token_length_reward(s, target_count, reward):
    calculated_reward = compute_letter_count_reward(s, target_count=target_count)
    assert calculated_reward == pytest.approx(reward, rel=1e-3)
