import pytest

from oumi.datasets.grpo.rewards import (
    compute_sharp_target_token_length_reward,
    compute_soft_target_token_length_reward,
)


def test_compute_soft_target_token_length_reward():
    assert compute_soft_target_token_length_reward(
        0, target_tokens=10
    ) == pytest.approx(0, 1e-5)

    assert compute_soft_target_token_length_reward(
        1, target_tokens=10
    ) == pytest.approx(0.24596031, 1e-5)

    assert compute_soft_target_token_length_reward(
        5, target_tokens=10
    ) == pytest.approx(0.824360635, 1e-5)

    assert compute_soft_target_token_length_reward(
        10, target_tokens=10
    ) == pytest.approx(1.0, 1e-5)

    assert compute_soft_target_token_length_reward(
        20, target_tokens=10
    ) == pytest.approx(0.7357588, 1e-5)

    assert compute_soft_target_token_length_reward(
        100, target_tokens=10
    ) == pytest.approx(0.001234098, 1e-5)

    assert compute_soft_target_token_length_reward(
        1000, target_tokens=10
    ) == pytest.approx(0, 1e-5)


def test_compute_sharp_target_token_length_reward():
    assert compute_sharp_target_token_length_reward(0, target_tokens=10) == -10
    assert compute_sharp_target_token_length_reward(1, target_tokens=10) == -9
    assert compute_sharp_target_token_length_reward(10, target_tokens=10) == 0
    assert compute_sharp_target_token_length_reward(20, target_tokens=19) == -1
    assert compute_sharp_target_token_length_reward(100, target_tokens=20) == -80
