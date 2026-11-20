import numpy as np
import pytest

from oumi_verl_ext.metrics import (
    compute_grpo_reward_group_metrics,
    compute_response_truncation_ratio,
    last_output_per_session,
)


def test_group_metrics_for_mixed_groups():
    rewards = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0.5])
    groups = np.array(list("aaaabbbbc"))

    metrics = compute_grpo_reward_group_metrics(rewards, groups)

    assert metrics == pytest.approx(
        {
            "grpo/frac_reward_zero_std": 2 / 3,
            "grpo/frac_reward_low_std": 2 / 3,
            "grpo/reward_group_std/mean": 0.19245008972987523,
            "grpo/reward_group_std/p50": 0.0,
            "grpo/reward_group_std/p90": 0.46188021535170065,
        }
    )


def test_threshold_counts_low_but_nonzero_std_groups():
    rewards = np.array([0.0, 0.002, 0.0, 1.0])
    groups = np.array(["a", "a", "b", "b"])

    default = compute_grpo_reward_group_metrics(rewards, groups)
    wide = compute_grpo_reward_group_metrics(rewards, groups, low_std_threshold=0.01)

    assert default["grpo/frac_reward_low_std"] == 0.0
    assert wide["grpo/frac_reward_zero_std"] == 0.0
    assert wide["grpo/frac_reward_low_std"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("rewards", "groups", "threshold", "match"),
    [
        ([], [], 1e-3, "empty"),
        ([1.0, 0.0], ["a"], 1e-3, "must match"),
        ([[1.0, 0.0]], [["a", "a"]], 1e-3, "1-D"),
        ([1.0, 0.0], ["a", "a"], -1.0, "low_std_threshold"),
        ([1.0, 0.0], ["a", "a"], float("nan"), "low_std_threshold"),
    ],
)
def test_group_metrics_refuse_bad_inputs(rewards, groups, threshold, match):
    with pytest.raises(ValueError, match=match):
        compute_grpo_reward_group_metrics(
            np.array(rewards), np.array(groups), low_std_threshold=threshold
        )


@pytest.mark.parametrize(
    ("lengths", "expected"), [([10, 20, 20], 2 / 3), ([5, 10, 15], 0.0)]
)
def test_truncation_ratio_is_measured_against_the_configured_limit(lengths, expected):
    ratio = compute_response_truncation_ratio(np.array(lengths), max_response_length=20)

    assert ratio == pytest.approx(expected)


@pytest.mark.parametrize(
    ("lengths", "limit", "match"),
    [([], 20, "empty"), ([5, 10], 0, "max_response_length")],
)
def test_truncation_ratio_refuses_bad_inputs(lengths, limit, match):
    with pytest.raises(ValueError, match=match):
        compute_response_truncation_ratio(np.array(lengths), max_response_length=limit)


def test_last_output_per_session_keeps_the_highest_index_in_input_order():
    keys = ["a_b_0_2", "u1_1_0", "a_b_0_10", "a_b_0_1", "u1_0_0"]

    assert last_output_per_session(keys) == ["u1_1_0", "a_b_0_10", "u1_0_0"]


@pytest.mark.parametrize("key", ["u1_3", "7", "u1_0_x"])
def test_last_output_per_session_names_a_malformed_key(key):
    with pytest.raises(ValueError, match=key):
        last_output_per_session(["u1_0_0", key])
