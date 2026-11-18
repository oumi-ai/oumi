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

    assert metrics["grpo/frac_reward_zero_std"] == pytest.approx(2 / 3)
    assert metrics["grpo/frac_reward_low_std"] == pytest.approx(2 / 3)
    assert metrics["grpo/reward_group_std/mean"] == pytest.approx(0.19245008972987523)
    assert metrics["grpo/reward_group_std/p50"] == pytest.approx(0.0)
    assert metrics["grpo/reward_group_std/p90"] == pytest.approx(0.46188021535170065)


def test_threshold_counts_low_but_nonzero_std_groups():
    rewards = np.array([0.0, 0.002, 0.0, 1.0])
    groups = np.array(["a", "a", "b", "b"])

    default = compute_grpo_reward_group_metrics(rewards, groups)
    wide = compute_grpo_reward_group_metrics(rewards, groups, low_std_threshold=0.01)

    assert default["grpo/frac_reward_zero_std"] == 0.0
    assert default["grpo/frac_reward_low_std"] == 0.0
    assert wide["grpo/frac_reward_zero_std"] == 0.0
    assert wide["grpo/frac_reward_low_std"] == pytest.approx(0.5)


@pytest.mark.parametrize("threshold", [-1.0, float("nan"), float("inf")])
def test_invalid_threshold_is_refused(threshold):
    with pytest.raises(ValueError, match="low_std_threshold"):
        compute_grpo_reward_group_metrics(
            np.array([1.0, 0.0]), np.array(["a", "a"]), low_std_threshold=threshold
        )


def test_empty_or_mismatched_inputs_are_refused():
    with pytest.raises(ValueError, match="empty"):
        compute_grpo_reward_group_metrics(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="must match"):
        compute_grpo_reward_group_metrics(np.array([1.0, 0.0]), np.array(["a"]))


def test_group_metrics_refuse_non_1d_inputs():
    with pytest.raises(ValueError, match="1-D"):
        compute_grpo_reward_group_metrics(
            np.array([[1.0, 0.0]]), np.array([["a", "a"]])
        )


def test_truncation_ratio_is_measured_against_the_configured_limit():
    ratio = compute_response_truncation_ratio(
        np.array([10, 20, 20]), max_response_length=20
    )

    assert ratio == pytest.approx(2 / 3)


def test_truncation_ratio_is_zero_when_nothing_hits_the_limit():
    ratio = compute_response_truncation_ratio(
        np.array([5, 10, 15]), max_response_length=20
    )

    assert ratio == 0.0


def test_truncation_ratio_refuses_empty_lengths():
    with pytest.raises(ValueError, match="empty"):
        compute_response_truncation_ratio(np.array([]), max_response_length=20)


@pytest.mark.parametrize("limit", [0, -5])
def test_truncation_ratio_refuses_a_non_positive_limit(limit):
    with pytest.raises(ValueError, match="max_response_length"):
        compute_response_truncation_ratio(np.array([5, 10]), max_response_length=limit)


def test_last_output_per_session_keeps_the_highest_index_in_input_order():
    keys = ["u1_0_0", "u1_0_1", "u1_1_0", "u2_0_0"]

    assert last_output_per_session(keys) == ["u1_0_1", "u1_1_0", "u2_0_0"]


def test_last_output_per_session_handles_underscores_in_uid_and_unordered_input():
    keys = ["a_b_0_2", "a_b_0_10", "a_b_0_1"]

    assert last_output_per_session(keys) == ["a_b_0_10"]


@pytest.mark.parametrize("key", ["u1_3", "7", "u1_0_x"])
def test_last_output_per_session_names_a_malformed_key(key):
    with pytest.raises(ValueError, match=key):
        last_output_per_session(["u1_0_0", key])
