import math

import pytest
from omegaconf import OmegaConf

from oumi.core.registry import RegistryType, register
from oumi.core.trainers.verl_grpo_rewards import (
    VERL_REWARD_SCORE_KEY,
    combined_reward,
    resolve_reward_functions,
)


@register("test_combined_constant", RegistryType.REWARD_FUNCTION)
def _constant_reward(data_source, solution_str, ground_truth, extra_info, value=1.0):
    return value


@register("test_combined_length", RegistryType.REWARD_FUNCTION)
def _length_reward(data_source, solution_str, ground_truth, extra_info):
    return float(len(solution_str))


@register("test_combined_dict", RegistryType.REWARD_FUNCTION)
def _dict_reward(data_source, solution_str, ground_truth, extra_info):
    return {VERL_REWARD_SCORE_KEY: 0.5, "detail": 7.0}


@register("test_combined_nan", RegistryType.REWARD_FUNCTION)
def _nan_reward(data_source, solution_str, ground_truth, extra_info):
    return math.nan


@register("test_combined_recording", RegistryType.REWARD_FUNCTION)
def _recording_reward(data_source, solution_str, ground_truth, extra_info, **kwargs):
    _recording_reward.calls.append(  # type: ignore[attr-defined]
        {
            "data_source": data_source,
            "solution_str": solution_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            **kwargs,
        }
    )
    return 0.0


_recording_reward.calls = []  # type: ignore[attr-defined]


def _call(**kwargs):
    return combined_reward("ds", "abcd", "gt", {"index": 3}, **kwargs)


def test_combined_reward_sums_with_default_weights():
    output = _call(
        reward_function_names=["test_combined_constant", "test_combined_length"]
    )

    assert output == {
        VERL_REWARD_SCORE_KEY: 5.0,
        "test_combined_constant": 1.0,
        "test_combined_length": 4.0,
    }


def test_combined_reward_applies_weights():
    output = _call(
        reward_function_names=["test_combined_constant", "test_combined_length"],
        reward_weights=[2.0, 0.5],
    )

    assert output[VERL_REWARD_SCORE_KEY] == pytest.approx(2.0 + 2.0)
    # Per-function scores are reported unweighted, as TRL logs them.
    assert output["test_combined_length"] == 4.0


def test_combined_reward_forwards_per_function_kwargs_and_call_args():
    _recording_reward.calls.clear()  # type: ignore[attr-defined]

    output = _call(
        reward_function_names=["test_combined_constant", "test_combined_recording"],
        reward_function_kwargs={
            "test_combined_constant": {"value": 3.0},
            "test_combined_recording": {"method": "strict"},
        },
    )

    assert output["test_combined_constant"] == 3.0
    assert _recording_reward.calls == [  # type: ignore[attr-defined]
        {
            "data_source": "ds",
            "solution_str": "abcd",
            "ground_truth": "gt",
            "extra_info": {"index": 3},
            "method": "strict",
        }
    ]


def test_combined_reward_reads_score_from_a_dict_result():
    output = _call(
        reward_function_names=["test_combined_dict", "test_combined_constant"]
    )

    assert output == {
        VERL_REWARD_SCORE_KEY: 1.5,
        "test_combined_dict": 0.5,
        "test_combined_constant": 1.0,
    }


def test_combined_reward_treats_nan_as_zero_like_trl():
    output = _call(
        reward_function_names=["test_combined_nan", "test_combined_constant"]
    )

    assert output[VERL_REWARD_SCORE_KEY] == 1.0
    assert math.isnan(output["test_combined_nan"])


def test_combined_reward_accepts_omegaconf_containers():
    # verl hands reward_kwargs through its OmegaConf config.
    cfg = OmegaConf.create(
        {
            "reward_function_names": ["test_combined_constant", "test_combined_length"],
            "reward_weights": [1.0, 0.25],
            "reward_function_kwargs": {"test_combined_constant": {"value": 2.0}},
        }
    )

    output = _call(
        reward_function_names=cfg.reward_function_names,
        reward_weights=cfg.reward_weights,
        reward_function_kwargs=cfg.reward_function_kwargs,
    )

    assert output[VERL_REWARD_SCORE_KEY] == pytest.approx(2.0 + 1.0)


def test_combined_reward_rejects_mismatched_weights():
    with pytest.raises(ValueError, match=r"Number of reward_weights \(1\) must match"):
        _call(
            reward_function_names=["test_combined_constant", "test_combined_length"],
            reward_weights=[1.0],
        )


def test_combined_reward_rejects_unknown_function():
    with pytest.raises(ValueError, match="'no_such_reward' is not registered"):
        _call(reward_function_names=["test_combined_constant", "no_such_reward"])


@pytest.mark.parametrize(
    ("returned", "match"),
    [
        ({"detail": 1.0}, "without a 'score' key"),
        ("1.0", "returned str"),
        (True, "returned bool"),
    ],
)
def test_combined_reward_rejects_unusable_results(returned, match):
    @register(
        f"test_combined_bad_{type(returned).__name__}", RegistryType.REWARD_FUNCTION
    )
    def _bad_reward(data_source, solution_str, ground_truth, extra_info):
        return returned

    with pytest.raises((ValueError, TypeError), match=match):
        _call(reward_function_names=[f"test_combined_bad_{type(returned).__name__}"])


def test_resolve_reward_functions_finds_built_in_rewards():
    functions = resolve_reward_functions(["gsm8k", "countdown"])

    assert functions[0].__name__ == "gsm8k_reward"
    assert len(functions) == 2
