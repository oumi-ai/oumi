# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import cast
from unittest.mock import patch

import pytest
import trl
from transformers.trainer_pt_utils import AcceleratorConfig

from oumi.core.configs.params.training_params import TrainerType, TrainingParams
from oumi.exceptions import OumiConfigError


def _grpo_config(params: TrainingParams) -> "trl.GRPOConfig":
    """``to_hf`` is typed as the union of TRL configs; narrow it to the GRPO one."""
    return cast("trl.GRPOConfig", params.to_hf())


def test_fractional_eval_steps_passed_to_trl_config():
    params = TrainingParams(
        trainer_type=TrainerType.TRL_SFT,
        eval_strategy="steps",
        eval_steps=0.1,
    )

    assert params.to_hf().eval_steps == 0.1


@pytest.mark.parametrize("trainer_type", [TrainerType.OUMI, TrainerType.VERL_GRPO])
def test_fractional_eval_steps_rejected_for_non_hf_trainers(trainer_type):
    with pytest.raises(
        OumiConfigError,
        match="Fractional eval_steps are only supported by Hugging Face and TRL",
    ):
        TrainingParams(trainer_type=trainer_type, eval_steps=0.1)


@pytest.mark.parametrize("eval_steps", [-0.1, 0.0, 1.0, 1.5, float("inf")])
def test_fractional_eval_steps_must_be_ratio(eval_steps):
    with pytest.raises(
        OumiConfigError,
        match="Fractional eval_steps must be greater than 0 and less than 1",
    ):
        TrainingParams(eval_steps=eval_steps)


def test_integer_eval_steps_supported_for_oumi_trainer():
    params = TrainingParams(trainer_type=TrainerType.OUMI, eval_steps=20)

    assert params.eval_steps == 20


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_token_tracking_kwargs_v5_returns_only_num_input_tokens_seen(_mock):
    """In transformers v5, only include_num_input_tokens_seen should be returned."""
    params = TrainingParams(include_performance_metrics=True)

    result = params._get_token_tracking_kwargs()

    assert result == {"include_num_input_tokens_seen": True}
    assert "include_tokens_per_second" not in result


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_token_tracking_kwargs_v5_with_metrics_disabled(_mock):
    """In transformers v5, disabled metrics should return False."""
    params = TrainingParams(include_performance_metrics=False)

    result = params._get_token_tracking_kwargs()

    assert result == {"include_num_input_tokens_seen": False}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_token_tracking_kwargs_v4_returns_both_params(_mock):
    """In transformers v4, both token tracking params should be returned."""
    params = TrainingParams(include_performance_metrics=True)

    result = params._get_token_tracking_kwargs()

    assert result == {
        "include_tokens_per_second": True,
        "include_num_input_tokens_seen": True,
    }


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_token_tracking_kwargs_v4_with_metrics_disabled(_mock):
    """In transformers v4, disabled metrics should return False for both."""
    params = TrainingParams(include_performance_metrics=False)

    result = params._get_token_tracking_kwargs()

    assert result == {
        "include_tokens_per_second": False,
        "include_num_input_tokens_seen": False,
    }


# Tests for _get_warmup_kwargs


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_with_warmup_ratio(_mock):
    """In transformers v5, warmup_ratio should be converted to warmup_steps."""
    params = TrainingParams(warmup_ratio=0.1)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 0.1}
    assert "warmup_ratio" not in result


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_with_warmup_steps(_mock):
    """In transformers v5, warmup_steps should be passed as-is."""
    params = TrainingParams(warmup_steps=100)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 100}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_warmup_steps_takes_precedence(_mock):
    """In transformers v5, warmup_steps should take precedence over warmup_ratio."""
    params = TrainingParams(warmup_ratio=0.1, warmup_steps=50)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 50}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=True)
def test_get_warmup_kwargs_v5_defaults_to_zero(_mock):
    """In transformers v5, default should be warmup_steps=0."""
    params = TrainingParams()

    result = params._get_warmup_kwargs()

    assert result == {"warmup_steps": 0}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_warmup_kwargs_v4_returns_both_params(_mock):
    """In transformers v4, both warmup_ratio and warmup_steps should be returned."""
    params = TrainingParams(warmup_ratio=0.1, warmup_steps=50)

    result = params._get_warmup_kwargs()

    assert result == {"warmup_ratio": 0.1, "warmup_steps": 50}


@patch("oumi.utils.packaging.is_transformers_v5", return_value=False)
def test_get_warmup_kwargs_v4_defaults(_mock):
    """In transformers v4, defaults should be warmup_ratio=0.0, warmup_steps=0."""
    params = TrainingParams()

    result = params._get_warmup_kwargs()

    assert result == {"warmup_ratio": 0.0, "warmup_steps": 0}


def test_trainer_kwargs_accelerator_config_merges_with_defaults():
    params = TrainingParams(
        trainer_type=TrainerType.TRL_SFT,
        trainer_kwargs={
            "accelerator_config": {
                "gradient_accumulation_kwargs": {"sync_each_batch": True},
                "even_batches": False,
            }
        },
    )

    accelerator_config = params.to_hf().accelerator_config

    assert isinstance(accelerator_config, AcceleratorConfig)
    assert accelerator_config.gradient_accumulation_kwargs == {"sync_each_batch": True}
    assert accelerator_config.even_batches is False
    assert accelerator_config.use_seedable_sampler is True
    # to_hf must not mutate the stored trainer_kwargs.
    assert "accelerator_config" in params.trainer_kwargs


def test_trainer_kwargs_accelerator_config_must_be_dict():
    params = TrainingParams(
        trainer_type=TrainerType.TRL_SFT,
        trainer_kwargs={"accelerator_config": "accelerate.json"},
    )

    with pytest.raises(OumiConfigError, match="must be a dict, got str"):
        params.to_hf()


@pytest.mark.parametrize("trainer_type", [TrainerType.TRL_GRPO, TrainerType.VERL_GRPO])
def test_several_reward_functions_are_accepted(trainer_type):
    params = TrainingParams(
        trainer_type=trainer_type,
        reward_functions=["gsm8k", "soft_target_token_length"],
        reward_weights=[1.0, 0.2],
    )

    assert params.reward_weights == [1.0, 0.2]


def test_reward_weights_length_must_match_reward_functions():
    with pytest.raises(OumiConfigError, match="reward_weights has 1 entries but"):
        TrainingParams(
            trainer_type=TrainerType.VERL_GRPO,
            reward_functions=["gsm8k", "soft_target_token_length"],
            reward_weights=[1.0],
        )


def test_reward_weights_ignore_blank_reward_function_names():
    params = TrainingParams(
        trainer_type=TrainerType.VERL_GRPO,
        reward_functions=["", "gsm8k", ""],
        reward_weights=[2.0],
    )

    assert params.reward_weights == [2.0]


def test_reward_weights_require_reward_functions():
    with pytest.raises(OumiConfigError, match="requires reward_functions"):
        TrainingParams(trainer_type=TrainerType.VERL_GRPO, reward_weights=[1.0])


@pytest.mark.parametrize("weights", [["1.0"], [True], [None]])
def test_reward_weights_must_be_numbers(weights):
    with pytest.raises(OumiConfigError, match="must be numbers"):
        TrainingParams(
            trainer_type=TrainerType.VERL_GRPO,
            reward_functions=["gsm8k"],
            reward_weights=weights,
        )


def test_reward_weights_reach_the_trl_grpo_config():
    params = TrainingParams(
        trainer_type=TrainerType.TRL_GRPO,
        reward_functions=["gsm8k", "soft_target_token_length"],
        reward_weights=[1.0, 0.2],
    )

    assert _grpo_config(params).reward_weights == [1.0, 0.2]


def test_reward_weights_set_twice_is_rejected():
    # The typed field and the trainer_kwargs escape hatch would silently
    # disagree, so setting both is a config error.
    params = TrainingParams(
        trainer_type=TrainerType.TRL_GRPO,
        reward_functions=["gsm8k", "countdown"],
        reward_weights=[3.0, 3.0],
        trainer_kwargs={"reward_weights": [1.0, 0.2]},
    )

    with pytest.raises(OumiConfigError, match="set both in trainer_kwargs"):
        params.to_hf()


def test_reward_weights_in_trainer_kwargs_alone_still_works():
    params = TrainingParams(
        trainer_type=TrainerType.TRL_GRPO,
        reward_functions=["gsm8k", "countdown"],
        trainer_kwargs={"reward_weights": [1.0, 0.2]},
    )

    assert _grpo_config(params).reward_weights == [1.0, 0.2]


def test_unset_reward_weights_leave_the_trl_default():
    params = TrainingParams(
        trainer_type=TrainerType.TRL_GRPO, reward_functions=["gsm8k"]
    )

    assert _grpo_config(params).reward_weights is None
