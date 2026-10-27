import functools

import pytest

from oumi.builders import build_rollout_function
from oumi.core.configs import TrainingParams
from oumi.core.configs.params.grpo_params import GrpoParams
from oumi.datasets.grpo.rollouts.tool_agent import tool_agent_rollout


def _training_params(**grpo_kwargs) -> TrainingParams:
    return TrainingParams(grpo=GrpoParams(**grpo_kwargs))


def test_no_rollout_function_by_default():
    assert build_rollout_function(_training_params()) is None


def test_unknown_rollout_function_is_rejected():
    with pytest.raises(KeyError, match="was not found in the registry"):
        build_rollout_function(_training_params(rollout_function="missing"))


def test_tool_agent_rollout_is_configured_from_grpo_params():
    rollout = build_rollout_function(
        _training_params(
            rollout_function="tool_agent",
            environment_config_path="env.yaml",
            max_tool_turns=2,
        )
    )

    assert isinstance(rollout, functools.partial)
    assert rollout.func is tool_agent_rollout
    assert rollout.keywords == {
        "environment_config_path": "env.yaml",
        "max_tool_turns": 2,
    }
