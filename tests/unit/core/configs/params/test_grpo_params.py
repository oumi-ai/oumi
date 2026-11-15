import pytest

from oumi.core.configs.params.grpo_params import GrpoParams
from oumi.exceptions import OumiConfigError


def test_tool_agent_rollout_requires_an_environment_config():
    with pytest.raises(OumiConfigError, match="environment_config_path is required"):
        GrpoParams(rollout_function="tool_agent")


def test_environment_config_requires_the_tool_agent_rollout():
    with pytest.raises(OumiConfigError, match="only used by the 'tool_agent'"):
        GrpoParams(environment_config_path="env.yaml")


def test_environment_config_cannot_be_empty():
    with pytest.raises(OumiConfigError, match="cannot be empty"):
        GrpoParams(rollout_function="tool_agent", environment_config_path="")


def test_max_tool_turns_must_be_positive():
    with pytest.raises(OumiConfigError, match="max_tool_turns must be positive"):
        GrpoParams(max_tool_turns=0)


def test_tool_agent_rollout_accepts_its_config():
    params = GrpoParams(
        rollout_function="tool_agent",
        environment_config_path="env.yaml",
        max_tool_turns=2,
    )

    assert params.environment_config_path == "env.yaml"
    assert params.max_tool_turns == 2


def test_rollout_config_is_not_forwarded_to_trl():
    params = GrpoParams(
        rollout_function="tool_agent",
        environment_config_path="env.yaml",
        max_tool_turns=2,
    )

    kwargs = params.to_hf_trainer_kwargs()

    assert "environment_config_path" not in kwargs
    assert "max_tool_turns" not in kwargs
    assert "rollout_function" not in kwargs
