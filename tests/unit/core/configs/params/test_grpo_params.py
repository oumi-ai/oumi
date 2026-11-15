import pytest

from oumi.core.configs.params.grpo_params import GrpoParams
from oumi.exceptions import OumiConfigError


def test_environment_config_path_is_off_by_default():
    assert GrpoParams().environment_config_path is None


def test_environment_config_path_accepts_a_yaml_path():
    params = GrpoParams(environment_config_path="env.yaml")

    assert params.environment_config_path == "env.yaml"


def test_max_tool_calling_iterations_must_be_positive():
    with pytest.raises(
        OumiConfigError, match="max_tool_calling_iterations must be positive"
    ):
        GrpoParams(max_tool_calling_iterations=0)


def test_max_tool_calling_iterations_defaults_to_no_limit():
    """``None`` is forwarded to TRL, which turns it into ``sys.maxsize``."""
    assert GrpoParams().max_tool_calling_iterations is None


def test_environment_config_is_not_forwarded_to_trl():
    """It is consumed to build ``environment_factory``, not a GRPOConfig field."""
    params = GrpoParams(
        environment_config_path="env.yaml",
        max_tool_calling_iterations=2,
    )

    kwargs = params.to_hf_trainer_kwargs()

    assert "environment_config_path" not in kwargs
    assert kwargs["max_tool_calling_iterations"] == 2
