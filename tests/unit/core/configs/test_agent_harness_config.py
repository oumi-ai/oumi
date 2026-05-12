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

"""Contract tests for ``AgentHarnessConfig``."""

import pytest

import oumi.environments  # noqa: F401  populates env registry
from oumi.core.configs.agent_harness_config import AgentHarnessConfig
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams


def _faq_tool(tool_id: str = "answer_faq") -> ToolParams:
    return ToolParams(
        id=tool_id,
        name="AnswerFAQ",
        description="Answer a FAQ question.",
    )


def _synthetic_env(
    *,
    env_id: str = "faq",
    tools: list[ToolParams] | None = None,
) -> EnvironmentParams:
    return EnvironmentParams(
        id=env_id,
        name="FAQ",
        description="FAQ tools",
        env_type="synthetic",
        tools=tools or [_faq_tool()],
        env_kwargs={"system_prompt": "Answer FAQs."},
    )


def test_default_harness_config_rejects_zero_environments():
    """Default config (no envs) fails validation."""
    config = AgentHarnessConfig()
    with pytest.raises(ValueError, match="at least one environment"):
        config.finalize_and_validate()


def test_harness_config_rejects_environment_with_zero_tools():
    """An env that exposes no tools fails the harness invariant."""
    empty_env = EnvironmentParams(
        id="empty",
        name="Empty",
        description="No tools.",
        env_type="synthetic",
        tools=[],
        env_kwargs={"system_prompt": "."},
    )
    config = AgentHarnessConfig(
        environment=EnvironmentConfig(environments=[empty_env]),
    )
    with pytest.raises(ValueError, match="zero tools"):
        config.finalize_and_validate()


def test_harness_config_validates_with_one_env_one_tool():
    """Smallest valid config: one env, one tool, default inference."""
    env_config = EnvironmentConfig(environments=[_synthetic_env()])
    config = AgentHarnessConfig(environment=env_config)

    config.finalize_and_validate()

    assert isinstance(config.inference, InferenceConfig)
    assert isinstance(config.environment, EnvironmentConfig)
    assert config.system_prompt == ""
    assert [t.id for t in config.environment.all_tools] == ["answer_faq"]
    assert config.environment.tool_environment_map == {"answer_faq": "faq"}


def test_harness_config_cascades_environment_validation():
    """Duplicate tool ids across envs fail at construction (via EnvironmentConfig)."""
    duplicated = _faq_tool("dup_tool")
    env_a = _synthetic_env(env_id="a", tools=[duplicated])
    env_b = _synthetic_env(env_id="b", tools=[_faq_tool("dup_tool")])
    with pytest.raises(ValueError, match="duplicate tool id"):
        AgentHarnessConfig(
            environment=EnvironmentConfig(environments=[env_a, env_b]),
        )


def test_harness_config_preserves_system_prompt():
    config = AgentHarnessConfig(
        environment=EnvironmentConfig(environments=[_synthetic_env()]),
        system_prompt="You are a helpful EHR assistant.",
    )
    config.finalize_and_validate()
    assert config.system_prompt == "You are a helpful EHR assistant."


def test_harness_config_resolves_env_var_interpolation_when_enabled(
    tmp_path, monkeypatch
):
    """``ignore_interpolation=False`` resolves ``${oc.env:...}`` in env_kwargs."""
    monkeypatch.setenv("AGENT_TEST_DB_PATH", "/tmp/agent_test_db_path.sqlite")
    yaml_text = """
inference:
  model:
    model_name: dummy-model
environment:
  environments:
    - id: test_env
      env_type: synthetic
      name: Test
      description: Test env.
      env_kwargs:
        system_prompt: "Hi."
        path: ${oc.env:AGENT_TEST_DB_PATH,/tmp/fallback.db}
      tools:
        - id: noop
          name: Noop
          description: A no-op tool.
"""
    config_path = tmp_path / "harness.yaml"
    config_path.write_text(yaml_text)

    cfg = AgentHarnessConfig.from_yaml_and_arg_list(
        str(config_path), [], ignore_interpolation=False
    )
    env_kwargs = cfg.environment.environments[0].env_kwargs
    assert env_kwargs is not None
    assert env_kwargs["path"] == "/tmp/agent_test_db_path.sqlite"


def test_harness_config_round_trips_through_dict_init():
    """Nested-dict init works (the OmegaConf -> dict path used by from_yaml)."""
    raw = {
        "inference": {},
        "environment": {
            "environments": [
                {
                    "id": "faq",
                    "name": "FAQ",
                    "description": "FAQ tools",
                    "env_type": "synthetic",
                    "tools": [
                        {
                            "id": "answer_faq",
                            "name": "AnswerFAQ",
                            "description": "Answer a FAQ question.",
                        }
                    ],
                    "env_kwargs": {"system_prompt": "Answer FAQs."},
                }
            ]
        },
        "system_prompt": "Be concise.",
    }
    config = AgentHarnessConfig(
        inference=InferenceConfig(**raw["inference"]),
        environment=EnvironmentConfig(**raw["environment"]),
        system_prompt=raw["system_prompt"],
    )
    config.finalize_and_validate()
    assert config.environment.get_tool("answer_faq") is not None
