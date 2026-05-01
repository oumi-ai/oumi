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

import pytest

import oumi.environments  # noqa: F401  populates env registry
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.configs.synthesis_config import SynthesisConfig, SynthesisStrategy
from oumi.core.types.conversation import Role


def test_default_synthesis_config():
    config = SynthesisConfig()

    assert config.strategy == SynthesisStrategy.GENERAL
    assert isinstance(config.strategy_params, GeneralSynthesisParams)
    assert isinstance(config.inference_config, InferenceConfig)
    assert config.num_samples == 1


def test_custom_synthesis_config():
    custom_params = GeneralSynthesisParams()
    custom_inference = InferenceConfig()

    config = SynthesisConfig(
        strategy=SynthesisStrategy.GENERAL,
        strategy_params=custom_params,
        inference_config=custom_inference,
        num_samples=10,
    )

    assert config.strategy == SynthesisStrategy.GENERAL
    assert config.strategy_params is custom_params
    assert config.inference_config is custom_inference
    assert config.num_samples == 10


def _make_faq_tool() -> ToolParams:
    return ToolParams(
        id="answer_faq",
        name="AnswerFAQ",
        description="Answer a FAQ question.",
    )


def _synthetic_env_params(
    *,
    env_id: str = "faq",
    name: str = "FAQ",
    description: str = "FAQ tools",
    system_prompt: str = "Answer FAQs.",
    tools: list[ToolParams] | None = None,
    extra_kwargs: dict | None = None,
) -> EnvironmentParams:
    env_kwargs: dict = {"system_prompt": system_prompt}
    if extra_kwargs:
        env_kwargs.update(extra_kwargs)
    return EnvironmentParams(
        id=env_id,
        name=name,
        description=description,
        env_type="synthetic",
        tools=tools or [_make_faq_tool()],
        env_kwargs=env_kwargs,
    )


def test_synthesis_config_with_top_level_environment_config():
    env_config = EnvironmentConfig(environments=[_synthetic_env_params()])
    params = GeneralSynthesisParams()
    params.multiturn_attributes = []

    config = SynthesisConfig(
        strategy_params=params,
        environment_config=env_config,
    )

    assert config.environment_config is not None
    assert config.environment_config == env_config
    assert config.environment_config.tool_environment_map == {"answer_faq": "faq"}


def test_synthesis_config_loads_environment_config_from_path(tmp_path):
    env_config_path = tmp_path / "environments.yaml"
    env_config = EnvironmentConfig(environments=[_synthetic_env_params()])
    env_config.to_yaml(env_config_path)

    config = SynthesisConfig(environment_config_path=str(env_config_path))

    assert config.environment_config is not None
    assert config.environment_config.all_tools[0].id == "answer_faq"


def test_synthesis_config_validates_available_tools():
    env_config = EnvironmentConfig(environments=[_synthetic_env_params()])
    params = GeneralSynthesisParams(
        multiturn_attributes=[
            MultiTurnAttribute(
                id="chat",
                min_turns=1,
                max_turns=2,
                role_instruction_messages={
                    Role.USER: "You are a user.",
                    Role.ASSISTANT: "You are an assistant.",
                },
                available_tools=["answer_faq"],
            )
        ]
    )

    config = SynthesisConfig(
        strategy_params=params,
        environment_config=env_config,
    )

    assert config.environment_config is not None
    assert config.environment_config.all_tools[0].id == "answer_faq"
    assert params.multiturn_attributes is not None
    mt_attr = params.multiturn_attributes[0]
    assert [t.id for t in config.resolve_multiturn_tools(mt_attr)] == ["answer_faq"]


def test_synthesis_config_requires_environment_config_for_available_tools():
    params = GeneralSynthesisParams(
        multiturn_attributes=[
            MultiTurnAttribute(
                id="chat",
                min_turns=1,
                max_turns=2,
                role_instruction_messages={
                    Role.USER: "You are a user.",
                    Role.ASSISTANT: "You are an assistant.",
                },
                available_tools=["answer_faq"],
            )
        ]
    )

    with pytest.raises(ValueError, match="Environment or tool references require"):
        SynthesisConfig(strategy_params=params)


def test_synthesis_config_validates_available_environments():
    env_config = EnvironmentConfig(environments=[_synthetic_env_params()])
    params = GeneralSynthesisParams(
        multiturn_attributes=[
            MultiTurnAttribute(
                id="chat",
                min_turns=1,
                max_turns=2,
                role_instruction_messages={
                    Role.USER: "You are a user.",
                    Role.ASSISTANT: "You are an assistant.",
                },
                available_environments=["missing_env"],
            )
        ]
    )

    with pytest.raises(ValueError, match="references unknown environment"):
        SynthesisConfig(strategy_params=params, environment_config=env_config)


def test_synthesis_config_restricts_tools_to_selected_environments():
    files_tool = ToolParams(
        id="read_file",
        name="ReadFile",
        description="Read a file.",
    )
    env_config = EnvironmentConfig(
        environments=[
            _synthetic_env_params(),
            _synthetic_env_params(
                env_id="files",
                name="Files",
                description="File tools",
                system_prompt="Manage files.",
                tools=[files_tool],
                extra_kwargs={
                    "state_params": {},
                    "cache_by_input": False,
                },
            ),
        ]
    )
    params = GeneralSynthesisParams(
        multiturn_attributes=[
            MultiTurnAttribute(
                id="chat",
                min_turns=1,
                max_turns=2,
                role_instruction_messages={
                    Role.USER: "You are a user.",
                    Role.ASSISTANT: "You are an assistant.",
                },
                available_environments=["faq"],
                available_tools=["read_file"],
            )
        ]
    )

    with pytest.raises(ValueError, match="references unknown tool"):
        SynthesisConfig(strategy_params=params, environment_config=env_config)


def test_synthesis_config_resolves_all_tools_from_selected_environments():
    files_tool = ToolParams(
        id="read_file",
        name="ReadFile",
        description="Read a file.",
    )
    env_config = EnvironmentConfig(
        environments=[
            _synthetic_env_params(),
            _synthetic_env_params(
                env_id="files",
                name="Files",
                description="File tools",
                system_prompt="Manage files.",
                tools=[files_tool],
                extra_kwargs={
                    "state_params": {},
                    "cache_by_input": False,
                },
            ),
        ]
    )
    mt_attr = MultiTurnAttribute(
        id="chat",
        min_turns=1,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["faq", "files"],
    )
    config = SynthesisConfig(
        strategy_params=GeneralSynthesisParams(multiturn_attributes=[mt_attr]),
        environment_config=env_config,
    )

    assert [env.id for env in config.resolve_multiturn_environments(mt_attr)] == [
        "faq",
        "files",
    ]
    assert [tool.id for tool in config.resolve_multiturn_tools(mt_attr)] == [
        "answer_faq",
        "read_file",
    ]
