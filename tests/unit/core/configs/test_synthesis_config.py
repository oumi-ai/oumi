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

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.configs.synthesis_config import SynthesisConfig, SynthesisStrategy
from oumi.core.types.conversation import Role
from oumi.environments import (
    GeneratedToolOutput,
    StatefulEnvironment,
    StatefulTool,
    StatelessEnvironment,
    StatelessTool,
)


def test_default_synthesis_config():
    """Test default initialization of SynthesisConfig."""
    config = SynthesisConfig()

    assert config.strategy == SynthesisStrategy.GENERAL
    assert isinstance(config.strategy_params, GeneralSynthesisParams)
    assert isinstance(config.inference_config, InferenceConfig)
    assert config.num_samples == 1


def test_custom_synthesis_config():
    """Test custom initialization of SynthesisConfig."""
    custom_params = GeneralSynthesisParams()
    custom_inference = InferenceConfig()

    config = SynthesisConfig(
        strategy=SynthesisStrategy.GENERAL,
        strategy_params=custom_params,
        inference_config=custom_inference,
        num_samples=5,
    )

    assert config.strategy == SynthesisStrategy.GENERAL
    assert config.strategy_params == custom_params
    assert config.inference_config == custom_inference
    assert config.num_samples == 5


def test_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    config = SynthesisConfig()
    config.strategy = "invalid_strategy"  # type: ignore

    with pytest.raises(ValueError, match="Unsupported synthesis strategy"):
        config.__post_init__()


def test_invalid_input_path():
    """Test that setting input_path raises ValueError."""
    inference_config = InferenceConfig(input_path="some/path")

    with pytest.raises(ValueError, match="Input path is not supported"):
        SynthesisConfig(inference_config=inference_config)


def test_invalid_output_path():
    """Test that setting output_path raises ValueError."""
    inference_config = InferenceConfig(output_path="some/path")

    with pytest.raises(ValueError, match="Output path is not supported"):
        SynthesisConfig(inference_config=inference_config)


def _make_faq_tool() -> StatelessTool:
    return StatelessTool(
        id="answer_faq",
        name="AnswerFAQ",
        description="Answer a FAQ question.",
        generated_output=GeneratedToolOutput(
            instruction="Answer the given FAQ question."
        ),
    )


def test_synthesis_config_with_top_level_environment_config():
    env_config = EnvironmentConfig(
        environments=[
            StatelessEnvironment(
                id="faq",
                name="FAQ",
                description="FAQ tools",
                system_prompt="Answer FAQs.",
                tools=[_make_faq_tool()],
            )
        ]
    )
    params = GeneralSynthesisParams()
    params.multiturn_attributes = []

    config = SynthesisConfig(
        strategy_params=params,
        environment_config=env_config,
    )

    assert config.environment_config == env_config
    assert config.environment_config.tool_environment_map == {
        "answer_faq": "faq"
    }


def test_synthesis_config_loads_environment_config_from_path(tmp_path):
    env_config_path = tmp_path / "environments.yaml"
    env_config = EnvironmentConfig(
        environments=[
            StatelessEnvironment(
                id="faq",
                name="FAQ",
                description="FAQ tools",
                system_prompt="Answer FAQs.",
                tools=[_make_faq_tool()],
            )
        ]
    )
    env_config.to_yaml(env_config_path)

    config = SynthesisConfig(environment_config_path=str(env_config_path))

    assert config.environment_config is not None
    assert config.environment_config.all_tools[0].id == "answer_faq"


def test_synthesis_config_validates_available_tools():
    env_config = EnvironmentConfig(
        environments=[
            StatelessEnvironment(
                id="faq",
                name="FAQ",
                description="FAQ tools",
                system_prompt="Answer FAQs.",
                tools=[_make_faq_tool()],
            )
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
    mt_attr = params.multiturn_attributes[0]
    assert [t.id for t in config.resolve_multiturn_tools(mt_attr)] == [
        "answer_faq"
    ]


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

    with pytest.raises(
        ValueError, match="Environment or tool references require"
    ):
        SynthesisConfig(strategy_params=params)


def test_synthesis_config_validates_available_environments():
    env_config = EnvironmentConfig(
        environments=[
            StatelessEnvironment(
                id="faq",
                name="FAQ",
                description="FAQ tools",
                system_prompt="Answer FAQs.",
                tools=[_make_faq_tool()],
            )
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
                available_environments=["missing_env"],
            )
        ]
    )

    with pytest.raises(ValueError, match="references unknown environment"):
        SynthesisConfig(
            strategy_params=params, environment_config=env_config
        )


def test_synthesis_config_restricts_tools_to_selected_environments():
    env_config = EnvironmentConfig(
        environments=[
            StatelessEnvironment(
                id="faq",
                name="FAQ",
                description="FAQ tools",
                system_prompt="Answer FAQs.",
                tools=[_make_faq_tool()],
            ),
            StatefulEnvironment(
                id="files",
                name="Files",
                description="File tools",
                system_prompt="Manage files.",
                tools=[
                    StatefulTool(
                        id="read_file",
                        name="ReadFile",
                        description="Read a file.",
                    )
                ],
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
        SynthesisConfig(
            strategy_params=params, environment_config=env_config
        )


def test_synthesis_config_resolves_all_tools_from_selected_environments():
    env_config = EnvironmentConfig(
        environments=[
            StatelessEnvironment(
                id="faq",
                name="FAQ",
                description="FAQ tools",
                system_prompt="Answer FAQs.",
                tools=[_make_faq_tool()],
            ),
            StatefulEnvironment(
                id="files",
                name="Files",
                description="File tools",
                system_prompt="Manage files.",
                tools=[
                    StatefulTool(
                        id="read_file",
                        name="ReadFile",
                        description="Read a file.",
                    )
                ],
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
        strategy_params=GeneralSynthesisParams(
            multiturn_attributes=[mt_attr]
        ),
        environment_config=env_config,
    )

    assert [
        env.id for env in config.resolve_multiturn_environments(mt_attr)
    ] == ["faq", "files"]
    assert [
        tool.id for tool in config.resolve_multiturn_tools(mt_attr)
    ] == ["answer_faq", "read_file"]
