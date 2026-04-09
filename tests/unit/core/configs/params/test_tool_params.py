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

from typing import Any

import pytest

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.environments import (
    BaseTool,
    DeterministicEnvironment,
    DeterministicTool,
    DeterministicToolOutput,
    GeneratedToolOutput,
    StatefulEnvironment,
    StatefulTool,
    StatelessEnvironment,
    StatelessTool,
    ToolEnvironmentType,
)


def _make_deterministic_tool(**overrides: Any) -> DeterministicTool:
    defaults: dict[str, Any] = dict(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    defaults.update(overrides)
    return DeterministicTool(**defaults)


def _make_stateless_tool(**overrides: Any) -> StatelessTool:
    defaults: dict[str, Any] = dict(
        id="tool2",
        name="GenTool",
        description="A generated tool",
        generated_output=GeneratedToolOutput(instruction="Do something."),
    )
    defaults.update(overrides)
    return StatelessTool(**defaults)


def _make_stateful_tool(**overrides: Any) -> StatefulTool:
    defaults: dict[str, Any] = dict(
        id="tool3",
        name="StatefulTool",
        description="A stateful tool",
    )
    defaults.update(overrides)
    return StatefulTool(**defaults)


# --- DeterministicToolOutput tests ---


def test_deterministic_tool_output_empty_input_raises():
    with pytest.raises(ValueError, match="input cannot be empty"):
        DeterministicToolOutput(input={}, output={"msg": "ok"})


def test_deterministic_tool_output_empty_output_raises():
    with pytest.raises(ValueError, match="output cannot be empty"):
        DeterministicToolOutput(input={"id": "1"}, output={})


def test_deterministic_tool_output_matches_exact():
    entry = DeterministicToolOutput(
        input={"id": "01", "status": "pending"},
        output={"message": "Order is pending"},
    )
    assert entry.matches({"id": "01", "status": "pending"}) is True
    assert entry.matches({"status": "pending", "id": "01"}) is True


def test_deterministic_tool_output_no_match():
    entry = DeterministicToolOutput(
        input={"id": "01"},
        output={"message": "ok"},
    )
    assert entry.matches({"id": "02"}) is False
    assert entry.matches({"id": "01", "extra": "arg"}) is False


# --- BaseTool tests ---


@pytest.mark.parametrize("field,value", [("id", ""), ("name", ""), ("description", "")])
def test_base_tool_empty_field_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        BaseTool(**{"id": "t", "name": "T", "description": "d", **{field: value}})


# --- DeterministicTool tests ---


def test_deterministic_tool_requires_outputs():
    with pytest.raises(ValueError, match="must have at least one"):
        DeterministicTool(id="t", name="T", description="d", deterministic_outputs=[])


def test_deterministic_tool_duplicate_inputs_raises():
    outputs = [
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "a"}),
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "b"}),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        _make_deterministic_tool(deterministic_outputs=outputs)


def test_deterministic_tool_resolve_match():
    tool = _make_deterministic_tool(
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "pending"}),
            DeterministicToolOutput(input={"id": "02"}, output={"msg": "delivered"}),
        ]
    )
    assert tool.resolve_deterministic({"id": "01"}) == {"msg": "pending"}
    assert tool.resolve_deterministic({"id": "02"}) == {"msg": "delivered"}


def test_deterministic_tool_resolve_no_match():
    tool = _make_deterministic_tool()
    assert tool.resolve_deterministic({"id": "99"}) is None


# --- StatelessTool tests ---


def test_stateless_tool_requires_generated_output():
    with pytest.raises(ValueError, match="must have a generated_output"):
        StatelessTool(id="t", name="T", description="d", generated_output=None)


# --- ToolEnvironmentType tests ---


def test_tool_environment_type_values_exist():
    assert ToolEnvironmentType.STATEFUL == "stateful"
    assert ToolEnvironmentType.STATELESS == "stateless"
    assert ToolEnvironmentType.DETERMINISTIC == "deterministic"


# --- Environment + typed tool integration tests ---


def test_stateful_environment_valid():
    env = StatefulEnvironment(
        id="filesystem",
        name="Filesystem",
        description="A simple filesystem",
        system_prompt="You manage a filesystem.",
        tools=[StatefulTool(id="read", name="Read", description="Read files.")],
    )
    assert env.id == "filesystem"
    assert env.state_schema is None
    assert env.initial_state is None
    assert isinstance(env.tools[0], StatefulTool)


def test_stateful_environment_coerces_dict_tools():
    env = StatefulEnvironment(
        id="fs",
        name="FS",
        description="d",
        system_prompt="p",
        tools=[{"id": "read", "name": "Read", "description": "Read files."}],  # type: ignore[arg-type]
    )
    assert isinstance(env.tools[0], StatefulTool)


def test_deterministic_environment_valid():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            DeterministicTool(
                id="policy",
                name="Policy",
                description="Look up policy.",
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "1"},
                        output={"result": "ok"},
                    )
                ],
            )
        ],
    )
    assert not hasattr(env, "system_prompt")
    assert isinstance(env.tools[0], DeterministicTool)


def test_environment_with_schema_and_state():
    schema = {
        "type": "object",
        "properties": {"files": {"type": "object"}},
        "required": ["files"],
    }
    state = {"files": {}}
    env = StatefulEnvironment(
        id="fs",
        name="FS",
        description="d",
        system_prompt="p",
        state_schema=schema,
        initial_state=state,
    )
    assert env.state_schema == schema
    assert env.initial_state == state


def test_environment_empty_id_raises():
    with pytest.raises(ValueError, match="id cannot be empty"):
        StatefulEnvironment(id="", name="n", description="d", system_prompt="p")


def test_environment_empty_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        StatefulEnvironment(id="x", name="", description="d", system_prompt="p")


def test_environment_empty_description_raises():
    with pytest.raises(ValueError, match="description cannot be empty"):
        StatefulEnvironment(id="x", name="n", description="", system_prompt="p")


def test_stateful_environment_empty_system_prompt_raises():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        StatefulEnvironment(id="x", name="n", description="d", system_prompt="")


def test_stateless_environment_empty_system_prompt_raises():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        StatelessEnvironment(id="x", name="n", description="d", system_prompt="")


def test_stateless_environment_with_state_schema_raises():
    with pytest.raises(TypeError, match="unexpected keyword argument 'state_schema'"):
        StatelessEnvironment(
            id="x",
            name="n",
            description="d",
            system_prompt="p",
            state_schema={"type": "object"},  # type: ignore[call-arg]
        )


def test_deterministic_environment_with_initial_state_raises():
    with pytest.raises(TypeError, match="unexpected keyword argument 'initial_state'"):
        DeterministicEnvironment(id="x", name="n", description="d", initial_state={})  # type: ignore[call-arg]


def test_environment_duplicate_tool_ids_raises():
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        StatefulEnvironment(
            id="env2",
            name="Env 2",
            description="d",
            system_prompt="p",
            tools=[
                StatefulTool(id="dup", name="Read", description="Read files."),
                StatefulTool(id="dup", name="Write", description="Write files."),
            ],
        )


def test_environment_config_duplicate_tool_ids_across_envs_raises():
    env1 = StatefulEnvironment(
        id="env1",
        name="Env 1",
        description="d",
        system_prompt="p",
        tools=[StatefulTool(id="dup", name="Read", description="Read files.")],
    )
    env2 = StatefulEnvironment(
        id="env2",
        name="Env 2",
        description="d",
        system_prompt="p",
        tools=[StatefulTool(id="dup", name="Write", description="Write files.")],
    )
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        EnvironmentConfig(environments=[env1, env2])


def test_environment_config_tool_environment_map():
    env = StatelessEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_stateless_tool(id="answer_faq")],
    )
    config = EnvironmentConfig(environments=[env])
    assert config.tool_environment_map == {"answer_faq": "faq"}


def test_deterministic_environment_requires_outputs_on_tool():
    with pytest.raises(ValueError, match="must have at least one"):
        DeterministicEnvironment(
            id="det_env",
            name="Deterministic",
            description="d",
            tools=[_make_deterministic_tool(deterministic_outputs=[])],
        )
