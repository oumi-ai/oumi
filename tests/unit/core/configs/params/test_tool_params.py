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
    BaseEnvironment,
    DeterministicEnvironment,
    DeterministicToolOutput,
    SyntheticEnvironment,
    SyntheticStateParams,
    Tool,
    ToolResult,
)


def _make_deterministic_tool(**overrides: Any) -> Tool:
    defaults: dict[str, Any] = dict(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    defaults.update(overrides)
    return Tool(**defaults)


def _make_synthetic_tool(**overrides: Any) -> Tool:
    defaults: dict[str, Any] = dict(
        id="tool2",
        name="GenTool",
        description="A generated tool",
    )
    defaults.update(overrides)
    return Tool(**defaults)


def _make_state_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        },
        "required": ["files"],
    }


def test_deterministic_tool_output_allows_empty_input():
    entry = DeterministicToolOutput(input={}, output={"msg": "ok"})
    assert entry.input == {}


def test_deterministic_tool_output_allows_empty_output():
    entry = DeterministicToolOutput(input={"id": "1"}, output={})
    assert entry.output == {}


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


@pytest.mark.parametrize("field,value", [("id", ""), ("name", ""), ("description", "")])
def test_tool_empty_field_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        Tool(**{"id": "t", "name": "T", "description": "d", **{field: value}})


def test_tool_to_llm_schema():
    tool = Tool(
        id="search",
        name="Search",
        description="Search the catalog.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
    assert tool.to_llm_schema() == {
        "name": "Search",
        "description": "Search the catalog.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    }


def test_tool_create_coerces_deterministic_outputs():
    env = BaseEnvironment.create(
        {
            "id": "lookup",
            "type": "deterministic",
            "name": "Lookup",
            "description": "Lookup tools",
            "tools": [
                {
                    "id": "policy",
                    "name": "Policy",
                    "description": "Look up policy.",
                    "deterministic_outputs": [
                        {"input": {"id": "1"}, "output": {"result": "ok"}}
                    ],
                }
            ],
        }
    )
    assert isinstance(env, DeterministicEnvironment)
    assert isinstance(env.tools[0].deterministic_outputs[0], DeterministicToolOutput)


def test_synthetic_state_params_validates_initial_state_against_schema():
    with pytest.raises(ValueError, match=r"\$\.files\.count must be an integer"):
        SyntheticStateParams(
            state_schema=_make_state_schema(),
            initial_state={"files": {"count": "bad"}},
        )


def test_synthetic_state_params_accepts_partial_inputs():
    assert SyntheticStateParams(state_schema=_make_state_schema()).state_schema is not None
    assert (
        SyntheticStateParams(initial_state={"files": {"count": 1}}).initial_state
        == {"files": {"count": 1}}
    )


def test_synthetic_environment_valid_stateless():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[Tool(id="answer", name="Answer", description="Answer a FAQ.")],
    )
    assert env.type == "synthetic"
    assert env.state_params is None
    assert env.current_state is None
    assert isinstance(env.tools[0], Tool)


def test_synthetic_environment_valid_stateful():
    env = SyntheticEnvironment(
        id="filesystem",
        name="Filesystem",
        description="A simple filesystem",
        system_prompt="You manage a filesystem.",
        state_params=SyntheticStateParams(
            state_schema=_make_state_schema(),
            initial_state={"files": {"count": 1}},
        ),
        cache_by_input=False,
        tools=[Tool(id="read", name="Read", description="Read files.")],
    )
    assert env.current_state == {"files": {"count": 1}}


def test_synthetic_environment_coerces_dict_tools():
    env = SyntheticEnvironment(
        id="fs",
        name="FS",
        description="d",
        system_prompt="p",
        tools=[{"id": "read", "name": "Read", "description": "Read files."}],  # type: ignore[list-item]
    )
    assert isinstance(env.tools[0], Tool)


def test_synthetic_environment_empty_system_prompt_raises():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        SyntheticEnvironment(id="x", name="n", description="d", system_prompt="")


def test_synthetic_environment_rejects_deterministic_outputs():
    with pytest.raises(ValueError, match="cannot define deterministic_outputs"):
        SyntheticEnvironment(
            id="x",
            name="n",
            description="d",
            system_prompt="p",
            tools=[_make_deterministic_tool()],
        )


def test_synthetic_environment_rejects_cache_when_stateful():
    with pytest.raises(ValueError, match="cache_by_input must be False"):
        SyntheticEnvironment(
            id="x",
            name="n",
            description="d",
            system_prompt="p",
            state_params=SyntheticStateParams(),
            cache_by_input=True,
        )


def test_synthetic_environment_cache_round_trip():
    env = SyntheticEnvironment(
        id="weather",
        name="Weather",
        description="Weather API",
        system_prompt="Simulate weather.",
        cache_by_input=True,
        tools=[_make_synthetic_tool(id="get_weather")],
    )
    result = ToolResult(output={"temp": 72})
    env._cache_result("get_weather", {"city": "SF"}, result)
    cached = env._resolve_cached("get_weather", {"city": "SF"})
    assert cached == result
    assert cached is not result


def test_synthetic_environment_step_unknown_tool_raises():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step("missing", {})


def test_synthetic_environment_step_known_tool_is_stub():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        env.step("answer", {})


def test_deterministic_environment_valid():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            Tool(
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
    assert env.type == "deterministic"
    assert isinstance(env.tools[0], Tool)


def test_deterministic_environment_requires_outputs_on_tool():
    with pytest.raises(ValueError, match="must have at least one"):
        DeterministicEnvironment(
            id="det_env",
            name="Deterministic",
            description="d",
            tools=[_make_deterministic_tool(deterministic_outputs=[])],
        )


def test_deterministic_environment_duplicate_inputs_raises():
    outputs = [
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "a"}),
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "b"}),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        DeterministicEnvironment(
            id="det_env",
            name="Deterministic",
            description="d",
            tools=[_make_deterministic_tool(deterministic_outputs=outputs)],
        )


def test_deterministic_environment_step_match():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            _make_deterministic_tool(
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "01"}, output={"msg": "pending"}
                    ),
                    DeterministicToolOutput(
                        input={"id": "02"}, output={"msg": "delivered"}
                    ),
                ]
            )
        ],
    )
    assert env.step("tool1", {"id": "01"}) == ToolResult(output={"msg": "pending"})
    assert env.step("tool1", {"id": "02"}) == ToolResult(output={"msg": "delivered"})


def test_deterministic_environment_step_no_match():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[_make_deterministic_tool()],
    )
    assert env.step("tool1", {"id": "99"}) == ToolResult(output=None)


def test_deterministic_environment_supports_empty_argument_match():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            Tool(
                id="ping",
                name="Ping",
                description="Zero-arg tool.",
                deterministic_outputs=[
                    DeterministicToolOutput(input={}, output={}),
                ],
            )
        ],
    )
    assert env.step("ping", {}) == ToolResult(output={})


def test_environment_empty_id_raises():
    with pytest.raises(ValueError, match="id cannot be empty"):
        SyntheticEnvironment(id="", name="n", description="d", system_prompt="p")


def test_environment_empty_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        SyntheticEnvironment(id="x", name="", description="d", system_prompt="p")


def test_environment_empty_description_raises():
    with pytest.raises(ValueError, match="description cannot be empty"):
        SyntheticEnvironment(id="x", name="n", description="", system_prompt="p")


def test_environment_duplicate_tool_ids_raises():
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        SyntheticEnvironment(
            id="env2",
            name="Env 2",
            description="d",
            system_prompt="p",
            tools=[
                Tool(id="dup", name="Read", description="Read files."),
                Tool(id="dup", name="Write", description="Write files."),
            ],
        )


def test_environment_config_duplicate_tool_ids_across_envs_raises():
    env1 = SyntheticEnvironment(
        id="env1",
        name="Env 1",
        description="d",
        system_prompt="p",
        tools=[Tool(id="dup", name="Read", description="Read files.")],
    )
    env2 = SyntheticEnvironment(
        id="env2",
        name="Env 2",
        description="d",
        system_prompt="p",
        tools=[Tool(id="dup", name="Write", description="Write files.")],
    )
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        EnvironmentConfig(environments=[env1, env2])


def test_environment_config_tool_environment_map():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer_faq")],
    )
    config = EnvironmentConfig(environments=[env])
    assert config.tool_environment_map == {"answer_faq": "faq"}


def test_base_environment_create_routes_synthetic():
    env = BaseEnvironment.create(
        {
            "id": "faq",
            "type": "synthetic",
            "name": "FAQ",
            "description": "FAQ tools",
            "system_prompt": "Answer FAQs.",
            "tools": [{"id": "answer", "name": "Answer", "description": "Answer."}],
        }
    )
    assert isinstance(env, SyntheticEnvironment)


def test_base_environment_create_routes_deterministic():
    env = BaseEnvironment.create(
        {
            "id": "lookup",
            "type": "deterministic",
            "name": "Lookup",
            "description": "Lookup tools",
            "tools": [
                {
                    "id": "policy",
                    "name": "Policy",
                    "description": "Look up policy.",
                    "deterministic_outputs": [
                        {"input": {"id": "1"}, "output": {"result": "ok"}}
                    ],
                }
            ],
        }
    )
    assert isinstance(env, DeterministicEnvironment)


def test_base_environment_create_missing_type_raises():
    with pytest.raises(ValueError, match="must include a 'type' field"):
        BaseEnvironment.create({"id": "faq"})


def test_base_environment_create_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported environment type"):
        BaseEnvironment.create({"id": "faq", "type": "unknown"})
