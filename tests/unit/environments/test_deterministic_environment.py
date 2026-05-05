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

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.deterministic_environment import (
    DeterministicEnvironment,
    DeterministicEnvironmentKwargs,
)
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)


def _make_tool(**overrides) -> DeterministicTool:
    defaults: dict = dict(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    defaults.update(overrides)
    return DeterministicTool(**defaults)


def _make_params(**overrides) -> EnvironmentParams:
    defaults: dict = dict(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        env_type="deterministic",
        tools=[_make_tool()],
    )
    defaults.update(overrides)
    return EnvironmentParams(**defaults)


def test_from_params_constructs_runtime_instance():
    env = DeterministicEnvironment.from_params(_make_params())
    assert isinstance(env, DeterministicEnvironment)
    assert isinstance(env._kwargs, DeterministicEnvironmentKwargs)


def test_rejects_env_kwargs():
    params = _make_params(env_kwargs={"unexpected": True})
    with pytest.raises(ValueError, match="does not accept env_kwargs"):
        DeterministicEnvironment.from_params(params)


def test_requires_deterministic_outputs_on_tool():
    params = _make_params(
        tools=[_make_tool(deterministic_outputs=[])],
    )
    with pytest.raises(ValueError, match="must have at least one"):
        DeterministicEnvironment.from_params(params)


def test_duplicate_deterministic_inputs_raises():
    outputs = [
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "a"}),
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "b"}),
    ]
    params = _make_params(tools=[_make_tool(deterministic_outputs=outputs)])
    with pytest.raises(ValueError, match="duplicate"):
        DeterministicEnvironment.from_params(params)


def test_step_returns_matching_output():
    params = _make_params(
        tools=[
            _make_tool(
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "01"}, output={"msg": "pending"}
                    ),
                    DeterministicToolOutput(
                        input={"id": "02"}, output={"msg": "delivered"}
                    ),
                ]
            )
        ]
    )
    env = DeterministicEnvironment.from_params(params)
    assert env.step("tool1", {"id": "01"}) == ToolResult(output={"msg": "pending"})
    assert env.step("tool1", {"id": "02"}) == ToolResult(output={"msg": "delivered"})


def test_step_no_match_returns_empty():
    env = DeterministicEnvironment.from_params(_make_params())
    assert env.step("tool1", {"id": "99"}) == ToolResult(output={})


def test_step_supports_zero_arg_tool():
    params = _make_params(
        tools=[
            DeterministicTool(
                id="ping",
                name="Ping",
                description="Zero-arg tool.",
                deterministic_outputs=[
                    DeterministicToolOutput(input={}, output={}),
                ],
            )
        ]
    )
    env = DeterministicEnvironment.from_params(params)
    assert env.step("ping", {}) == ToolResult(output={})


def test_step_unknown_tool_raises():
    env = DeterministicEnvironment.from_params(_make_params())
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step("missing", {"id": "01"})


def test_from_params_coerces_raw_deterministic_outputs():
    tool = DeterministicTool(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[{"input": {"id": "1"}, "output": {"msg": "ok"}}],  # type: ignore[list-item]
    )
    env = DeterministicEnvironment.from_params(_make_params(tools=[tool]))
    assert isinstance(
        env._params.tools[0].deterministic_outputs[0], DeterministicToolOutput
    )
