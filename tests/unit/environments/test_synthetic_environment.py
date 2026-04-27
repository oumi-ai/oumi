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
from oumi.core.configs.params.tool_params import (
    DeterministicToolOutput,
    ToolParams,
    ToolResult,
)
from oumi.environments.synthetic_environment import (
    SyntheticEnvironment,
    SyntheticEnvironmentKwargs,
    SyntheticStateParams,
)


def _make_state_schema() -> dict:
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


def _make_tool(**overrides) -> ToolParams:
    defaults: dict = dict(id="answer", name="Answer", description="Answer.")
    defaults.update(overrides)
    return ToolParams(**defaults)


def _make_params(**overrides) -> EnvironmentParams:
    defaults: dict = dict(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        env_type="synthetic",
        tools=[_make_tool()],
        env_kwargs={"system_prompt": "Answer FAQs."},
    )
    defaults.update(overrides)
    return EnvironmentParams(**defaults)


def test_from_params_constructs_stateless():
    env = SyntheticEnvironment.from_params(_make_params())
    assert isinstance(env, SyntheticEnvironment)
    assert env.current_state is None
    assert isinstance(env._kwargs, SyntheticEnvironmentKwargs)


def test_from_params_constructs_stateful():
    params = _make_params(
        env_kwargs={
            "system_prompt": "You manage a filesystem.",
            "state_params": SyntheticStateParams(
                state_schema=_make_state_schema(),
                initial_state={"files": {"count": 1}},
            ),
            "cache_by_input": False,
        }
    )
    env = SyntheticEnvironment.from_params(params)
    assert env.current_state == {"files": {"count": 1}}


def test_empty_system_prompt_raises():
    params = _make_params(env_kwargs={"system_prompt": ""})
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        SyntheticEnvironment.from_params(params)


def test_rejects_deterministic_outputs_on_tool():
    bad_tool = _make_tool(
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    params = _make_params(tools=[bad_tool])
    with pytest.raises(ValueError, match="cannot define deterministic_outputs"):
        SyntheticEnvironment.from_params(params)


def test_rejects_cache_when_stateful():
    params = _make_params(
        env_kwargs={
            "system_prompt": "p",
            "state_params": SyntheticStateParams(),
            "cache_by_input": True,
        }
    )
    with pytest.raises(ValueError, match="cache_by_input must be False"):
        SyntheticEnvironment.from_params(params)


def test_cache_round_trip_stateless_caching():
    env = SyntheticEnvironment.from_params(_make_params())
    result = ToolResult(output={"temp": 72})
    env._cache_result("answer", {"city": "SF"}, result)
    cached = env._resolve_cached("answer", {"city": "SF"})
    assert cached == result
    assert cached is not result


def test_step_unknown_tool_raises():
    env = SyntheticEnvironment.from_params(_make_params())
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step("missing", {})


def test_step_known_tool_is_stub():
    env = SyntheticEnvironment.from_params(_make_params())
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        env.step("answer", {})
