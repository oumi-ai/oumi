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
from unittest.mock import Mock

import pytest

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.deterministic_environment import DeterministicEnvironment


def _tool(tool_id: str, schema: dict[str, Any] | None = None) -> ToolParams:
    return ToolParams(
        id=tool_id,
        name=tool_id,
        description=f"Tool {tool_id}",
        parameters=schema or {"type": "object"},
    )


def _det_env_params(
    env_id: str,
    tools: list[ToolParams],
    lookup: dict[str, list[dict]] | None = None,
) -> EnvironmentParams:
    """Build a DeterministicEnvironment params block with a default identity lookup."""
    return EnvironmentParams(
        id=env_id,
        name=env_id,
        description=f"Env {env_id}",
        env_type="deterministic",
        tools=tools,
        env_kwargs={
            "lookup_table": lookup
            or {t.id: [{"input": {}, "output": {}}] for t in tools}
        },
    )


# ---------- from_environment_config ----------


def test_from_environment_config_builds_all_fields():
    tool = _tool("t1")
    env_config = EnvironmentConfig(environments=[_det_env_params("env1", [tool])])

    router = ToolRouter.from_environment_config(env_config)

    assert set(router.env_by_id) == {"env1"}
    assert isinstance(router.env_by_id["env1"], DeterministicEnvironment)
    assert router.tool_to_env["t1"] is router.env_by_id["env1"]
    assert router.tools_by_id["t1"] is tool
    assert len(router.tool_specs) == 1
    assert router.tool_specs[0].function.name == "t1"


def test_from_environment_config_invokes_on_env_built_for_each_env():
    env_config = EnvironmentConfig(
        environments=[
            _det_env_params("env1", [_tool("t1")]),
            _det_env_params("env2", [_tool("t2")]),
        ]
    )
    seen: list[BaseEnvironment] = []

    ToolRouter.from_environment_config(env_config, on_env_built=seen.append)

    assert len(seen) == 2
    assert all(isinstance(env, DeterministicEnvironment) for env in seen)


def test_from_environment_config_routes_tools_across_envs():
    """Two envs, multiple tools each; every tool resolves to its owning env."""
    t1, t2, t3 = _tool("t1"), _tool("t2"), _tool("t3")
    env_config = EnvironmentConfig(
        environments=[
            _det_env_params("env1", [t1, t2]),
            _det_env_params("env2", [t3]),
        ]
    )

    router = ToolRouter.from_environment_config(env_config)

    assert router.tool_to_env["t1"] is router.env_by_id["env1"]
    assert router.tool_to_env["t2"] is router.env_by_id["env1"]
    assert router.tool_to_env["t3"] is router.env_by_id["env2"]
    assert {spec.function.name for spec in router.tool_specs} == {"t1", "t2", "t3"}


# ---------- parse_and_validate_arguments ----------


def _typed_tool(tool_id: str) -> ToolParams:
    return _tool(
        tool_id,
        schema={
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        },
    )


def _router_with(*tools: ToolParams) -> ToolRouter:
    env_config = EnvironmentConfig(environments=[_det_env_params("env1", list(tools))])
    return ToolRouter.from_environment_config(env_config)


def test_parse_and_validate_arguments_happy_path():
    router = _router_with(_typed_tool("t1"))
    assert router.parse_and_validate_arguments("t1", '{"q": "hello"}') == {"q": "hello"}


def test_parse_and_validate_arguments_unknown_tool_raises_lookup_error():
    router = _router_with(_typed_tool("t1"))
    with pytest.raises(ToolLookupError, match="Unknown tool 'ghost'"):
        router.parse_and_validate_arguments("ghost", '{"q": "x"}')


def test_parse_and_validate_arguments_malformed_json_raises_argument_error():
    router = _router_with(_typed_tool("t1"))
    with pytest.raises(ToolArgumentError, match="not valid JSON"):
        router.parse_and_validate_arguments("t1", "{not json")


def test_parse_and_validate_arguments_non_dict_raises_argument_error():
    router = _router_with(_typed_tool("t1"))
    with pytest.raises(ToolArgumentError, match="must be a JSON object"):
        router.parse_and_validate_arguments("t1", "[1, 2, 3]")


def test_parse_and_validate_arguments_schema_violation_raises_argument_error():
    """`tool.validate_arguments` rejects payloads missing required fields."""
    router = _router_with(_typed_tool("t1"))
    with pytest.raises(ToolArgumentError):
        router.parse_and_validate_arguments("t1", '{"unrelated": "x"}')


def test_parse_and_validate_arguments_empty_string_defaults_to_empty_dict():
    """Empty raw_arguments → '{}' so tools with no required args pass through."""
    router = _router_with(_tool("t1", schema={"type": "object"}))
    assert router.parse_and_validate_arguments("t1", "") == {}


# ---------- route_batch ----------


def _mock_router(tool_to_env: dict[str, BaseEnvironment]) -> ToolRouter:
    """Build a ToolRouter directly with mocked envs (skip from_environment_config)."""
    env_by_id = {f"env_{i}": env for i, env in enumerate(set(tool_to_env.values()))}
    return ToolRouter(
        tool_specs=[],
        tools_by_id={},
        env_by_id=env_by_id,
        tool_to_env=tool_to_env,
    )


def test_route_batch_empty_returns_empty():
    router = _router_with(_tool("t1"))
    assert router.route_batch([]) == []


def test_route_batch_dispatches_through_real_env():
    """End-to-end: real DeterministicEnvironment lookup via the router."""
    env_params = _det_env_params(
        "env1",
        [_tool("t1")],
        lookup={"t1": [{"input": {"id": "01"}, "output": {"msg": "ok"}}]},
    )
    router = ToolRouter.from_environment_config(
        EnvironmentConfig(environments=[env_params])
    )

    [result] = router.route_batch([("t1", {"id": "01"})])
    assert result == ToolResult(output={"msg": "ok"})


def test_route_batch_groups_same_env_into_one_step():
    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.return_value = [
        ToolResult(output={"i": 0}),
        ToolResult(output={"i": 1}),
        ToolResult(output={"i": 2}),
    ]
    router = _mock_router({"t1": fake_env, "t2": fake_env})

    results = router.route_batch([("t1", {"i": 0}), ("t2", {"i": 1}), ("t1", {"i": 2})])

    assert [r.output for r in results] == [{"i": 0}, {"i": 1}, {"i": 2}]
    fake_env.step.assert_called_once_with(
        [("t1", {"i": 0}), ("t2", {"i": 1}), ("t1", {"i": 2})]
    )


def test_route_batch_splits_across_envs_preserves_order():
    """Two envs, interleaved calls: one step() per env, output order matches input."""
    env_a = Mock(spec=BaseEnvironment)
    env_b = Mock(spec=BaseEnvironment)
    env_a.step.side_effect = lambda calls: [
        ToolResult(output={"env": "a", **args}) for _, args in calls
    ]
    env_b.step.side_effect = lambda calls: [
        ToolResult(output={"env": "b", **args}) for _, args in calls
    ]
    router = _mock_router({"ta": env_a, "tb": env_b})

    results = router.route_batch(
        [
            ("ta", {"i": 0}),  # env_a
            ("tb", {"i": 1}),  # env_b
            ("ta", {"i": 2}),  # env_a
            ("tb", {"i": 3}),  # env_b
        ]
    )

    assert [r.output for r in results] == [
        {"env": "a", "i": 0},
        {"env": "b", "i": 1},
        {"env": "a", "i": 2},
        {"env": "b", "i": 3},
    ]
    assert env_a.step.call_count == 1
    assert env_b.step.call_count == 1


def test_route_batch_unknown_tool_raises_before_any_step():
    """Validation pass surfaces unknown tools before any env.step is invoked."""
    fake_env = Mock(spec=BaseEnvironment)
    router = _mock_router({"known": fake_env})
    with pytest.raises(ToolLookupError, match="Unknown tool 'ghost'"):
        router.route_batch([("known", {}), ("ghost", {})])
    fake_env.step.assert_not_called()


def test_route_batch_env_exception_propagates():
    """Fail-fast contract: env.step exceptions are not caught by the router."""
    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.side_effect = RuntimeError("boom")
    router = _mock_router({"t1": fake_env})
    with pytest.raises(RuntimeError, match="boom"):
        router.route_batch([("t1", {})])
