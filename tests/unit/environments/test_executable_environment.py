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

"""Skeleton-shape tests for ExecutableEnvironment."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolError,
    ToolLookupError,
)
from oumi.core.types.tool_call import ToolResult
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool


def _echo_executor(arguments, context):
    return ToolResult(output={"args": arguments, "context": context})


def _echo_n(arguments, context):
    return ToolResult(output={"n": arguments["n"]})


def _bad_return_executor(arguments, context):
    return {"not": "a ToolResult"}


def _raising_executor(arguments, context):
    raise RuntimeError("boom")


class _EchoExecEnv(ExecutableEnvironment):
    """Concrete env whose executor echoes the context it was handed."""

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        yield {"ctx_for": tool.id}


def _tool(tool_id: str, executor: str = f"{__name__}._echo_executor", **overrides):
    return ExecutableTool(
        id=tool_id, name=tool_id, description="d", executor=executor, **overrides
    )


def _env(tools: list[ExecutableTool]) -> _EchoExecEnv:
    return _EchoExecEnv(
        EnvironmentParams(
            id="echo", name="echo", description="d", env_type="executable", tools=tools
        )
    )


def test_cannot_instantiate_abstract_base():
    """ExecutableEnvironment is abstract — _build_execution_context must be supplied."""
    with pytest.raises(TypeError, match="abstract"):
        ExecutableEnvironment(  # type: ignore[abstract]
            EnvironmentParams(id="t", name="t", description="d", env_type="executable")
        )


def test_default_tool_params_cls_is_executable_tool():
    assert ExecutableEnvironment.tool_params_cls is ExecutableTool


def test_close_is_noop():
    result = _env([]).close()
    assert result is None


def test_absorb_result_is_noop():
    result = _env([])._absorb_result(_tool("t"), ToolResult(output={"ok": True}))
    assert result is None


def test_step_dispatches_to_executor_with_context():
    [result] = _env([_tool("t")]).step([("t", {"a": 1})])
    assert result.output == {"args": {"a": 1}, "context": {"ctx_for": "t"}}


def test_step_preserves_batch_order():
    results = _env([_tool("a"), _tool("b")]).step([("b", {}), ("a", {})])
    assert [r.output for r in results] == [
        {"args": {}, "context": {"ctx_for": "b"}},
        {"args": {}, "context": {"ctx_for": "a"}},
    ]


def test_unknown_tool_raises_lookup_error():
    with pytest.raises(ToolLookupError):
        _env([]).step([("missing", {})])


def test_unresolvable_executor_raises_at_construction():
    with pytest.raises(ValueError, match="executor"):
        _env([_tool("t", executor="oumi.does.not.exist_fn")])


def test_bad_arguments_rejected_before_dispatch():
    tool = _tool(
        "t",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    )
    with pytest.raises(ToolArgumentError):
        _env([tool]).step([("t", {})])


def test_rejects_non_toolresult_executor_return():
    with pytest.raises(ToolError):
        _env([_tool("bad", executor=f"{__name__}._bad_return_executor")]).step(
            [("bad", {})]
        )


def test_output_schema_conforming_output_passes():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    [result] = _env(
        [_tool("t", executor=f"{__name__}._echo_n", output_schema=schema)]
    ).step([("t", {"n": 3})])
    assert result.output == {"n": 3}


def test_output_schema_violation_raises():
    schema = {"type": "object", "properties": {"n": {"type": "string"}}}
    with pytest.raises(ToolError, match="schema"):
        _env([_tool("t", executor=f"{__name__}._echo_n", output_schema=schema)]).step(
            [("t", {"n": 3})]
        )


class _TrackingExecEnv(ExecutableEnvironment):
    """Records context-manager enter/exit ordering."""

    def __init__(self, params: EnvironmentParams) -> None:
        super().__init__(params)
        self.events: list[str] = []

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        self.events.append("enter")
        try:
            yield None
        finally:
            self.events.append("exit")


def _tracking_env(tools: list[ExecutableTool]) -> _TrackingExecEnv:
    return _TrackingExecEnv(
        EnvironmentParams(
            id="t", name="t", description="d", env_type="executable", tools=tools
        )
    )


def test_context_manager_teardown_runs_after_executor():
    env = _tracking_env([_tool("t")])
    env.step([("t", {})])
    assert env.events == ["enter", "exit"]


def test_context_manager_teardown_runs_when_result_invalid():
    env = _tracking_env([_tool("bad", executor=f"{__name__}._bad_return_executor")])
    with pytest.raises(ToolError):
        env.step([("bad", {})])
    assert env.events == ["enter", "exit"]


def test_context_manager_teardown_runs_when_executor_raises():
    env = _tracking_env([_tool("bad", executor=f"{__name__}._raising_executor")])
    with pytest.raises(RuntimeError, match="boom"):
        env.step([("bad", {})])
    assert env.events == ["enter", "exit"]


class _AbsorbingExecEnv(_EchoExecEnv):
    """Captures every result handed to the _absorb_result post-hook."""

    def __init__(self, params: EnvironmentParams) -> None:
        super().__init__(params)
        self.absorbed: list[ToolResult] = []

    def _absorb_result(self, tool: ExecutableTool, result: ToolResult) -> None:
        self.absorbed.append(result)


def test_absorb_result_receives_the_validated_result():
    env = _AbsorbingExecEnv(
        EnvironmentParams(
            id="e",
            name="e",
            description="d",
            env_type="executable",
            tools=[_tool("t")],
        )
    )
    [result] = env.step([("t", {"a": 1})])
    assert env.absorbed == [result]


def test_absorb_result_skipped_when_result_invalid():
    """The post-hook must not fire when the executor call fails validation."""
    env = _AbsorbingExecEnv(
        EnvironmentParams(
            id="e",
            name="e",
            description="d",
            env_type="executable",
            tools=[_tool("bad", executor=f"{__name__}._bad_return_executor")],
        )
    )
    with pytest.raises(ToolError):
        env.step([("bad", {})])
    assert env.absorbed == []
