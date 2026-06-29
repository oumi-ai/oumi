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
from oumi.core.types.tool_call import ToolResult
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool


class _MinimalExecEnv(ExecutableEnvironment):
    """Smallest concrete subclass that satisfies the abstract surface."""

    def __init__(self) -> None:
        self._params = EnvironmentParams(
            id="test", name="test", description="d", env_type="executable"
        )
        self._executors = {}

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        yield None


def test_cannot_instantiate_abstract_base():
    """ExecutableEnvironment is abstract — _build_execution_context must be supplied."""
    with pytest.raises(TypeError, match="abstract"):
        ExecutableEnvironment()  # type: ignore[abstract]


def test_default_tool_params_cls_is_executable_tool():
    """ExecutableEnvironment binds to ExecutableTool by default."""
    assert ExecutableEnvironment.tool_params_cls is ExecutableTool


def test_close_is_noop():
    """Default close() returns None without raising."""
    env = _MinimalExecEnv()
    result = env.close()
    assert result is None


def test_absorb_result_is_noop():
    """Default _absorb_result returns None for any ToolResult."""
    env = _MinimalExecEnv()
    tool = ExecutableTool(id="t", name="t", description="d", executor="x.y")
    assert env._absorb_result(tool, ToolResult(output={"ok": True})) is None


def test_step_batch_dispatches_to_step_one():
    """Batch step() dispatches each call to _step_one, which looks up the tool."""
    from oumi.core.configs.params.tool_params import ToolLookupError

    env = _MinimalExecEnv()
    with pytest.raises(ToolLookupError):
        env.step([("tool_a", {})])


def test_step_one_unknown_tool_raises_lookup_error_minimal():
    """_step_one raises a lookup error when the tool id is unknown."""
    from oumi.core.configs.params.tool_params import ToolLookupError

    env = _MinimalExecEnv()
    with pytest.raises(ToolLookupError):
        env._step_one("tool_a", {})


class _EchoExecEnv(ExecutableEnvironment):
    """Concrete env whose executor echoes the context it was handed."""

    def __init__(self, tools: list[ExecutableTool]) -> None:
        self._params = EnvironmentParams(
            id="echo", name="echo", description="d", env_type="executable", tools=tools
        )
        self._executors = {t.id: _echo_executor for t in tools}

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        yield {"ctx_for": tool.id}


def _echo_executor(*, arguments: dict[str, Any], context: Any) -> ToolResult:
    return ToolResult(output={"args": arguments, "context": context})


def test_step_one_dispatches_to_executor_with_context():
    tool = ExecutableTool(id="t", name="t", description="d", executor="x.y")
    env = _EchoExecEnv([tool])
    [result] = env.step([("t", {"a": 1})])
    assert result.output == {"args": {"a": 1}, "context": {"ctx_for": "t"}}


def test_step_one_unknown_tool_raises_lookup_error():
    from oumi.core.configs.params.tool_params import ToolLookupError

    env = _EchoExecEnv([])
    with pytest.raises(ToolLookupError):
        env.step([("missing", {})])


def test_step_one_rejects_non_toolresult_executor_return():
    from oumi.core.configs.params.tool_params import ToolError

    tool = ExecutableTool(id="bad", name="bad", description="d", executor="x.y")
    env = _EchoExecEnv([tool])
    env._executors["bad"] = lambda **_: {"not": "a ToolResult"}
    with pytest.raises(ToolError):
        env.step([("bad", {})])
