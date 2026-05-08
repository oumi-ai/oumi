# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for the abstract ExecutableEnvironment base class."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.executable_environment import (
    ExecutableEnvironment,
    _import_executor,
)
from oumi.environments.executable_tool import ExecutableTool


# Module-scope test executors (so dotted-path resolution works).
def _ok_executor(arguments, ctx):
    return ToolResult(output={"echo": arguments, "ctx": ctx})


def _bad_executor_returns_str(arguments, ctx):
    return "not a ToolResult"


def _executor_raises_value_error(arguments, ctx):
    raise ValueError("boom")


_NOT_CALLABLE = 42


@register_environment("_test_fake_executable")
class _FakeExecutableEnvironment(ExecutableEnvironment):
    """Concrete subclass for testing — yields a sentinel string as the context."""

    tool_params_cls = ExecutableTool
    _executor_context_kwarg = "ctx"

    def __init__(self, params, kwargs=None):
        self._params = params
        self._kwargs = kwargs
        self._executors = {}

    @classmethod
    def from_params(cls, params):
        env = cls(params)
        for tool in params.tools:
            env._executors[tool.id] = _import_executor(tool.executor, tool.id)
        return env

    @contextmanager
    def _build_execution_context(self, tool, arguments):
        yield "fake-ctx-sentinel"


def _make_tool(executor, tool_id="t1"):
    return ExecutableTool(
        id=tool_id,
        name=tool_id.upper(),
        description="A test tool.",
        executor=executor,
    )


def _make_params(tools):
    return EnvironmentParams(
        id="env1",
        name="Env1",
        description="A fake executable env.",
        env_type="_test_fake_executable",
        tools=tools,
    )


def test_step_returns_executor_result():
    params = _make_params([
        _make_tool("tests.unit.environments.test_executable_environment._ok_executor")
    ])
    env = _FakeExecutableEnvironment.from_params(params)
    result = env.step("t1", {"k": "v"})
    assert isinstance(result, ToolResult)
    assert result.output == {"echo": {"k": "v"}, "ctx": "fake-ctx-sentinel"}


def test_step_rejects_non_toolresult_return():
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_executable_environment._bad_executor_returns_str"
        )
    ])
    env = _FakeExecutableEnvironment.from_params(params)
    with pytest.raises(ToolError, match="ToolResult"):
        env.step("t1", {})


def test_step_propagates_executor_value_error():
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_executable_environment._executor_raises_value_error"
        )
    ])
    env = _FakeExecutableEnvironment.from_params(params)
    with pytest.raises(ValueError, match="boom"):
        env.step("t1", {})


def test_import_executor_rejects_missing_module():
    with pytest.raises(ValueError, match="could not import module"):
        _import_executor("oumi.does_not_exist.foo", "t1")


def test_import_executor_rejects_missing_attr():
    with pytest.raises(ValueError, match="has no attribute"):
        _import_executor("oumi.environments.executable_tool.NopeNotHere", "t1")


def test_import_executor_rejects_non_callable():
    with pytest.raises(ValueError, match="not callable"):
        _import_executor(
            "tests.unit.environments.test_executable_environment._NOT_CALLABLE",
            "t1",
        )


def test_import_executor_rejects_non_dotted_path():
    with pytest.raises(ValueError, match="dotted path"):
        _import_executor("just_a_name", "t1")


def test_close_default_is_no_op():
    params = _make_params([
        _make_tool("tests.unit.environments.test_executable_environment._ok_executor")
    ])
    env = _FakeExecutableEnvironment.from_params(params)
    env.close()  # should not raise
