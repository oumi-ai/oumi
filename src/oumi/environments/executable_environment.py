# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Abstract base class for environments backed by user-supplied dotted-path executors."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, ClassVar, cast

import jsonschema

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import (
    ToolError,
    ToolParams,
)
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.executable_tool import ExecutableTool

_MISSING = object()


def _import_executor(dotted: str, tool_id: str) -> Callable[..., Any]:
    """Resolve a dotted import path to a callable. Raises ValueError on failure."""
    module_path, _, attr = dotted.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"ExecutableTool '{tool_id}' executor '{dotted}' must be a dotted path."
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"ExecutableTool '{tool_id}' executor '{dotted}': "
            f"could not import module '{module_path}': {e}"
        ) from e
    fn = getattr(module, attr, _MISSING)
    if fn is _MISSING:
        raise ValueError(
            f"ExecutableTool '{tool_id}' executor '{dotted}': "
            f"module '{module_path}' has no attribute '{attr}'."
        )
    if not callable(fn):
        raise ValueError(
            f"ExecutableTool '{tool_id}' executor '{dotted}' is not callable."
        )
    return fn


class ExecutableEnvironment(BaseEnvironment, ABC):
    """Abstract base for envs that run user-supplied dotted-path executors.

    Subclasses provide the per-call execution context (DB connection, HTTP
    client, ...) by implementing ``_build_execution_context`` as a context
    manager. The orchestration (executor resolution, ``ToolResult``
    validation, schema validation, ``_absorb_result`` post-hook, ``close``
    lifecycle) lives here.
    """

    tool_params_cls: type[ToolParams] = ExecutableTool

    #: Keyword name under which subclasses pass the execution context to
    #: user executors. Defaults to ``"context"``; ``DatabaseExecutableEnvironment``
    #: overrides to ``"db"``.
    _executor_context_kwarg: ClassVar[str] = "context"

    _params: EnvironmentParams
    _executors: dict[str, Callable[..., Any]]

    @abstractmethod
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> AbstractContextManager[Any]:
        """Yield the per-call execution context (DB conn, HTTP client, ...)."""

    def _absorb_result(self, tool: ExecutableTool, result: ToolResult) -> None:
        """Post-hook called after a successful executor call. Default no-op."""
        return None

    def close(self) -> None:
        """Release any resources owned by this env. Default no-op."""
        return None

    def _invoke_executor(
        self,
        executor: Callable[..., Any],
        arguments: dict[str, Any],
        ctx: Any,
        tool: ExecutableTool,
    ) -> tuple[ToolResult, bool]:
        """Run the executor; return (result, was_auto_wrapped).

        Default: pass ``ctx`` via ``_executor_context_kwarg``; never auto-wrap.
        Subclasses override to translate transport-level exceptions into
        structured ``ToolResult``s.
        """
        return (
            executor(arguments=arguments, **{self._executor_context_kwarg: ctx}),
            False,
        )

    def _lookup_tool(self, tool_id: str) -> ExecutableTool:
        for tool in self._params.tools:
            if tool.id == tool_id:
                return cast(ExecutableTool, tool)
        raise ValueError(
            f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
            f"Available tools: {[tool.id for tool in self._params.tools]}"
        )

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a single tool call and return its result."""
        tool = self._lookup_tool(tool_id)
        tool.validate_arguments(arguments)
        executor = self._executors[tool_id]

        with self._build_execution_context(tool, arguments) as ctx:
            result, auto_wrapped = self._invoke_executor(executor, arguments, ctx, tool)

        if not isinstance(result, ToolResult):
            raise ToolError(
                f"Executor '{tool.executor}' must return ToolResult, "
                f"got {type(result).__name__}."
            )
        if tool.output_schema is not None and not auto_wrapped:
            try:
                jsonschema.validate(result.output, tool.output_schema)
            except jsonschema.ValidationError as e:
                raise ToolError(
                    f"Tool '{tool_id}' output failed schema validation: {e.message}"
                ) from e
        self._absorb_result(tool, result)
        return result
