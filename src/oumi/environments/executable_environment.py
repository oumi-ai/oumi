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

"""Abstract base for envs backed by user-supplied dotted-path executors."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolLookupError, ToolParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.utils import import_executor, validate_executor_result


class ExecutableEnvironment(BaseEnvironment):
    """Abstract base for envs that dispatch tool calls to Python executors.

    Each tool declares its executor as a dotted import path; the base resolves
    them into ``_executors`` at construction. Subclasses supply the per-call
    execution context (DB connection, HTTP client, FS root, ...) by
    implementing ``_build_execution_context``. The base owns tool lookup,
    argument and result validation, the ``_absorb_result`` post-hook, and the
    ``close`` lifecycle. Executors are invoked as
    ``executor(arguments=<dict>, context=<ctx>)`` and must return a
    ``ToolResult``. Result validation runs inside the execution context so a
    transactional context manager sees a validation failure and can roll back;
    ``_absorb_result`` runs only after the context exits cleanly.
    """

    tool_params_cls: type[ToolParams] = ExecutableTool

    def __init__(self, params: EnvironmentParams) -> None:
        """Resolve each tool's dotted-path executor into ``_executors``."""
        self._params = params
        self._executors: dict[str, Callable[..., Any]] = {
            tool.id: import_executor(tool.executor, tool.id) for tool in params.tools
        }

    @abstractmethod
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> AbstractContextManager[Any]:
        """Yield the per-call execution context (DB conn, HTTP client, ...)."""

    def _absorb_result(self, tool: ExecutableTool, result: ToolResult) -> None:
        """Post-hook called after a successful executor call. Default no-op."""

    def close(self) -> None:
        """Release any resources owned by this env. Default no-op."""

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute a batch of tool calls; results are returned in input order."""
        return [self._step_one(tool_id, arguments) for tool_id, arguments in calls]

    def _lookup_tool(self, tool_id: str) -> ExecutableTool:
        for tool in self._params.tools:
            if tool.id == tool_id:
                return tool
        raise ToolLookupError(
            f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
            f"Available tools: {[t.id for t in self._params.tools]}"
        )

    def _step_one(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._lookup_tool(tool_id)
        tool.validate_arguments(arguments)
        with self._build_execution_context(tool, arguments) as ctx:
            result = self._executors[tool_id](arguments=arguments, context=ctx)
            validated = validate_executor_result(tool, result)
        self._absorb_result(tool, validated)
        return validated
