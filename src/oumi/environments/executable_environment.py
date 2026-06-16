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

import jsonschema
from abc import abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, ClassVar

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError, ToolLookupError, ToolParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.executable_tool import ExecutableTool


class ExecutableEnvironment(BaseEnvironment):
    """Abstract base for envs that run user-supplied dotted-path executors.

    Subclasses supply the per-call execution context (DB connection, HTTP
    client, FS root, ...) by implementing ``_build_execution_context`` as a
    context manager. The base owns executor resolution, result validation,
    schema validation, the ``_absorb_result`` post-hook, and the ``close``
    lifecycle.
    """

    tool_params_cls: type[ToolParams] = ExecutableTool

    #: Keyword name under which subclasses pass the execution context to
    #: user executors. Defaults to ``"context"``; concrete subclasses such
    #: as ``DatabaseExecutableEnvironment`` override this (e.g. to ``"db"``).
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

    def _validate_result(self, tool: ExecutableTool, result: Any) -> ToolResult:
        if not isinstance(result, ToolResult):
            raise ToolError(
                f"Tool '{tool.id}' executor must return ToolResult, got "
                f"{type(result).__name__}."
            )
        if tool.output_schema is not None:
            try:
                jsonschema.validate(result.output, tool.output_schema)
            except jsonschema.ValidationError as e:
                raise ToolError(
                    f"Tool '{tool.id}' executor output failed schema validation: {e}"
                ) from e
        return result

    def _step_one(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._lookup_tool(tool_id)
        tool.validate_arguments(arguments)
        with self._build_execution_context(tool, arguments) as ctx:
            result = self._executors[tool_id](
                **{"arguments": arguments, self._executor_context_kwarg: ctx}
            )
        validated = self._validate_result(tool, result)
        self._absorb_result(tool, validated)
        return validated
