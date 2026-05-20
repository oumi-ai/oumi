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

"""SQLAlchemy-backed environment that runs user-supplied executors."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, ClassVar

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.database_connection_params import (
    DatabaseConnectionConfig,
)
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.environments.database_executable_tool import DatabaseExecutableTool
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool


@dataclass
class DatabaseExecutableEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for ``DatabaseExecutableEnvironment``."""

    connection: DatabaseConnectionConfig | None = None
    read_only: bool = False
    statement_timeout_ms: int | None = None
    audit: bool = False

    def __post_init__(self) -> None:
        """Coerce ``connection`` dict into a ``DatabaseConnectionConfig``."""
        if isinstance(self.connection, dict):
            self.connection = DatabaseConnectionConfig(**self.connection)

    def __finalize_and_validate__(self) -> None:
        """Validate connection presence and numeric bounds."""
        if self.connection is None:
            raise ValueError(
                "DatabaseExecutableEnvironmentKwargs.connection is required."
            )
        if self.statement_timeout_ms is not None and self.statement_timeout_ms <= 0:
            raise ValueError(
                f"statement_timeout_ms must be > 0, got {self.statement_timeout_ms}."
            )


@register_environment("database")
class DatabaseExecutableEnvironment(ExecutableEnvironment):
    """Environment that runs user-supplied executors against a real SQL database.

    The DB *is* the state. Each tool call checks out a connection from the
    SQLAlchemy pool, runs the executor in autocommit mode, and returns the
    connection. SQL errors that escape the executor are auto-wrapped as a
    structured ``ToolResult`` so the agent can self-correct.
    """

    tool_params_cls = DatabaseExecutableTool
    _executor_context_kwarg: ClassVar[str] = "db"

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DatabaseExecutableEnvironment:
        """Build a ``DatabaseExecutableEnvironment`` from its params object."""
        raise NotImplementedError

    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> AbstractContextManager[Any]:
        """Check out a SQLAlchemy ``Connection`` for one tool call."""
        raise NotImplementedError
