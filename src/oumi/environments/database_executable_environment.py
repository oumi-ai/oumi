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

"""SQLAlchemy-backed environment that runs user-supplied executors.

Skeleton phase: declares the class shape, the env-type registration, and the
per-env kwargs dataclass. Engine construction, dialect-aware safety guards,
``DBAPIError`` auto-wrapping, audit logging, and the per-call connection
checkout land in follow-on phases. ``from_params`` and
``_build_execution_context`` therefore raise ``NotImplementedError`` here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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

    The DB *is* the state. Each ``_step_one`` will check out a connection
    from the SQLAlchemy pool, run the executor in autocommit mode, and
    return the connection. SQL errors that escape the executor will be
    auto-wrapped as a structured ``ToolResult`` so the agent can self-correct.

    This phase ships the skeleton: class registered, kwargs shape declared,
    abstract hooks left to raise ``NotImplementedError`` until the engine
    implementation phase.
    """

    tool_params_cls = DatabaseExecutableTool
    _executor_context_kwarg: ClassVar[str] = "db"

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DatabaseExecutableEnvironment:
        """Build a ``DatabaseExecutableEnvironment`` from its params object."""
        raise NotImplementedError(
            "DatabaseExecutableEnvironment.from_params is not yet implemented "
            "(skeleton phase). The implementation phase wires the SQLAlchemy "
            "engine, dialect guards, and fail-fast SELECT 1."
        )

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        """Yield a per-call SQLAlchemy ``Connection`` (implementation pending)."""
        raise NotImplementedError(
            "DatabaseExecutableEnvironment._build_execution_context is not yet "
            "implemented (skeleton phase)."
        )
        yield None  # unreachable; satisfies the Iterator return type
