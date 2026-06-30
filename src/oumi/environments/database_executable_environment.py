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

"""Executable environment backed by a Database-isolated SQLite session."""

from __future__ import annotations

import contextvars
import importlib
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_session import (
    DatabaseSession,
    materialize_sqlite_snapshot,
)
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool

_ACTIVE_CONNECTION: contextvars.ContextVar[sqlite3.Connection] = contextvars.ContextVar(
    "oumi_active_db_connection"
)


def current_connection() -> sqlite3.Connection:
    """Return the SQLite connection bound to the in-flight tool call.

    Raises:
        RuntimeError: if called outside a tool execution (nothing is bound).
    """
    try:
        return _ACTIVE_CONNECTION.get()
    except LookupError as e:
        raise RuntimeError(
            "current_connection() called outside a DatabaseExecutableEnvironment "
            "tool execution; no connection is bound."
        ) from e


@contextmanager
def using_connection(connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Bind ``connection`` as the active connection for the duration of the block.

    The environment uses this internally per call. It is also the supported way
    to run an executor directly — in a unit test or a grading path — without
    threading the connection through it::

        with using_connection(conn):
            run_my_tool(arg=1)
    """
    token = _ACTIVE_CONNECTION.set(connection)
    try:
        yield connection
    finally:
        _ACTIVE_CONNECTION.reset(token)


def _import_executor(dotted: str, tool_id: str) -> Callable[..., Any]:
    """Resolve a dotted import path to a callable."""
    module_path, _, attr = dotted.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"Tool '{tool_id}': executor '{dotted}' must be a dotted import path."
        )
    module = importlib.import_module(module_path)
    executor = getattr(module, attr, None)
    if not callable(executor):
        raise ValueError(
            f"Tool '{tool_id}': executor '{dotted}' did not resolve to a callable."
        )
    return executor


@register_environment("database")
class DatabaseExecutableEnvironment(ExecutableEnvironment):
    """Runs SQL-executing tools against an isolated database session."""

    def __init__(self, params: EnvironmentParams, session: DatabaseSession) -> None:
        """Bind the env to its params and an already-open Database session."""
        self._params = params
        self._session = session
        self._executors = {
            tool.id: _import_executor(tool.executor, tool.id) for tool in params.tools
        }

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DatabaseExecutableEnvironment:
        """Build the env, opening a Database session over its configured DB.

        ``db_path`` shares one snapshot file across rollouts (Database isolation,
        scales to large DBs). It is safe for concurrent *readers*, but SQLite
        serializes concurrent *writers* on one file, so concurrent rollouts that
        write will contend — those tasks should use ``schema_sql`` (a fresh
        per-rollout file) until copy-on-write isolation lands.
        """
        kwargs = dict(params.env_kwargs or {})
        db_path = kwargs.get("db_path")
        schema_sql = kwargs.get("schema_sql")
        if db_path:
            session = DatabaseSession(db_path)
        elif schema_sql:
            snapshot = materialize_sqlite_snapshot(
                schema_sql=schema_sql, seed_sql=kwargs.get("seed_sql")
            )
            session = DatabaseSession(snapshot, owns_file=True)
        else:
            raise ValueError(
                f"DatabaseExecutableEnvironment '{params.id}': env_kwargs must "
                f"provide either 'db_path' or 'schema_sql'."
            )
        return cls(params, session)

    def requires_isolation(self) -> bool:
        """Each rollout needs its own session; never share across samples."""
        return True

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[None]:
        """Bind the episode's connection as the active connection for this call."""
        with using_connection(self._session.connection):
            yield None

    def _invoke_executor(
        self, executor: Callable[..., Any], arguments: dict[str, Any], ctx: Any
    ) -> Any:
        """Call the executor with unpacked tool params and coerce the return."""
        result = executor(**arguments)
        return result if isinstance(result, ToolResult) else ToolResult(output=result)

    def close(self) -> None:
        """Roll back the episode's writes and tear down the session."""
        self._session.close()
