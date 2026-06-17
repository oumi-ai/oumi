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

"""Executable environment backed by a rollback-isolated SQLite session."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.environments.db_isolation import RollbackSession, materialize_sqlite_snapshot
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool


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
    """Runs SQL-executing tools against a rollback-isolated SQLite session.

    One instance owns one session for the duration of an episode. Executors
    must NOT commit; the env rolls back on ``close()`` so writes never persist.
    ``requires_isolation()`` is ``True``, so the router builds a fresh instance
    (hence a fresh session) per rollout. See ``db_isolation`` for the contract.
    """

    _executor_context_kwarg: ClassVar[str] = "db"

    def __init__(self, params: EnvironmentParams, session: RollbackSession) -> None:
        """Bind the env to its params and an already-open rollback session."""
        self._params = params
        self._session = session
        self._executors = {
            tool.id: _import_executor(tool.executor, tool.id) for tool in params.tools
        }

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DatabaseExecutableEnvironment:
        """Build the env, opening a rollback session over its configured DB."""
        kwargs = dict(params.env_kwargs or {})
        db_path = kwargs.get("db_path")
        schema_sql = kwargs.get("schema_sql")
        if db_path:
            # Shared snapshot: connect read-side, roll back on close.
            session = RollbackSession(db_path)
        elif schema_sql:
            # Inline: build a fresh per-rollout DB this instance owns.
            snapshot = materialize_sqlite_snapshot(
                schema_sql=schema_sql, seed_sql=kwargs.get("seed_sql")
            )
            session = RollbackSession(snapshot, owns_file=True)
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
    ) -> Iterator[Any]:
        """Yield the episode's connection (uncommitted writes persist within it)."""
        yield self._session.connection

    def close(self) -> None:
        """Roll back the episode's writes and tear down the session."""
        self._session.close()
