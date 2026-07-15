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

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_session import (
    DatabaseSession,
    materialize_sqlite_snapshot,
)
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.utils import parse_env_kwargs


@dataclass
class DatabaseExecutableEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for :class:`DatabaseExecutableEnvironment`."""

    db_path: Path | str | None = None
    schema_sql: str | None = None
    seed_sql: str | None = None

    def __finalize_and_validate__(self) -> None:
        """Validate the database source configuration."""
        if self.seed_sql and not self.schema_sql:
            raise ValueError("seed_sql requires schema_sql.")
        if bool(self.db_path) == bool(self.schema_sql):
            raise ValueError("Provide exactly one of db_path or schema_sql.")


@contextmanager
def _savepoint(connection: sqlite3.Connection, name: str) -> Iterator[None]:
    # Randomized so model SQL can't RELEASE/ROLLBACK TO our savepoint and break cleanup.
    sp = f"{name}_{uuid.uuid4().hex}"
    connection.execute(f"SAVEPOINT {sp}")
    try:
        yield
    except BaseException:
        # Best-effort rollback: a cleanup failure must not mask the real error.
        with suppress(sqlite3.Error):
            connection.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            connection.execute(f"RELEASE SAVEPOINT {sp}")
        raise
    else:
        connection.execute(f"RELEASE SAVEPOINT {sp}")


@register_environment("database")
class DatabaseExecutableEnvironment(ExecutableEnvironment):
    """Runs SQL-executing tools against an isolated database session."""

    def __init__(self, params: EnvironmentParams, session: DatabaseSession) -> None:
        """Bind the env to its params and an already-open Database session."""
        super().__init__(params)
        self._session = session

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DatabaseExecutableEnvironment:
        """Build the env, opening a session over its configured DB.

        ``db_path`` shares one snapshot file across rollouts (scales to large DBs)
        and is safe for concurrent *readers*, but SQLite serializes concurrent
        *writers* on one file, so write-heavy concurrent rollouts should use
        ``schema_sql`` (a fresh per-rollout file) instead.
        """
        kwargs = parse_env_kwargs(
            DatabaseExecutableEnvironmentKwargs,
            params,
            env_label="DatabaseExecutableEnvironment",
        )
        if kwargs.db_path:
            path = Path(kwargs.db_path)
            if not path.is_file():
                raise ValueError(
                    f"DatabaseExecutableEnvironment '{params.id}': db_path "
                    f"must reference an existing file: {path}"
                )
            session = DatabaseSession(path)
        else:
            assert kwargs.schema_sql is not None
            snapshot = materialize_sqlite_snapshot(
                schema_sql=kwargs.schema_sql, seed_sql=kwargs.seed_sql
            )
            session = DatabaseSession(snapshot, owns_file=True)
        try:
            return cls(params, session)
        except BaseException:
            session.close()
            raise

    def requires_isolation(self) -> bool:
        """Each rollout needs its own session; never share across samples."""
        return True

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute a batch atomically within the rollout transaction."""
        with _savepoint(self._session.connection, "oumi_batch"):
            return super().step(calls)

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[sqlite3.Connection]:
        """Bind a savepoint-scoped connection and enforce read-only tools."""
        connection = self._session.connection
        if tool.read_only:
            connection.execute("PRAGMA query_only = ON")
        try:
            with _savepoint(connection, "oumi_tool_call"):
                yield connection
        finally:
            if tool.read_only:
                # Best-effort reset: don't let it mask an in-flight executor error.
                with suppress(sqlite3.Error):
                    connection.execute("PRAGMA query_only = OFF")

    def close(self) -> None:
        """Roll back the episode's writes and tear down the session."""
        self._session.close()
