# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""SQLAlchemy-backed environment that runs user-supplied executors."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import sqlalchemy
import sqlalchemy.engine
import sqlalchemy.event as sa_event
import sqlalchemy.exc

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.database_connection_params import (
    DatabaseConnectionConfig,
)
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_executable_tool import DatabaseExecutableTool
from oumi.environments.executable_environment import (
    ExecutableEnvironment,
    _import_executor,
)
from oumi.environments.executable_tool import ExecutableTool
from oumi.utils.logging import logger


def _install_dialect_guards(
    engine: sqlalchemy.engine.Engine,
    kwargs: "DatabaseExecutableEnvironmentKwargs",
) -> None:
    """Install dialect-specific read-only / statement-timeout session settings.

    Wired as a SQLAlchemy ``connect`` event so every new pooled connection
    gets the right session config on first acquisition.
    """
    dialect = engine.dialect.name
    read_only = kwargs.read_only
    timeout_ms = kwargs.statement_timeout_ms

    @sa_event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        try:
            if dialect == "postgresql":
                if read_only:
                    cursor.execute("SET default_transaction_read_only = on")
                if timeout_ms is not None:
                    cursor.execute(f"SET statement_timeout = {int(timeout_ms)}")
            elif dialect == "mysql":
                if read_only:
                    cursor.execute("SET SESSION TRANSACTION READ ONLY")
                if timeout_ms is not None:
                    cursor.execute(
                        f"SET SESSION max_execution_time = {int(timeout_ms)}"
                    )
            elif dialect == "sqlite":
                if read_only:
                    cursor.execute("PRAGMA query_only = ON")
                if timeout_ms is not None:
                    logger.warning(
                        "SQLite does not support statement_timeout; ignoring "
                        "statement_timeout_ms=%s.",
                        timeout_ms,
                    )
            else:
                if read_only or timeout_ms is not None:
                    logger.warning(
                        "Dialect '%s' has no registered guard handler; "
                        "read_only=%s and statement_timeout_ms=%s will not be "
                        "enforced. Set them at the DB role / DSN level instead.",
                        dialect, read_only, timeout_ms,
                    )
        finally:
            cursor.close()


@dataclass
class DatabaseExecutableEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for DatabaseExecutableEnvironment."""

    connection: DatabaseConnectionConfig | None = None
    read_only: bool = False
    statement_timeout_ms: int | None = None
    audit: bool = False

    def __post_init__(self) -> None:
        """Coerce ``connection`` dict into a ``DatabaseConnectionConfig``."""
        if isinstance(self.connection, dict):
            self.connection = DatabaseConnectionConfig(**self.connection)

    def __finalize_and_validate__(self) -> None:
        """Validate kwargs and the nested connection config."""
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

    The DB *is* the state. Each ``step`` checks out a connection from the
    SQLAlchemy pool, runs the executor in autocommit mode, and returns the
    connection. SQL errors that escape the executor are auto-wrapped as a
    structured ``ToolResult`` so the agent can self-correct (added in a
    later task).
    """

    tool_params_cls = DatabaseExecutableTool
    _executor_context_kwarg: ClassVar[str] = "db"

    def __init__(
        self,
        params: EnvironmentParams,
        kwargs: DatabaseExecutableEnvironmentKwargs,
        engine: sqlalchemy.engine.Engine,
    ) -> None:
        """Initialize the env. Use ``from_params`` rather than calling directly."""
        self._params = params
        self._kwargs = kwargs
        self._engine = engine
        self._executors: dict[str, Any] = {}

    @classmethod
    def from_params(
        cls, params: EnvironmentParams
    ) -> DatabaseExecutableEnvironment:
        """Build a DatabaseExecutableEnvironment from its params object."""
        kwargs = DatabaseExecutableEnvironmentKwargs(**(params.env_kwargs or {}))
        kwargs.finalize_and_validate()
        assert kwargs.connection is not None  # validated above

        url = kwargs.connection.resolve_url()
        engine_kwargs: dict[str, Any] = {
            "pool_pre_ping": kwargs.connection.pool_pre_ping,
            "isolation_level": "AUTOCOMMIT",
            "future": True,
        }
        # SQLite uses SingletonThreadPool which does not accept pool_size /
        # max_overflow; skip those args for in-process databases.
        if url.get_dialect().name != "sqlite":
            engine_kwargs["pool_size"] = kwargs.connection.pool_size
            engine_kwargs["max_overflow"] = kwargs.connection.pool_max_overflow
        engine = sqlalchemy.create_engine(url, **engine_kwargs)
        _install_dialect_guards(engine, kwargs)

        # Fail-fast: prove the connection works before any tool runs.
        try:
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
        except Exception as e:
            engine.dispose()
            raise ValueError(
                f"Failed to connect to database for env '{params.id}': {e}"
            ) from e

        env = cls(params, kwargs, engine)
        for tool in params.tools:
            assert isinstance(tool, DatabaseExecutableTool)
            env._executors[tool.id] = _import_executor(tool.executor, tool.id)
        return env

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[sqlalchemy.engine.Connection]:
        """Check out a connection from the pool for one tool call."""
        conn = self._engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        """Dispose the engine and its connection pool."""
        self._engine.dispose()

    def _invoke_executor(
        self,
        executor,
        arguments,
        ctx,
        tool,
    ):
        """Run the executor; auto-wrap SQL errors as structured ToolResults.

        Returns ``(result, was_auto_wrapped)``. The flag tells
        ``ExecutableEnvironment._step_one`` (the parent's ``step``) to skip
        ``output_schema`` validation when the wrap shape was substituted for
        the executor's normal return value.
        """
        try:
            return executor(arguments=arguments, db=ctx), False
        except sqlalchemy.exc.DBAPIError as e:
            wrapped = ToolResult(
                output={
                    "status": "error",
                    "error": type(e).__name__,
                    "message": str(e.orig) if e.orig else str(e),
                    "sql_state": (
                        getattr(e.orig, "sqlstate", None) if e.orig else None
                    ),
                }
            )
            return wrapped, True

    def _absorb_result(self, tool, result) -> None:
        """DB envs hold state in the DB; reject any ToolResult.updated_state."""
        if result.updated_state is not None:
            raise ToolError(
                f"DatabaseExecutable tool '{tool.id}' returned updated_state; "
                f"DB-backed envs hold state in the database, not in ToolResult."
            )
