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

"""Postgres integration tests for DatabaseExecutableEnvironment.

Gated behind the ``requires_postgres`` marker. Run via:

    uv run --extra dev pytest -m requires_postgres

Starts a Postgres container per session via testcontainers-python.
Verifies dialect-specific behavior that SQLite cannot exercise:
``default_transaction_read_only``, real ``statement_timeout``, and the
``IntegrityError.sqlstate`` propagation.
"""

from __future__ import annotations

import pytest
import sqlalchemy

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)
from oumi.environments.database_executable_tool import DatabaseExecutableTool

pytestmark = pytest.mark.requires_postgres


# Module-scope executors (dotted-path resolution).


def _setup_executor(arguments, db):
    db.execute(
        sqlalchemy.text(
            "CREATE TABLE IF NOT EXISTS patients ("
            "id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)"
        )
    )
    db.execute(
        sqlalchemy.text(
            "INSERT INTO patients (id, name) VALUES (1, 'Jane') "
            "ON CONFLICT (id) DO NOTHING"
        )
    )
    return ToolResult(output={"status": "ok"})


def _try_insert_executor(arguments, db):
    db.execute(sqlalchemy.text("INSERT INTO patients (id, name) VALUES (2, 'Marcus')"))
    return ToolResult(output={"status": "ok"})


def _slow_query_executor(arguments, db):
    db.execute(sqlalchemy.text("SELECT pg_sleep(2)"))
    return ToolResult(output={"status": "ok"})


def _duplicate_pk_executor(arguments, db):
    db.execute(sqlalchemy.text("INSERT INTO patients (id, name) VALUES (1, 'X')"))
    return ToolResult(output={"status": "should not reach"})


@pytest.fixture(scope="session")
def postgres_dsn():
    """Yield a SQLAlchemy DSN for a fresh Postgres container."""
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg.get_connection_url().replace("postgresql://", "postgresql+psycopg://")


@pytest.fixture
def setup_db(postgres_dsn, monkeypatch):
    monkeypatch.setenv("PG_TEST_DSN", postgres_dsn)
    setup_params = EnvironmentParams(
        id="setup",
        name="setup",
        description="seed",
        env_type="database",
        env_kwargs={
            "connection": {"dsn_env_var": "PG_TEST_DSN"},
        },
        tools=[
            DatabaseExecutableTool(
                id="setup",
                name="setup",
                description="seed",
                executor=(
                    "tests.integration.environments."
                    "test_database_executable_environment_postgres._setup_executor"
                ),
                read_only=False,
            )
        ],
    )
    env = DatabaseExecutableEnvironment.from_params(setup_params)
    try:
        env.step("setup", {})
    finally:
        env.close()
    yield postgres_dsn


def test_read_only_blocks_writes(setup_db, monkeypatch):
    monkeypatch.setenv("PG_TEST_DSN", setup_db)
    params = EnvironmentParams(
        id="ro",
        name="ro",
        description="ro",
        env_type="database",
        env_kwargs={
            "connection": {"dsn_env_var": "PG_TEST_DSN"},
            "read_only": True,
        },
        tools=[
            DatabaseExecutableTool(
                id="t",
                name="t",
                description="t",
                executor=(
                    "tests.integration.environments."
                    "test_database_executable_environment_postgres._try_insert_executor"
                ),
                read_only=True,
            )
        ],
    )
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        result = env.step("t", {})
        assert isinstance(result.output, dict)
        assert result.output["status"] == "error"
        msg = result.output["message"].lower()
        assert "read-only" in msg or "read only" in msg
    finally:
        env.close()


def test_statement_timeout_cancels_long_query(setup_db, monkeypatch):
    monkeypatch.setenv("PG_TEST_DSN", setup_db)
    params = EnvironmentParams(
        id="slow",
        name="slow",
        description="slow",
        env_type="database",
        env_kwargs={
            "connection": {"dsn_env_var": "PG_TEST_DSN"},
            "statement_timeout_ms": 200,
        },
        tools=[
            DatabaseExecutableTool(
                id="slow",
                name="slow",
                description="slow",
                executor=(
                    "tests.integration.environments."
                    "test_database_executable_environment_postgres._slow_query_executor"
                ),
                read_only=True,
            )
        ],
    )
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        result = env.step("slow", {})
        assert isinstance(result.output, dict)
        assert result.output["status"] == "error"
        msg = result.output["message"].lower()
        assert "statement timeout" in msg or "canceling statement" in msg
    finally:
        env.close()


def test_integrity_error_carries_sqlstate(setup_db, monkeypatch):
    monkeypatch.setenv("PG_TEST_DSN", setup_db)
    params = EnvironmentParams(
        id="dup",
        name="dup",
        description="dup",
        env_type="database",
        env_kwargs={
            "connection": {"dsn_env_var": "PG_TEST_DSN"},
        },
        tools=[
            DatabaseExecutableTool(
                id="dup",
                name="dup",
                description="dup",
                executor=(
                    "tests.integration.environments."
                    "test_database_executable_environment_postgres._duplicate_pk_executor"
                ),
                read_only=False,
            )
        ],
    )
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        result = env.step("dup", {})
        assert isinstance(result.output, dict)
        assert result.output["error"] == "IntegrityError"
        # Postgres unique-violation SQLSTATE is 23505.
        assert result.output["sql_state"] == "23505"
    finally:
        env.close()
