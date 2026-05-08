# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for DatabaseExecutableEnvironment (SQLite-backed)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import sqlalchemy
import sqlalchemy.exc

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
    DatabaseExecutableEnvironmentKwargs,
)
from oumi.environments.database_executable_tool import DatabaseExecutableTool


# Test executors at module scope (dotted-path resolution).
def _select_one_executor(arguments, db):
    rows = db.execute(sqlalchemy.text("SELECT 1 AS one")).mappings().all()
    return ToolResult(output={"rows": [dict(r) for r in rows]})


def _make_tool(executor, tool_id="t1", read_only=True, **extra):
    return DatabaseExecutableTool(
        id=tool_id,
        name=tool_id.upper(),
        description="A DB test tool.",
        executor=executor,
        read_only=read_only,
        **extra,
    )


def _make_params(tools, env_kwargs=None, env_id="env1"):
    if env_kwargs is None:
        env_kwargs = {
            "connection": {
                "driver": "sqlite",
                "database": ":memory:",
            }
        }
    return EnvironmentParams(
        id=env_id,
        name="Env1",
        description="A DB env.",
        env_type="database",
        tools=tools,
        env_kwargs=env_kwargs,
    )


def test_from_params_constructs_engine():
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_database_executable_environment._select_one_executor"
        )
    ])
    env = DatabaseExecutableEnvironment.from_params(params)
    assert isinstance(env, DatabaseExecutableEnvironment)
    assert isinstance(env._kwargs, DatabaseExecutableEnvironmentKwargs)
    assert env._engine is not None
    env.close()


def test_from_params_requires_connection():
    params = _make_params(
        [
            _make_tool(
                "tests.unit.environments.test_database_executable_environment._select_one_executor"
            )
        ],
        env_kwargs={},
    )
    with pytest.raises(ValueError, match="connection"):
        DatabaseExecutableEnvironment.from_params(params)


def test_from_params_fail_fast_on_bad_url():
    params = _make_params(
        [
            _make_tool(
                "tests.unit.environments.test_database_executable_environment._select_one_executor"
            )
        ],
        env_kwargs={
            "connection": {
                "driver": "sqlite",
                "database": "/nonexistent_dir_for_test/db.sqlite",
            }
        },
    )
    with pytest.raises(ValueError, match="Failed to connect"):
        DatabaseExecutableEnvironment.from_params(params)


def test_close_disposes_engine():
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_database_executable_environment._select_one_executor"
        )
    ])
    env = DatabaseExecutableEnvironment.from_params(params)
    engine = env._engine
    assert engine is not None
    env.close()
    # After dispose() we still hold the engine reference; the test confirms
    # close() ran without error.
    assert env._engine is engine


def _create_table_executor(arguments, db):
    db.execute(sqlalchemy.text(
        "CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT)"
    ))
    db.execute(sqlalchemy.text(
        "INSERT INTO patients (id, name) VALUES (1, 'Jane'), (2, 'Marcus')"
    ))
    return ToolResult(output={"status": "ok"})


def _list_patients_executor(arguments, db):
    rows = db.execute(
        sqlalchemy.text("SELECT id, name FROM patients ORDER BY id")
    ).mappings().all()
    return ToolResult(output={"patients": [dict(r) for r in rows]})


def test_step_runs_executor_and_returns_rows():
    # Use a file-backed SQLite DB so two tool calls hit the same database.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        env_kwargs = {
            "connection": {
                "driver": "sqlite",
                "database": str(db_path),
            }
        }
        params = _make_params(
            [
                _make_tool(
                    "tests.unit.environments.test_database_executable_environment._create_table_executor",
                    tool_id="setup",
                    read_only=False,
                ),
                _make_tool(
                    "tests.unit.environments.test_database_executable_environment._list_patients_executor",
                    tool_id="list",
                ),
            ],
            env_kwargs=env_kwargs,
        )
        env = DatabaseExecutableEnvironment.from_params(params)
        try:
            setup_result = env.step("setup", {})
            assert setup_result.output == {"status": "ok"}

            list_result = env.step("list", {})
            assert list_result.output == {
                "patients": [
                    {"id": 1, "name": "Jane"},
                    {"id": 2, "name": "Marcus"},
                ]
            }
        finally:
            env.close()


def _try_insert_executor(arguments, db):
    """Catches OperationalError and returns the message."""
    try:
        db.execute(
            sqlalchemy.text("INSERT INTO patients (id, name) VALUES (3, 'Aisha')")
        )
        return ToolResult(output={"status": "ok"})
    except sqlalchemy.exc.OperationalError as e:
        return ToolResult(
            output={"status": "error", "message": str(e.orig) if e.orig else str(e)}
        )


def test_dialect_guards_sqlite_read_only_rejects_writes():
    """With read_only=True the SQLite connection has PRAGMA query_only=ON."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "ro.db"
        # Seed a row first under a separate, non-read-only env.
        seed_params = _make_params(
            [
                _make_tool(
                    "tests.unit.environments.test_database_executable_environment._create_table_executor",
                    tool_id="setup",
                    read_only=False,
                )
            ],
            env_kwargs={
                "connection": {"driver": "sqlite", "database": str(db_path)}
            },
        )
        seed_env = DatabaseExecutableEnvironment.from_params(seed_params)
        try:
            seed_env.step("setup", {})
        finally:
            seed_env.close()

        # Now open a read-only env on the same file.
        ro_params = _make_params(
            [
                _make_tool(
                    "tests.unit.environments.test_database_executable_environment._try_insert_executor",
                    tool_id="try_write",
                    read_only=True,
                )
            ],
            env_kwargs={
                "connection": {"driver": "sqlite", "database": str(db_path)},
                "read_only": True,
            },
        )
        ro_env = DatabaseExecutableEnvironment.from_params(ro_params)
        try:
            result = ro_env.step("try_write", {})
            assert isinstance(result.output, dict)
            assert result.output["status"] == "error"
            msg = result.output["message"].lower()
            assert "read" in msg or "readonly" in msg
        finally:
            ro_env.close()


def test_dialect_guards_sqlite_statement_timeout_warns(caplog):
    """SQLite doesn't support statement_timeout — env warns instead of failing."""
    import logging
    params = _make_params(
        [
            _make_tool(
                "tests.unit.environments.test_database_executable_environment._select_one_executor"
            )
        ],
        env_kwargs={
            "connection": {"driver": "sqlite", "database": ":memory:"},
            "statement_timeout_ms": 5000,
        },
    )
    with caplog.at_level(logging.WARNING):
        env = DatabaseExecutableEnvironment.from_params(params)
        try:
            env.step("t1", {})
        finally:
            env.close()
    assert any(
        "SQLite" in record.message and "statement_timeout" in record.message
        for record in caplog.records
    )


def _executor_returns_updated_state(arguments, db):
    return ToolResult(output={"status": "ok"}, updated_state={"x": 1})


def _executor_lets_integrity_error_escape(arguments, db):
    db.execute(sqlalchemy.text(
        "CREATE TABLE IF NOT EXISTS uq (id INTEGER PRIMARY KEY)"
    ))
    db.execute(sqlalchemy.text("INSERT INTO uq (id) VALUES (1)"))
    db.execute(sqlalchemy.text("INSERT INTO uq (id) VALUES (1)"))  # PK conflict
    return ToolResult(output={"status": "should not reach"})


def _executor_lets_programming_error_escape(arguments, db):
    db.execute(sqlalchemy.text("SELECT * FROM table_that_does_not_exist"))
    return ToolResult(output={"status": "should not reach"})


def test_step_rejects_updated_state_in_result():
    """DB envs hold state in the DB; ToolResult.updated_state is forbidden."""
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_database_executable_environment._executor_returns_updated_state",
            read_only=False,
        )
    ])
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        with pytest.raises(Exception, match="updated_state"):
            env.step("t1", {})
    finally:
        env.close()


def test_step_auto_wraps_integrity_error():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "iw.db"
        params = _make_params(
            [
                _make_tool(
                    "tests.unit.environments.test_database_executable_environment._executor_lets_integrity_error_escape",
                    read_only=False,
                )
            ],
            env_kwargs={
                "connection": {"driver": "sqlite", "database": str(db_path)},
            },
        )
        env = DatabaseExecutableEnvironment.from_params(params)
        try:
            result = env.step("t1", {})
            assert isinstance(result.output, dict)
            assert result.output["status"] == "error"
            assert result.output["error"] == "IntegrityError"
            msg_lower = result.output["message"].lower()
            assert ("unique" in msg_lower) or ("primary key" in msg_lower)
        finally:
            env.close()


def test_step_auto_wraps_programming_error():
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_database_executable_environment._executor_lets_programming_error_escape"
        )
    ])
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        result = env.step("t1", {})
        assert isinstance(result.output, dict)
        assert result.output["status"] == "error"
        # SQLite raises OperationalError, not ProgrammingError, for missing tables.
        assert result.output["error"] in {"OperationalError", "ProgrammingError"}
        assert (
            "table_that_does_not_exist" in result.output["message"]
            or "no such table" in result.output["message"].lower()
        )
    finally:
        env.close()


def test_auto_wrap_skips_output_schema_validation():
    """A tool with strict output_schema still surfaces auto-wrap shape on SQL error."""
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_database_executable_environment._executor_lets_programming_error_escape",
            output_schema={
                "type": "object",
                "properties": {"some_field": {"type": "string"}},
                "required": ["some_field"],
            },
        )
    ])
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        result = env.step("t1", {})
        # The wrap shape doesn't include "some_field"; we should get the wrap,
        # not a schema-validation error.
        assert result.output["status"] == "error"
    finally:
        env.close()


def test_per_tool_timeout_override_on_sqlite_is_no_op_smoke():
    """SQLite can't enforce per-statement timeouts, but the SET path must not crash."""
    params = _make_params([
        _make_tool(
            "tests.unit.environments.test_database_executable_environment._select_one_executor",
            statement_timeout_ms=500,
        )
    ])
    env = DatabaseExecutableEnvironment.from_params(params)
    try:
        result = env.step("t1", {})
        assert result.output == {"rows": [{"one": 1}]}
    finally:
        env.close()
