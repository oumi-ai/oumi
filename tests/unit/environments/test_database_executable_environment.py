# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for DatabaseExecutableEnvironment (SQLite-backed)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import sqlalchemy

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
