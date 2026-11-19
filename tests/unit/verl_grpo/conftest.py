import sqlite3
from collections.abc import Callable

import pytest

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult


def run_sql(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    """Runs the query and also lists the arguments the executor received."""
    cursor = context.execute(arguments["query"])
    return ToolResult(
        output={
            "rows": [list(row) for row in cursor.fetchall()],
            "args": sorted(arguments),
        }
    )


def _sql_env(
    env_id: str, *tool_ids: str, executor: Callable = run_sql
) -> EnvironmentParams:
    tool = {
        "description": "run sql",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        "executor": f"{executor.__module__}.{executor.__name__}",
        "read_only": True,
    }
    return EnvironmentParams(
        id=env_id,
        env_type="database",
        env_kwargs={
            "schema_sql": "CREATE TABLE t(x INTEGER);",
            "seed_sql": "INSERT INTO t VALUES (7);",
        },
        tools=[{"id": tool_id, "name": tool_id, **tool} for tool_id in tool_ids],
    )


@pytest.fixture
def sql_env() -> Callable[..., EnvironmentParams]:
    """Builds a database environment whose table ``t`` holds the one row 7."""
    return _sql_env


@pytest.fixture
def env_config(tmp_path) -> Callable[..., str]:
    """Writes an environment config; by default one ``db`` env with ``run_sql``."""

    def write(*environments: EnvironmentParams, name: str = "env.yaml") -> str:
        path = tmp_path / name
        default = [_sql_env("db", "run_sql")]
        EnvironmentConfig(environments=list(environments) or default).to_yaml(path)
        return str(path)

    return write
