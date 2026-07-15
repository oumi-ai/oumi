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

"""Behavior + rollback-isolation tests for DatabaseExecutableEnvironment."""

from __future__ import annotations

import sqlite3

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)
from oumi.environments.database_session import materialize_sqlite_snapshot

_SCHEMA = "CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, meds TEXT);"
_SEED = "INSERT INTO patients VALUES (1, 'Bob', 'aspirin');"


# Minimal executors, kept local so this suite doesn't depend on the EHR example.
def lookup_patient(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    row = context.execute(
        "SELECT name, meds FROM patients WHERE id = ?", (arguments["pat_id"],)
    ).fetchone()
    if row is None:
        return ToolResult(output={"error": "not found"})
    return ToolResult(output={"name": row[0], "meds": row[1]})


def update_meds(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    cursor = context.execute(
        "UPDATE patients SET meds = ? WHERE id = ?",
        (arguments["medication"], arguments["pat_id"]),
    )
    return ToolResult(output={"updated_rows": cursor.rowcount})


def write_meds_read_only(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    """A read_only-declared tool that (illegally) attempts a write."""
    context.execute("UPDATE patients SET meds = 'sneaky' WHERE id = 1")
    return ToolResult(output={})


def write_then_raise(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    """Writes, then fails, so the call's savepoint must roll the write back."""
    context.execute("UPDATE patients SET meds = 'partial' WHERE id = 1")
    raise RuntimeError("boom")


def _params(tools):
    return EnvironmentParams(
        id="ehr",
        name="ehr",
        description="EHR test env",
        env_type="database",
        tools=tools,
        env_kwargs={"schema_sql": _SCHEMA, "seed_sql": _SEED},
    )


def _lookup_tool():
    return {
        "id": "lookup",
        "name": "lookup",
        "description": "look up a patient",
        "parameters": {
            "type": "object",
            "properties": {"pat_id": {"type": "integer"}},
            "required": ["pat_id"],
        },
        "executor": f"{__name__}.lookup_patient",
        "read_only": True,
    }


def _update_tool():
    return {
        "id": "update",
        "name": "update",
        "description": "update meds",
        "parameters": {
            "type": "object",
            "properties": {
                "pat_id": {"type": "integer"},
                "medication": {"type": "string"},
            },
            "required": ["pat_id", "medication"],
        },
        "executor": f"{__name__}.update_meds",
        "read_only": False,
    }


def test_requires_isolation_is_true():
    env = DatabaseExecutableEnvironment.from_params(_params([_lookup_tool()]))
    try:
        assert env.requires_isolation() is True
    finally:
        env.close()


def test_executes_read_tool_against_isolated_db():
    env = DatabaseExecutableEnvironment.from_params(_params([_lookup_tool()]))
    try:
        [result] = env.step([("lookup", {"pat_id": 1})])
        assert result.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        env.close()


def test_uncommitted_write_visible_within_one_episode():
    env = DatabaseExecutableEnvironment.from_params(
        _params([_lookup_tool(), _update_tool()])
    )
    try:
        env.step([("update", {"pat_id": 1, "medication": "statin"})])
        [seen] = env.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "statin"}
    finally:
        env.close()


def test_close_rolls_back_so_a_fresh_env_starts_clean():
    params = _params([_lookup_tool(), _update_tool()])
    env = DatabaseExecutableEnvironment.from_params(params)
    env.step([("update", {"pat_id": 1, "medication": "mutated"})])
    env.close()  # rolls back; the inline-built DB is also discarded
    fresh = DatabaseExecutableEnvironment.from_params(params)
    try:
        [seen] = fresh.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        fresh.close()


def test_writes_do_not_leak_across_concurrent_rollouts():
    params = _params([_lookup_tool(), _update_tool()])
    # N rollouts of the same task; each builds its own inline DB.
    envs = [DatabaseExecutableEnvironment.from_params(params) for _ in range(4)]
    try:
        for i, env in enumerate(envs):
            env.step([("update", {"pat_id": 1, "medication": f"drug_{i}"})])
        for i, env in enumerate(envs):
            [seen] = env.step([("lookup", {"pat_id": 1})])
            assert seen.output == {"name": "Bob", "meds": f"drug_{i}"}
    finally:
        for env in envs:
            env.close()


def test_shared_snapshot_is_never_mutated():
    snapshot = materialize_sqlite_snapshot(schema_sql=_SCHEMA, seed_sql=_SEED)
    params = EnvironmentParams(
        id="ehr",
        name="ehr",
        description="d",
        env_type="database",
        tools=[_lookup_tool(), _update_tool()],
        env_kwargs={"db_path": str(snapshot)},
    )
    env = DatabaseExecutableEnvironment.from_params(params)
    env.step([("update", {"pat_id": 1, "medication": "mutated"})])
    env.close()  # rollback
    # The shared snapshot file is untouched.
    fresh = DatabaseExecutableEnvironment.from_params(params)
    try:
        [seen] = fresh.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        fresh.close()


def _readonly_write_tool():
    return {
        "id": "ro_write",
        "name": "ro_write",
        "description": "a read-only tool that tries to write",
        "parameters": {"type": "object"},
        "executor": f"{__name__}.write_meds_read_only",
        "read_only": True,
    }


def _boom_write_tool():
    return {
        "id": "boom",
        "name": "boom",
        "description": "writes then raises",
        "parameters": {"type": "object"},
        "executor": f"{__name__}.write_then_raise",
        "read_only": False,
    }


def test_read_only_tool_cannot_write_then_pragma_resets():
    env = DatabaseExecutableEnvironment.from_params(
        _params([_readonly_write_tool(), _update_tool(), _lookup_tool()])
    )
    try:
        # PRAGMA query_only=ON blocks the read-only tool's illegal write.
        with pytest.raises(sqlite3.OperationalError):
            env.step([("ro_write", {})])
        # query_only was reset OFF, so a normal write still succeeds afterwards.
        env.step([("update", {"pat_id": 1, "medication": "statin"})])
        [seen] = env.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "statin"}
    finally:
        env.close()


def test_batch_rolls_back_all_writes_when_one_call_fails():
    env = DatabaseExecutableEnvironment.from_params(
        _params([_update_tool(), _boom_write_tool(), _lookup_tool()])
    )
    try:
        with pytest.raises(RuntimeError):
            env.step([("update", {"pat_id": 1, "medication": "first"}), ("boom", {})])
        # The batch savepoint discarded the earlier update too.
        [seen] = env.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        env.close()


def test_failing_call_discards_its_partial_write():
    env = DatabaseExecutableEnvironment.from_params(
        _params([_boom_write_tool(), _lookup_tool()])
    )
    try:
        with pytest.raises(RuntimeError):
            env.step([("boom", {})])
        [seen] = env.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        env.close()


@pytest.mark.parametrize(
    "env_kwargs",
    [
        {},  # neither db_path nor schema_sql
        {"db_path": "nonexistent.sqlite", "schema_sql": _SCHEMA},  # both
        {"seed_sql": _SEED},  # seed_sql without schema_sql
    ],
)
def test_invalid_db_source_config_raises(env_kwargs):
    params = EnvironmentParams(
        id="ehr",
        name="ehr",
        description="d",
        env_type="database",
        tools=[_lookup_tool()],
        env_kwargs=env_kwargs,
    )
    with pytest.raises(ValueError):
        DatabaseExecutableEnvironment.from_params(params)


def test_construction_failure_deletes_owned_snapshot(monkeypatch):
    import oumi.environments.database_executable_environment as mod

    created: list = []
    real = mod.materialize_sqlite_snapshot

    def _spy(**kwargs):
        path = real(**kwargs)
        created.append(path)
        return path

    monkeypatch.setattr(mod, "materialize_sqlite_snapshot", _spy)
    bad_tool = dict(_lookup_tool(), executor="oumi.nonexistent.module.fn")
    with pytest.raises(ValueError):
        DatabaseExecutableEnvironment.from_params(_params([bad_tool]))
    # The owned snapshot built before the failing constructor was cleaned up.
    assert created and not created[0].exists()


def run_raw_sql(arguments: dict, context: sqlite3.Connection) -> ToolResult:
    try:
        context.execute(arguments["sql"])
    except sqlite3.Error as e:
        return ToolResult(output={"error": str(e)})
    return ToolResult(output={"ok": True})


def _raw_tool():
    return {
        "id": "raw",
        "name": "raw",
        "description": "run raw sql",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
        "executor": f"{__name__}.run_raw_sql",
        "read_only": True,
    }


def test_model_sql_cannot_break_framework_savepoints():
    # Model SQL runs on the env's own connection. A query issuing
    # `RELEASE SAVEPOINT oumi_tool_call` must not free the framework's own
    # savepoint and crash step() at cleanup: savepoint names are randomized, so
    # the attack just returns a graceful error and the env stays usable.
    env = DatabaseExecutableEnvironment.from_params(
        _params([_raw_tool(), _lookup_tool()])
    )
    try:
        [attack] = env.step([("raw", {"sql": "RELEASE SAVEPOINT oumi_tool_call"})])
        assert "error" in attack.output  # graceful, not a crash
        [seen] = env.step([("lookup", {"pat_id": 1})])
        assert seen.output == {"name": "Bob", "meds": "aspirin"}
    finally:
        env.close()
