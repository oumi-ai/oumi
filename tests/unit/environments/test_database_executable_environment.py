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

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)
from oumi.environments.db_isolation import materialize_sqlite_snapshot

_SCHEMA = "CREATE TABLE patients (id INTEGER PRIMARY KEY, name TEXT, meds TEXT);"
_SEED = "INSERT INTO patients VALUES (1, 'Bob', 'aspirin');"


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
        "executor": "oumi.environments.examples.ehr.lookup_patient",
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
        "executor": "oumi.environments.examples.ehr.update_meds",
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
        assert seen.output["meds"] == "aspirin"
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
            assert seen.output["meds"] == f"drug_{i}"
    finally:
        for env in envs:
            env.close()


def test_shared_snapshot_is_never_mutated(tmp_path):
    snapshot = materialize_sqlite_snapshot(
        schema_sql=_SCHEMA, seed_sql=_SEED, dest=tmp_path / "shared.sqlite"
    )
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
        assert seen.output["meds"] == "aspirin"
    finally:
        fresh.close()
