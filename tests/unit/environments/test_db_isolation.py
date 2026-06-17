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

from __future__ import annotations

import sqlite3

from oumi.environments.db_isolation import RollbackSession, materialize_sqlite_snapshot

_SCHEMA = "CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);"
_SEED = "INSERT INTO t VALUES (1, 'a');"


def test_materialize_builds_a_seeded_snapshot(tmp_path):
    path = materialize_sqlite_snapshot(
        schema_sql=_SCHEMA, seed_sql=_SEED, dest=tmp_path / "seed.sqlite"
    )
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT v FROM t WHERE id = 1").fetchone()[0] == "a"
    conn.close()


def test_rollback_session_discards_uncommitted_writes(tmp_path):
    path = materialize_sqlite_snapshot(
        schema_sql=_SCHEMA, seed_sql=_SEED, dest=tmp_path / "seed.sqlite"
    )
    session = RollbackSession(path)
    # Write without committing; visible on this connection...
    session.connection.execute("UPDATE t SET v = 'mutated' WHERE id = 1")
    assert session.connection.execute("SELECT v FROM t WHERE id = 1").fetchone()[0] == (
        "mutated"
    )
    session.close()  # rolls back + closes
    # ...gone from the snapshot afterwards.
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT v FROM t WHERE id = 1").fetchone()[0] == "a"
    conn.close()


def test_two_sessions_on_one_snapshot_do_not_see_each_others_uncommitted_writes(
    tmp_path,
):
    path = materialize_sqlite_snapshot(
        schema_sql=_SCHEMA, seed_sql=_SEED, dest=tmp_path / "seed.sqlite"
    )
    a = RollbackSession(path)
    b = RollbackSession(path)
    try:
        a.connection.execute("UPDATE t SET v = 'from_a' WHERE id = 1")
        # b never sees a's uncommitted write.
        assert b.connection.execute("SELECT v FROM t WHERE id = 1").fetchone()[0] == "a"
    finally:
        a.close()
        b.close()


def test_leading_ddl_is_rolled_back(tmp_path):
    # DDL as the first statement must still roll back. Legacy sqlite3 only opens
    # an implicit transaction before DML, so without the explicit BEGIN a leading
    # CREATE TABLE would run in autocommit and persist past close().
    path = materialize_sqlite_snapshot(
        schema_sql=_SCHEMA, seed_sql=_SEED, dest=tmp_path / "seed.sqlite"
    )
    session = RollbackSession(path)
    session.connection.execute("CREATE TABLE leaked (x INTEGER)")
    session.close()
    conn = sqlite3.connect(path)
    leaked = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE name = 'leaked'"
    ).fetchone()[0]
    conn.close()
    assert leaked == 0


def test_owned_session_deletes_its_file_on_close(tmp_path):
    path = materialize_sqlite_snapshot(
        schema_sql=_SCHEMA, dest=tmp_path / "owned.sqlite"
    )
    session = RollbackSession(path, owns_file=True)
    assert path.exists()
    session.close()
    assert not path.exists()
