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

"""Rollback-based SQLite isolation for per-rollout database environments.

The SkyRL pattern: each rollout gets its own connection that opens a
transaction and never commits, so uncommitted writes are visible within the
rollout and discarded on close. The environment owns the transaction;
executors must not call ``commit()``.
"""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path


def materialize_sqlite_snapshot(
    *,
    schema_sql: str,
    seed_sql: str | None = None,
    dest: Path | str | None = None,
) -> Path:
    """Build a snapshot SQLite file from DDL (+ optional seed INSERTs)."""
    path = (
        Path(dest)
        if dest is not None
        else Path(tempfile.gettempdir()) / f"oumi_snapshot_{uuid.uuid4().hex}.sqlite"
    )
    conn = sqlite3.connect(path)
    try:
        conn.executescript(schema_sql)
        if seed_sql:
            conn.executescript(seed_sql)
        conn.commit()
    except BaseException:
        conn.close()
        # Drop the partial temp file we just created so a bad DDL/seed can't leak it.
        if dest is None:
            path.unlink(missing_ok=True)
        raise
    conn.close()
    return path


class RollbackSession:
    """A per-rollout SQLite connection that never commits and rolls back on close.

    Set ``owns_file=True`` when the env built a throwaway per-rollout database
    that should be deleted on teardown (as opposed to a shared snapshot).
    """

    def __init__(self, db_path: Path | str, *, owns_file: bool = False) -> None:
        """Open a per-rollout connection; set owns_file to delete the DB on close.

        Opens with ``isolation_level=None`` and an explicit ``BEGIN`` so the whole
        session is one transaction. ``sqlite3``'s legacy mode only opens an
        implicit transaction before DML, which would let a leading DDL statement
        (e.g. ``CREATE TABLE`` as the first call) run in autocommit and escape the
        rollback; the explicit ``BEGIN`` brings DDL under transaction control too.
        Executors still must not call ``commit()`` — an explicit commit persists
        regardless and there is no way to undo it.
        """
        self._path = Path(db_path)
        self._owns_file = owns_file
        self.connection = sqlite3.connect(self._path, isolation_level=None)
        try:
            self.connection.execute("BEGIN")
        except BaseException:
            self.connection.close()
            if self._owns_file:
                self._path.unlink(missing_ok=True)
            raise

    def close(self) -> None:
        """Roll back any open transaction, close, and delete an owned file."""
        try:
            self.connection.rollback()
        finally:
            self.connection.close()
            if self._owns_file:
                self._path.unlink(missing_ok=True)
