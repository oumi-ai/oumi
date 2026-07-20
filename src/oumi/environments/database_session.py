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

"""Rollback-based SQLite isolation for per-rollout database environments."""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from collections.abc import Callable
from contextlib import closing
from pathlib import Path


def materialize_sqlite_snapshot(
    *,
    schema_sql: str,
    seed_sql: str | None = None,
) -> Path:
    """Build a snapshot SQLite file from DDL (+ optional seed INSERTs)."""
    path = Path(tempfile.gettempdir()) / f"oumi_snapshot_{uuid.uuid4().hex}.sqlite"
    try:
        with closing(sqlite3.connect(path)) as connection:
            _enable_foreign_keys(connection)
            connection.executescript(schema_sql)
            if seed_sql:
                connection.executescript(seed_sql)
            connection.commit()
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _enable_foreign_keys(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.execute("PRAGMA foreign_keys").fetchone() != (1,):
        raise RuntimeError("Failed to enable SQLite foreign-key enforcement.")


def _deny_transaction_control(
    action_code: int,
    _arg1: str | None,
    _arg2: str | None,
    _database_name: str | None,
    _trigger_name: str | None,
) -> int:
    if action_code == sqlite3.SQLITE_TRANSACTION:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class DatabaseSession:
    """A per-rollout SQLite connection that never commits and rolls back on close.

    Set ``owns_file=True`` when the env built a throwaway per-rollout database
    that should be deleted on teardown (as opposed to a shared snapshot).
    """

    def __init__(self, db_path: Path | str, *, owns_file: bool = False) -> None:
        """Open a per-rollout connection over ``db_path``."""
        self._path = Path(db_path)
        self._owns_file = owns_file
        self._closed = False
        try:
            self.connection = sqlite3.connect(self._path, isolation_level=None)
        except BaseException:
            if self._owns_file:
                try:
                    self._path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        transaction_started = False
        try:
            _enable_foreign_keys(self.connection)
            self.connection.execute("BEGIN")
            transaction_started = True
            self.connection.set_authorizer(_deny_transaction_control)
        except BaseException:
            self._cleanup(rollback=transaction_started, suppress_errors=True)
            raise

    def _cleanup(self, *, rollback: bool, suppress_errors: bool = False) -> None:
        operations: list[Callable[[], object]] = []
        if rollback:
            operations.extend(
                (
                    lambda: self.connection.set_authorizer(None),
                    self.connection.rollback,
                )
            )
        operations.append(self.connection.close)
        if self._owns_file:
            operations.append(lambda: self._path.unlink(missing_ok=True))

        first_error: BaseException | None = None
        for operation in operations:
            try:
                operation()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None and not suppress_errors:
            raise first_error

    def close(self) -> None:
        """Roll back any open transaction, close, and delete an owned file.

        Idempotent: a router may close the same session more than once (build-time
        teardown plus an explicit ``close()``), so a second call is a no-op.
        """
        if self._closed:
            return
        self._closed = True
        self._cleanup(rollback=True)
