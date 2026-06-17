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

"""Execution-match reward for SQL tasks, graded on an isolated snapshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.environments.db_isolation import RollbackSession, materialize_sqlite_snapshot


def _run(connection: sqlite3.Connection, sql: str) -> list[tuple] | None:
    try:
        return connection.execute(sql).fetchall()
    except sqlite3.Error:
        return None


@register("sql_execution_match", RegistryType.REWARD_FUNCTION)
def sql_execution_match(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Return 1.0 if candidate SQL's result set matches the gold's, else 0.0.

    ``extra_info`` carries the DB descriptor: either ``db_path`` or
    ``schema_sql`` (+ optional ``seed_sql``). Grading runs on a rollback
    session, so a candidate that writes never mutates the snapshot.
    """
    info = extra_info or {}
    owns = False
    if info.get("db_path"):
        path = Path(info["db_path"])
    else:
        path = materialize_sqlite_snapshot(
            schema_sql=info["schema_sql"], seed_sql=info.get("seed_sql")
        )
        owns = True
    # Grade gold and candidate on separate sessions so each runs against the
    # pristine snapshot — a mutating gold query can't contaminate the candidate.
    gold_session = RollbackSession(path)
    try:
        gold_rows = _run(gold_session.connection, ground_truth)
    finally:
        gold_session.close()
    cand_session = RollbackSession(path, owns_file=owns)
    try:
        cand_rows = _run(cand_session.connection, solution_str)
    finally:
        cand_session.close()
    if gold_rows is None or cand_rows is None:
        return 0.0
    return 1.0 if gold_rows == cand_rows else 0.0
