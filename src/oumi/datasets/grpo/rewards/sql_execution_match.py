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

"""Execution-match (EX) reward for NL2SQL: compare predicted vs gold result sets."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.environments.database_session import materialize_sqlite_snapshot

_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"(?im)^\s*select\b")
_SQL_COMMENT = re.compile(r"--[^\n\r]*|/\*.*?\*/", re.DOTALL)
_SQL_QUOTED = re.compile(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^\]]*\]")
_SQL_INNER_PARENS = re.compile(r"\([^()]*\)")
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_READ_ACTIONS = {
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_RECURSIVE,
    sqlite3.SQLITE_SELECT,
}


@dataclass(frozen=True)
class _PathDatabaseSpec:
    db_path: Path


@dataclass(frozen=True)
class _InlineDatabaseSpec:
    schema_sql: str
    seed_sql: str | None


_DatabaseSpec = _PathDatabaseSpec | _InlineDatabaseSpec
_RowResult = list[tuple[Any, ...]] | list[str]


def _extract_sql(text: str) -> str:
    """Extract the model's final fenced or unfenced SQL query."""
    fences = _SQL_FENCE.findall(text or "")
    if fences:
        return fences[-1].strip()
    starts = list(_SQL_START.finditer(text or ""))
    if starts:
        return text[starts[-1].start() :].strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _read_only_authorizer(
    action: int,
    _arg1: str | None,
    _arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def _run(conn: sqlite3.Connection, sql: str, ordered: bool) -> _RowResult:
    """Return raw ordered rows or sorted row representations from a read query."""
    rows = conn.execute(sql).fetchall()
    return rows if ordered else sorted(str(row) for row in rows)


def _db_spec(extra_info: dict[str, Any]) -> _DatabaseSpec:
    """The run_sql tool's per-rollout DB spec (db_path, or schema_sql[/seed_sql])."""
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    run_sql = tools_kwargs.get("run_sql") or {}
    values = run_sql.get("create_kwargs") or {}
    if not isinstance(values, dict):
        raise ValueError("run_sql create_kwargs must be a mapping.")

    db_path = values.get("db_path")
    schema_sql = values.get("schema_sql")
    seed_sql = values.get("seed_sql")
    if db_path is not None and not isinstance(db_path, (str, Path)):
        raise ValueError("db_path must be a path string.")
    if schema_sql is not None and not isinstance(schema_sql, str):
        raise ValueError("schema_sql must be a string.")
    if seed_sql is not None and not isinstance(seed_sql, str):
        raise ValueError("seed_sql must be a string.")
    if seed_sql is not None and not schema_sql:
        raise ValueError("seed_sql requires schema_sql.")
    if db_path and schema_sql:
        raise ValueError("Provide exactly one of db_path or schema_sql.")
    if db_path:
        return _PathDatabaseSpec(Path(db_path))
    if not schema_sql:
        raise ValueError("Provide exactly one of db_path or schema_sql.")
    return _InlineDatabaseSpec(schema_sql, seed_sql)


def _has_top_level_order_by(sql: str) -> bool:
    """Return whether the outer query contains an ORDER BY clause.

    ponytail: strip-then-match, not a real parser. Blank out comments, quoted
    text and every parenthesized group, so whatever ORDER BY survives is the
    outer query's. Use sqlglot if this ever needs true clause awareness.
    """
    sql = _SQL_QUOTED.sub(" ", _SQL_COMMENT.sub(" ", sql))
    while _SQL_INNER_PARENS.search(sql):
        sql = _SQL_INNER_PARENS.sub(" ", sql)
    return _ORDER_BY.search(sql) is not None


@register("sql_execution_match", RegistryType.REWARD_FUNCTION)
def sql_execution_match(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> float:
    """1.0 if the predicted SQL's result set matches the gold SQL's, else 0.0.

    The DB comes from ``extra_info["tools_kwargs"]["run_sql"]["create_kwargs"]``:
    a "db_path" to a pre-staged SQLite file, or "schema_sql" (+ optional "seed_sql").
    """
    pred_sql = _extract_sql(solution_str)
    if not pred_sql:
        return 0.0

    db_spec = _db_spec(extra_info)
    if isinstance(db_spec, _PathDatabaseSpec):
        db_path, owns_file = db_spec.db_path, False
    else:
        db_path = materialize_sqlite_snapshot(
            schema_sql=db_spec.schema_sql, seed_sql=db_spec.seed_sql
        )
        owns_file = True
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            conn.set_authorizer(_read_only_authorizer)
            ordered = _has_top_level_order_by(ground_truth)
            gold_rows = _run(conn, ground_truth, ordered)
            try:
                pred_rows = _run(conn, pred_sql, ordered)
            except sqlite3.Error:
                return 0.0
            return 1.0 if pred_rows == gold_rows else 0.0
        finally:
            conn.close()
    finally:
        if owns_file:
            db_path.unlink(missing_ok=True)
