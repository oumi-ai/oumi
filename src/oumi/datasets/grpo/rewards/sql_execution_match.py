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

import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironmentKwargs,
)
from oumi.environments.database_session import (
    materialize_sqlite_snapshot,
    query_work_budget,
)

_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SQL_START = re.compile(r"(?im)^\s*(?:select|with)\b")
# A chat/tool tag, as opposed to a `<` comparison: `<tool_call>`, `<|im_start|>`.
_CHAT_TAG = re.compile(r"<[/|a-zA-Z]")
# One alternation keeps the scan left-to-right, so a quoted `--` stays a literal.
_SQL_NOISE = re.compile(
    r"--[^\n\r]*|/\*.*?\*/"
    r"|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`[^`]*`|\[[^\]]*\]",
    re.DOTALL,
)
_SQL_INNER_PARENS = re.compile(r"\([^()]*\)")
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_READ_ACTIONS = {
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_RECURSIVE,
    sqlite3.SQLITE_SELECT,
}
_RowResult = list[tuple[Any, ...]] | Counter[tuple[Any, ...]]


def _extract_sql(text: str) -> str:
    """Extract the model's final fenced or unfenced SQL statement."""
    fences = _SQL_FENCE.findall(text)
    if fences:
        return fences[-1].strip()
    starts = [match.start() for match in _SQL_START.finditer(text)]
    if not starts:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
    # The last line-initial SELECT can be a UNION arm, a subquery or a CTE body, so
    # walk back over the starts that only continue it. A `;`, a blank line or a chat
    # tag ends a statement; anything else between two starts joins them.
    start = starts[-1]
    for earlier in reversed(starts[:-1]):
        between = text[earlier:start]
        if ";" in between or "\n\n" in between or _CHAT_TAG.search(between):
            break
        start = earlier
    return text[start:].strip()


def _read_only_authorizer(
    action: int,
    _arg1: str | None,
    _arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY


def _run(conn: sqlite3.Connection, sql: str, ordered: bool) -> _RowResult:
    """Return a read query's rows in order, or as an order-insensitive multiset."""
    rows = conn.execute(sql).fetchall()
    return rows if ordered else Counter(rows)


def _resolve_db(extra_info: dict[str, Any]) -> tuple[Path, bool]:
    """Resolve run_sql's per-rollout DB spec to (db path, whether we own the file)."""
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    values = (tools_kwargs.get("run_sql") or {}).get("create_kwargs") or {}
    if not isinstance(values, dict):
        raise ValueError("run_sql create_kwargs must be a mapping.")
    # Same spec the database environment takes, validated the same way.
    kwargs = DatabaseExecutableEnvironmentKwargs(**values)
    kwargs.finalize_and_validate()
    if kwargs.db_path:
        return Path(kwargs.db_path), False
    assert kwargs.schema_sql is not None
    snapshot = materialize_sqlite_snapshot(
        schema_sql=kwargs.schema_sql, seed_sql=kwargs.seed_sql
    )
    return snapshot, True


def _has_top_level_order_by(sql: str) -> bool:
    """Return whether an ORDER BY survives blanking quotes and nested groups.

    Strip-then-match, not a real parser; use sqlglot for true clause awareness.
    """
    sql = _SQL_NOISE.sub(" ", sql)
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
    """Score the model's final SQL by running it against the gold query's database.

    Row order matters only when the gold query has a top-level ``ORDER BY``.

    Args:
        data_source: The data source. Unused.
        solution_str: The response from the LLM; the final SQL is extracted from it.
        ground_truth: The gold SQL.
        extra_info: Extra information about the sample. The DB spec is read from
            ``["tools_kwargs"]["run_sql"]["create_kwargs"]``: a "db_path" to a
            pre-staged SQLite file, or "schema_sql" (+ optional "seed_sql").

    Returns:
        1.0 if the predicted SQL's result set matches the gold SQL's, else 0.0.

    Raises:
        ValueError: The DB spec is missing or malformed.
        sqlite3.Error: The gold SQL does not execute. A row whose ground truth
            cannot be scored aborts the run rather than scoring every rollout 0.
    """
    pred_sql = _extract_sql(solution_str)
    if not pred_sql:
        return 0.0

    db_path, owns_file = _resolve_db(extra_info)
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
            # Benchmark DBs carry non-UTF-8 text; a decode error would kill the run.
            # surrogateescape, not replace: replace maps distinct bytes to one U+FFFD,
            # which scores a wrong row as a match.
            conn.text_factory = lambda b: b.decode("utf-8", "surrogateescape")
            conn.set_authorizer(_read_only_authorizer)
            ordered = _has_top_level_order_by(ground_truth)
            gold_rows = _run(conn, ground_truth, ordered)
            try:
                # The prediction is untrusted; a runaway query must not stall the
                # trainer. Over-budget raises sqlite3.Error, so it scores 0.0.
                with query_work_budget(conn):
                    pred_rows = _run(conn, pred_sql, ordered)
            except sqlite3.Error:
                return 0.0
            return 1.0 if pred_rows == gold_rows else 0.0
    finally:
        if owns_file:
            db_path.unlink(missing_ok=True)
