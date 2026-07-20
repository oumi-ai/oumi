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
from pathlib import Path
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.environments.database_session import materialize_sqlite_snapshot

_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """Extract the model's final SQL: last ```sql fence, else last non-empty line."""
    fences = _SQL_FENCE.findall(text or "")
    if fences:
        return fences[-1].strip()
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _run(db_path: Path, sql: str, ordered: bool) -> list:
    """Execute `sql` read-only and return its rows, sorted unless `ordered`."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return rows if ordered else sorted(str(row) for row in rows)


def _run_or_none(db_path: Path, sql: str, ordered: bool) -> list | None:
    """`_run`, but returns None instead of raising on invalid SQL."""
    try:
        return _run(db_path, sql, ordered)
    except Exception:
        return None


def _db_spec(extra_info: dict[str, Any]) -> dict[str, Any]:
    """The run_sql tool's per-rollout DB spec (db_path, or schema_sql[/seed_sql])."""
    tools_kwargs = extra_info.get("tools_kwargs") or {}
    run_sql = tools_kwargs.get("run_sql") or {}
    return run_sql.get("create_kwargs") or {}


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
    shared_db = db_spec.get("db_path")
    if shared_db:
        db_path, owns_file = Path(shared_db), False
    else:
        db_path = materialize_sqlite_snapshot(
            schema_sql=db_spec["schema_sql"], seed_sql=db_spec.get("seed_sql")
        )
        owns_file = True
    try:
        ordered = "order by" in str(ground_truth).lower()
        pred_rows = _run_or_none(db_path, pred_sql, ordered)
        gold_rows = _run_or_none(db_path, ground_truth, ordered)
        if pred_rows is None or gold_rows is None:
            return 0.0  # invalid SQL (pred or gold) scores 0
        return 1.0 if pred_rows == gold_rows else 0.0
    finally:
        if owns_file:
            db_path.unlink(missing_ok=True)
