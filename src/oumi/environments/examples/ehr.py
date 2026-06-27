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

"""EHR example: schema, seed data, and three SQL tool executors.

Executors take (*, arguments, db) where ``db`` is a sqlite3.Connection handed
in by DatabaseExecutableEnvironment, and return a ToolResult. They must NOT
commit: the environment owns the transaction and rolls back on close.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from oumi.core.types.tool_call import ToolResult

EHR_SCHEMA = (
    "CREATE TABLE patients ( id INTEGER PRIMARY KEY, name TEXT NOT NULL, meds TEXT);"
)
EHR_SEED = (
    "INSERT INTO patients (id, name, meds) VALUES"
    " (1, 'Bob', 'aspirin'), (2, 'Alice', 'ibuprofen'), (3, 'Carol', NULL);"
)


def list_patients(*, arguments: dict[str, Any], db: sqlite3.Connection) -> ToolResult:
    """List every patient's id and name."""
    rows = db.execute("SELECT id, name FROM patients ORDER BY id").fetchall()
    return ToolResult(output={"patients": [{"id": r[0], "name": r[1]} for r in rows]})


def lookup_patient(*, arguments: dict[str, Any], db: sqlite3.Connection) -> ToolResult:
    """Return one patient's name and meds by id, or an error if absent."""
    row = db.execute(
        "SELECT name, meds FROM patients WHERE id = ?", (arguments["pat_id"],)
    ).fetchone()
    if row is None:
        return ToolResult(output={"error": "not found"})
    return ToolResult(output={"name": row[0], "meds": row[1]})


def update_meds(*, arguments: dict[str, Any], db: sqlite3.Connection) -> ToolResult:
    """Set a patient's medication (uncommitted; rolled back at episode end)."""
    # No commit: the environment rolls back at episode end. The write is
    # visible to later calls on this same connection within the episode.
    cur = db.execute(
        "UPDATE patients SET meds = ? WHERE id = ?",
        (arguments["medication"], arguments["pat_id"]),
    )
    return ToolResult(output={"updated_rows": cur.rowcount})
