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

"""Tool executors for the EHR database synthesis example.

``DatabaseExecutableEnvironment`` passes each executor a live SQLite connection
inside a savepoint rather than a copied state document, so writes are ordinary
SQL. The environment owns the transaction: executors must never commit, and
every write is rolled back when the episode ends.

For tools over small state that fits in a JSON snapshot, use
``SyntheticEnvironment`` instead; see ``../ehr_stateful_agent/executors.py``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from oumi.core.registry import register_tool_executor
from oumi.core.types.tool_call import ToolResult


@register_tool_executor("ehr_db.list_patients")
def list_patients(arguments: dict[str, Any], context: sqlite3.Connection) -> ToolResult:
    """List every patient's id and name."""
    rows = context.execute("SELECT id, name FROM patients ORDER BY id").fetchall()
    return ToolResult(
        output={"patients": [{"id": row[0], "name": row[1]} for row in rows]}
    )


@register_tool_executor("ehr_db.lookup_patient")
def lookup_patient(
    arguments: dict[str, Any], context: sqlite3.Connection
) -> ToolResult:
    """Return one patient's name and meds by id, or an error if absent."""
    row = context.execute(
        "SELECT name, meds FROM patients WHERE id = ?", (arguments["pat_id"],)
    ).fetchone()
    if row is None:
        return ToolResult(output={"error": "not found"})
    return ToolResult(output={"name": row[0], "meds": row[1]})


@register_tool_executor("ehr_db.update_meds")
def update_meds(arguments: dict[str, Any], context: sqlite3.Connection) -> ToolResult:
    """Set a patient's medication (uncommitted; rolled back at episode end)."""
    cursor = context.execute(
        "UPDATE patients SET meds = ? WHERE id = ?",
        (arguments["medication"], arguments["pat_id"]),
    )
    return ToolResult(output={"updated_rows": cursor.rowcount})
