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

"""EHR example: three SQL tool executors for the database environment.

Each executor takes only its declared tool parameters and reaches the episode's
SQLite connection via ``current_connection()`` — the environment binds it per
call. They return a plain dict (the env wraps it in a ToolResult) and must NOT
commit: the environment owns the transaction and rolls back on close. The schema
and seed data live in the example configs under ``configs/examples/``.
"""

from __future__ import annotations

from typing import Any

from oumi.environments.database_executable_environment import current_connection


def list_patients() -> dict[str, Any]:
    """List every patient's id and name."""
    rows = (
        current_connection()
        .execute("SELECT id, name FROM patients ORDER BY id")
        .fetchall()
    )
    return {"patients": [{"id": r[0], "name": r[1]} for r in rows]}


def lookup_patient(pat_id: int) -> dict[str, Any]:
    """Return one patient's name and meds by id, or an error if absent."""
    row = (
        current_connection()
        .execute("SELECT name, meds FROM patients WHERE id = ?", (pat_id,))
        .fetchone()
    )
    if row is None:
        return {"error": "not found"}
    return {"name": row[0], "meds": row[1]}


def update_meds(pat_id: int, medication: str) -> dict[str, Any]:
    """Set a patient's medication (uncommitted; rolled back at episode end)."""
    cur = current_connection().execute(
        "UPDATE patients SET meds = ? WHERE id = ?", (medication, pat_id)
    )
    return {"updated_rows": cur.rowcount}
