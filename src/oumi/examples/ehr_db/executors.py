# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""EHR Database tool executors.

Each executor takes a SQLAlchemy connection (autocommit mode — every
``conn.execute(...)`` is its own transaction). Executors that handle known
constraint violations (duplicate diagnosis, allergy conflict, etc.) catch
the matching ``IntegrityError`` and return a structured ``{"status": "error"}``
``ToolResult`` so the agent can self-correct. Unhandled SQL errors propagate
to the env, which auto-wraps them into a generic structured error.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy
import sqlalchemy.exc
from sqlalchemy.engine import Connection

from oumi.core.types.tool_call import ToolResult


def _patient_exists(conn: Connection, patient_id: str) -> bool:
    row = conn.execute(
        sqlalchemy.text("SELECT 1 FROM patients WHERE patient_id = :pid"),
        {"pid": patient_id},
    ).first()
    return row is not None


def list_patients(arguments: dict[str, Any], db: Connection) -> ToolResult:
    """Return patient summaries (read-only)."""
    rows = db.execute(
        sqlalchemy.text(
            "SELECT patient_id, name, dob, status FROM patients ORDER BY name"
        )
    ).mappings().all()
    return ToolResult(output={"patients": [dict(r) for r in rows]})


def get_patient(arguments: dict[str, Any], db: Connection) -> ToolResult:
    """Fetch the full record for a patient_id (read-only)."""
    patient_id = arguments["patient_id"]
    base = db.execute(
        sqlalchemy.text(
            "SELECT patient_id, name, dob, status FROM patients "
            "WHERE patient_id = :pid"
        ),
        {"pid": patient_id},
    ).mappings().first()
    if base is None:
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )

    allergies = db.execute(
        sqlalchemy.text("SELECT substance FROM allergies WHERE patient_id = :pid"),
        {"pid": patient_id},
    ).scalars().all()
    medications = db.execute(
        sqlalchemy.text(
            "SELECT name, dose FROM medications WHERE patient_id = :pid"
        ),
        {"pid": patient_id},
    ).mappings().all()
    diagnoses = db.execute(
        sqlalchemy.text(
            "SELECT code, description, date FROM diagnoses WHERE patient_id = :pid"
        ),
        {"pid": patient_id},
    ).mappings().all()
    vitals_history = db.execute(
        sqlalchemy.text(
            "SELECT timestamp, bp, hr, temp_f FROM vitals "
            "WHERE patient_id = :pid ORDER BY timestamp"
        ),
        {"pid": patient_id},
    ).mappings().all()

    return ToolResult(
        output={
            "status": "ok",
            "patient": {
                **dict(base),
                "allergies": list(allergies),
                "medications": [dict(m) for m in medications],
                "diagnoses": [dict(d) for d in diagnoses],
                "vitals_history": [dict(v) for v in vitals_history],
            },
        }
    )


def record_vitals(arguments: dict[str, Any], db: Connection) -> ToolResult:
    """Append a vitals reading."""
    patient_id = arguments["patient_id"]
    if not _patient_exists(db, patient_id):
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    entry = {
        "patient_id": patient_id,
        "timestamp": arguments["timestamp"],
        "bp": arguments["bp"],
        "hr": arguments["hr"],
        "temp_f": arguments["temp_f"],
    }
    db.execute(
        sqlalchemy.text(
            "INSERT INTO vitals (patient_id, timestamp, bp, hr, temp_f) "
            "VALUES (:patient_id, :timestamp, :bp, :hr, :temp_f)"
        ),
        entry,
    )
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "vitals_recorded": {
                k: entry[k] for k in ("timestamp", "bp", "hr", "temp_f")
            },
        }
    )


def add_diagnosis(arguments: dict[str, Any], db: Connection) -> ToolResult:
    """Append an ICD-10 diagnosis. Refuses duplicate codes."""
    patient_id = arguments["patient_id"]
    if not _patient_exists(db, patient_id):
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    new_diagnosis = {
        "patient_id": patient_id,
        "code": arguments["code"],
        "description": arguments["description"],
        "date": arguments["date"],
    }
    try:
        db.execute(
            sqlalchemy.text(
                "INSERT INTO diagnoses (patient_id, code, description, date) "
                "VALUES (:patient_id, :code, :description, :date)"
            ),
            new_diagnosis,
        )
    except sqlalchemy.exc.IntegrityError:
        return ToolResult(
            output={
                "status": "error",
                "error": "duplicate_diagnosis",
                "patient_id": patient_id,
                "code": new_diagnosis["code"],
            }
        )
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "diagnosis_added": {
                k: new_diagnosis[k] for k in ("code", "description", "date")
            },
        }
    )


def prescribe_medication(arguments: dict[str, Any], db: Connection) -> ToolResult:
    """Add a medication. Refuses duplicates and allergy conflicts."""
    patient_id = arguments["patient_id"]
    if not _patient_exists(db, patient_id):
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    name = arguments["name"]
    dose = arguments["dose"]

    # Allergy conflict check (case-insensitive match on the substance name).
    allergy_hit = db.execute(
        sqlalchemy.text(
            "SELECT 1 FROM allergies WHERE patient_id = :pid "
            "AND LOWER(substance) = LOWER(:name)"
        ),
        {"pid": patient_id, "name": name},
    ).first()
    if allergy_hit is not None:
        return ToolResult(
            output={
                "status": "error",
                "error": "allergy_conflict",
                "patient_id": patient_id,
                "medication": name,
            }
        )

    try:
        db.execute(
            sqlalchemy.text(
                "INSERT INTO medications (patient_id, name, dose) "
                "VALUES (:patient_id, :name, :dose)"
            ),
            {"patient_id": patient_id, "name": name, "dose": dose},
        )
    except sqlalchemy.exc.IntegrityError:
        return ToolResult(
            output={
                "status": "error",
                "error": "already_prescribed",
                "patient_id": patient_id,
                "medication": name,
            }
        )
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "medication_added": {"name": name, "dose": dose},
        }
    )


def update_allergies(arguments: dict[str, Any], db: Connection) -> ToolResult:
    """Replace a patient's allergy list with the supplied list."""
    patient_id = arguments["patient_id"]
    if not _patient_exists(db, patient_id):
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    new_allergies = list(arguments["allergies"])
    db.execute(
        sqlalchemy.text("DELETE FROM allergies WHERE patient_id = :pid"),
        {"pid": patient_id},
    )
    for substance in new_allergies:
        db.execute(
            sqlalchemy.text(
                "INSERT INTO allergies (patient_id, substance) "
                "VALUES (:pid, :substance)"
            ),
            {"pid": patient_id, "substance": substance},
        )
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "allergies": new_allergies,
        }
    )
