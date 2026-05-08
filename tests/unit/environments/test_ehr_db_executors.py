# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for EHR DB executors (SQLite-backed in-memory)."""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy

from oumi.core.types.tool_call import ToolResult
from oumi.examples.ehr_db import executors as ehr_db


SCHEMA_DIR = Path(__file__).resolve().parents[3] / "src" / "oumi" / "examples" / "ehr_db"


@pytest.fixture
def db():
    """In-memory SQLite engine seeded from schema.sql + seed.sql."""
    engine = sqlalchemy.create_engine(
        "sqlite:///:memory:", isolation_level="AUTOCOMMIT", future=True
    )
    schema_sql = (SCHEMA_DIR / "schema.sql").read_text()
    seed_sql = (SCHEMA_DIR / "seed.sql").read_text()
    def _exec_sql(conn: sqlalchemy.engine.Connection, raw: str) -> None:
        # Strip single-line comments before splitting on ";" so that
        # leading comment lines don't get concatenated into the first statement.
        lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith("--")]
        cleaned = "\n".join(lines)
        for stmt in cleaned.split(";"):
            if stmt.strip():
                conn.execute(sqlalchemy.text(stmt))

    with engine.connect() as conn:
        _exec_sql(conn, schema_sql)
        _exec_sql(conn, seed_sql)
    yield engine
    engine.dispose()


def test_list_patients(db):
    with db.connect() as conn:
        result = ehr_db.list_patients({}, conn)
    assert isinstance(result, ToolResult)
    assert isinstance(result.output, dict)
    patients = result.output["patients"]
    assert len(patients) == 6
    assert {p["patient_id"] for p in patients} == {f"P00{i}" for i in range(1, 7)}
    assert all({"patient_id", "name", "dob", "status"} <= set(p) for p in patients)


def test_get_patient_known(db):
    with db.connect() as conn:
        result = ehr_db.get_patient({"patient_id": "P001"}, conn)
    assert result.output["status"] == "ok"
    patient = result.output["patient"]
    assert patient["name"] == "Jane Smith"
    assert "penicillin" in patient["allergies"]
    assert any(m["name"] == "lisinopril" for m in patient["medications"])


def test_get_patient_unknown(db):
    with db.connect() as conn:
        result = ehr_db.get_patient({"patient_id": "P999"}, conn)
    assert result.output == {
        "status": "error",
        "error": "not_found",
        "patient_id": "P999",
    }


def test_record_vitals_appends(db):
    args = {
        "patient_id": "P001",
        "timestamp": "2026-05-01T09:00",
        "bp": "120/80",
        "hr": 70,
        "temp_f": 98.6,
    }
    with db.connect() as conn:
        result = ehr_db.record_vitals(args, conn)
    assert result.output["status"] == "ok"
    with db.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT timestamp, bp, hr, temp_f FROM vitals "
                "WHERE patient_id='P001' ORDER BY timestamp"
            )
        ).mappings().all()
    assert any(r["timestamp"] == "2026-05-01T09:00" for r in rows)


def test_add_diagnosis_duplicate_returns_error(db):
    args = {
        "patient_id": "P001",
        "code": "I10",  # already present in seed
        "description": "Essential hypertension",
        "date": "2026-05-01",
    }
    with db.connect() as conn:
        result = ehr_db.add_diagnosis(args, conn)
    assert result.output["status"] == "error"
    assert result.output["error"] == "duplicate_diagnosis"


def test_prescribe_medication_allergy_conflict(db):
    args = {"patient_id": "P001", "name": "Penicillin", "dose": "500mg"}
    with db.connect() as conn:
        result = ehr_db.prescribe_medication(args, conn)
    assert result.output["status"] == "error"
    assert result.output["error"] == "allergy_conflict"


def test_prescribe_medication_already_prescribed(db):
    args = {"patient_id": "P001", "name": "lisinopril", "dose": "20mg daily"}
    with db.connect() as conn:
        result = ehr_db.prescribe_medication(args, conn)
    assert result.output["status"] == "error"
    assert result.output["error"] == "already_prescribed"


def test_update_allergies_replaces(db):
    args = {"patient_id": "P001", "allergies": ["latex"]}
    with db.connect() as conn:
        result = ehr_db.update_allergies(args, conn)
    assert result.output["status"] == "ok"
    with db.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT substance FROM allergies WHERE patient_id='P001'"
            )
        ).scalars().all()
    assert sorted(rows) == ["latex"]
