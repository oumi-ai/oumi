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

"""End-to-end checks for the EHR database example.

Loads the shipped YAML, builds the env via the registry, seeds a fresh
SQLite DB from the bundled schema/seed SQL, then walks through realistic
clinical flows by calling ``env.step`` directly. No LLM required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy

from oumi.builders.environments import build_environment
from oumi.core.configs.synthesis_config import SynthesisConfig
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "examples"
    / "synthesis"
    / "ehr_db_synth.yaml"
)
SCHEMA_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "oumi"
    / "examples"
    / "ehr_db"
)


def _exec_sql_file(conn, path: Path) -> None:
    """Execute statements in a `.sql` file. Strips line comments before splitting."""
    raw = path.read_text()
    lines = [
        line for line in raw.splitlines()
        if not line.lstrip().startswith("--")
    ]
    body = "\n".join(lines)
    for stmt in body.split(";"):
        if stmt.strip():
            conn.execute(sqlalchemy.text(stmt))


@pytest.fixture
def db_path(tmp_path):
    """Create a fresh SQLite DB seeded from schema.sql + seed.sql."""
    db_file = tmp_path / "ehr_test.db"
    engine = sqlalchemy.create_engine(
        f"sqlite:///{db_file}", isolation_level="AUTOCOMMIT", future=True
    )
    with engine.connect() as conn:
        _exec_sql_file(conn, SCHEMA_DIR / "schema.sql")
        _exec_sql_file(conn, SCHEMA_DIR / "seed.sql")
    engine.dispose()
    return db_file


@pytest.fixture
def env_params(db_path):
    cfg = SynthesisConfig.from_yaml(str(CONFIG_PATH))
    assert cfg.environment_config is not None
    env_params = cfg.environment_config.environments[0]
    # YAML uses an OmegaConf env-var interpolation for the DB path; in tests we
    # override directly so each test gets its own fresh sqlite file.
    assert env_params.env_kwargs is not None
    env_params.env_kwargs["connection"]["database"] = str(db_path)
    return env_params


def test_yaml_loads_and_env_builds(env_params):
    env = build_environment(env_params)
    try:
        assert isinstance(env, DatabaseExecutableEnvironment)
        assert set(env._executors.keys()) == {
            "list_patients",
            "get_patient",
            "record_vitals",
            "add_diagnosis",
            "prescribe_medication",
            "update_allergies",
        }
    finally:
        if isinstance(env, DatabaseExecutableEnvironment):
            env.close()


def test_chart_review_flow(env_params):
    """Clinician asks for the chart for Jane Smith."""
    env = build_environment(env_params)
    assert isinstance(env, DatabaseExecutableEnvironment)
    try:
        listing = env.step("list_patients", {})
        assert isinstance(listing.output, dict)
        summaries = listing.output["patients"]
        jane = next(p for p in summaries if p["name"] == "Jane Smith")

        record = env.step("get_patient", {"patient_id": jane["patient_id"]})
        assert isinstance(record.output, dict)
        assert record.output["status"] == "ok"
        patient = record.output["patient"]
        assert "penicillin" in patient["allergies"]
        assert any(m["name"] == "lisinopril" for m in patient["medications"])
    finally:
        env.close()


def test_record_vitals_flow(env_params):
    env = build_environment(env_params)
    assert isinstance(env, DatabaseExecutableEnvironment)
    try:
        write = env.step(
            "record_vitals",
            {
                "patient_id": "P002",
                "timestamp": "2026-05-01T08:00",
                "bp": "118/76",
                "hr": 70,
                "temp_f": 98.4,
            },
        )
        assert write.output["status"] == "ok"

        read = env.step("get_patient", {"patient_id": "P002"})
        history = read.output["patient"]["vitals_history"]
        assert any(v["timestamp"] == "2026-05-01T08:00" for v in history)
    finally:
        env.close()


def test_allergy_conflict_blocks_prescription(env_params):
    """Jane Smith is allergic to penicillin — prescribing it must be refused."""
    env = build_environment(env_params)
    assert isinstance(env, DatabaseExecutableEnvironment)
    try:
        result = env.step(
            "prescribe_medication",
            {"patient_id": "P001", "name": "penicillin", "dose": "500mg"},
        )
        assert result.output["status"] == "error"
        assert result.output["error"] == "allergy_conflict"
    finally:
        env.close()


def test_duplicate_diagnosis_returns_error(env_params):
    env = build_environment(env_params)
    assert isinstance(env, DatabaseExecutableEnvironment)
    try:
        result = env.step(
            "add_diagnosis",
            {
                "patient_id": "P001",
                "code": "I10",
                "description": "Essential hypertension",
                "date": "2026-05-01",
            },
        )
        assert result.output["status"] == "error"
        assert result.output["error"] == "duplicate_diagnosis"
    finally:
        env.close()
