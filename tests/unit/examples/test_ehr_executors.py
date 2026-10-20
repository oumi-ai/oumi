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

"""Tests for the EHR stateful synthesis example's tool executors."""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest

from oumi.core.types.tool_call import ToolResult
from oumi.environments.utils import resolve_executor

_EXAMPLE_DIR = (
    Path(__file__).parents[3]
    / "configs"
    / "examples"
    / "synthesis"
    / "ehr_stateful_agent"
)


@pytest.fixture(scope="module")
def ehr():
    """Load the example's executors the way OUMI_EXTRA_DEPS_FILE does."""
    spec = importlib.util.spec_from_file_location(
        "ehr_example_executors", _EXAMPLE_DIR / "executors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state() -> dict[str, Any]:
    return {
        "patients": [
            {
                "patient_id": "P001",
                "name": "Jane Smith",
                "dob": "1985-03-15",
                "allergies": ["penicillin"],
                "medications": [{"name": "lisinopril", "dose": "10mg daily"}],
                "diagnoses": [
                    {
                        "code": "I10",
                        "description": "Essential hypertension",
                        "date": "2024-06-12",
                    }
                ],
                "vitals_history": [
                    {
                        "timestamp": "2024-06-12T10:00",
                        "bp": "138/85",
                        "hr": 72,
                        "temp_f": 98.4,
                    }
                ],
                "status": "active",
            },
            {
                "patient_id": "P002",
                "name": "Marcus Lee",
                "dob": "1972-11-04",
                "allergies": [],
                "medications": [],
                "diagnoses": [],
                "vitals_history": [],
                "status": "active",
            },
        ]
    }


def _output(result: ToolResult) -> dict[str, Any]:
    """Narrow ``output`` from ``JsonValue`` to a dict for indexing."""
    assert isinstance(result.output, dict)
    return result.output


def _patient(state: dict[str, Any], patient_id: str) -> dict[str, Any]:
    return next(p for p in state["patients"] if p["patient_id"] == patient_id)


def test_config_executor_names_resolve_to_the_example_functions(ehr):
    """Every ``executor:`` in the config must be a registered ``ehr.*`` name."""
    config = (_EXAMPLE_DIR / "ehr_stateful_synth.yaml").read_text()
    names = re.findall(r"^\s*executor:\s*(\S+)", config, re.MULTILINE)
    assert len(names) == 6
    for name in names:
        namespace, _, attr = name.partition(".")
        assert namespace == "ehr"
        assert resolve_executor(name, "t") is getattr(ehr, attr)


def test_list_patients_returns_summaries_only(ehr, state):
    result = ehr.list_patients({}, state)
    assert result.updated_state is None
    assert _output(result) == {
        "patients": [
            {
                "patient_id": "P001",
                "name": "Jane Smith",
                "dob": "1985-03-15",
                "status": "active",
            },
            {
                "patient_id": "P002",
                "name": "Marcus Lee",
                "dob": "1972-11-04",
                "status": "active",
            },
        ]
    }


def test_get_patient_returns_full_record(ehr, state):
    result = ehr.get_patient({"patient_id": "P001"}, state)
    assert result.updated_state is None
    assert _output(result) == {"status": "ok", "patient": state["patients"][0]}


def test_get_patient_unknown_returns_not_found(ehr, state):
    result = ehr.get_patient({"patient_id": "P999"}, state)
    assert result.updated_state is None
    assert _output(result) == {
        "status": "error",
        "error": "not_found",
        "patient_id": "P999",
    }


def test_record_vitals_appends_and_leaves_other_patients_alone(ehr, state):
    before = copy.deepcopy(state)
    result = ehr.record_vitals(
        {
            "patient_id": "P002",
            "timestamp": "2026-05-01T09:30",
            "bp": "120/78",
            "hr": 68,
            "temp_f": 98.6,
        },
        state,
    )
    assert _output(result)["status"] == "ok"
    assert result.updated_state is not None
    vitals = _patient(result.updated_state, "P002")["vitals_history"]
    assert [v["bp"] for v in vitals] == ["120/78"]
    assert _patient(result.updated_state, "P001") == _patient(before, "P001")


def test_record_vitals_unknown_patient(ehr, state):
    result = ehr.record_vitals(
        {
            "patient_id": "PZZZ",
            "timestamp": "x",
            "bp": "120/78",
            "hr": 68,
            "temp_f": 1.0,
        },
        state,
    )
    assert result.updated_state is None
    assert _output(result)["error"] == "not_found"


def test_add_diagnosis_appends(ehr, state):
    result = ehr.add_diagnosis(
        {
            "patient_id": "P002",
            "code": "E11.9",
            "description": "Type 2 diabetes mellitus without complications",
            "date": "2026-05-01",
        },
        state,
    )
    assert _output(result)["status"] == "ok"
    assert result.updated_state is not None
    codes = [d["code"] for d in _patient(result.updated_state, "P002")["diagnoses"]]
    assert codes == ["E11.9"]


@pytest.mark.parametrize("code", ["I10", "i10", "  I10  "])
def test_add_diagnosis_rejects_duplicate_regardless_of_case_or_padding(
    ehr, state, code
):
    result = ehr.add_diagnosis(
        {
            "patient_id": "P001",
            "code": code,
            "description": "Essential hypertension",
            "date": "2026-05-01",
        },
        state,
    )
    assert result.updated_state is None
    assert _output(result)["error"] == "duplicate_diagnosis"


def test_prescribe_medication_appends(ehr, state):
    result = ehr.prescribe_medication(
        {"patient_id": "P002", "name": "metformin", "dose": "500mg twice daily"}, state
    )
    assert _output(result)["status"] == "ok"
    assert result.updated_state is not None
    meds = [m["name"] for m in _patient(result.updated_state, "P002")["medications"]]
    assert meds == ["metformin"]


def test_prescribe_medication_rejects_already_prescribed(ehr, state):
    result = ehr.prescribe_medication(
        {"patient_id": "P001", "name": "Lisinopril", "dose": "10mg daily"}, state
    )
    assert result.updated_state is None
    assert _output(result)["error"] == "already_prescribed"


def test_prescribe_medication_blocks_allergy_conflict(ehr, state):
    result = ehr.prescribe_medication(
        {"patient_id": "P001", "name": "Penicillin", "dose": "500mg"}, state
    )
    assert result.updated_state is None
    assert _output(result)["error"] == "allergy_conflict"


@pytest.mark.parametrize("allergies", [["penicillin", "sulfa"], []])
def test_update_allergies_replaces_the_list(ehr, state, allergies):
    result = ehr.update_allergies({"patient_id": "P001", "allergies": allergies}, state)
    assert _output(result)["status"] == "ok"
    assert result.updated_state is not None
    assert _patient(result.updated_state, "P001")["allergies"] == allergies


def test_executors_never_mutate_the_input_state(ehr, state):
    snapshot = copy.deepcopy(state)
    ehr.record_vitals(
        {"patient_id": "P001", "timestamp": "x", "bp": "1", "hr": 1, "temp_f": 1.0},
        state,
    )
    ehr.add_diagnosis(
        {"patient_id": "P002", "code": "X99", "description": "x", "date": "d"}, state
    )
    ehr.prescribe_medication(
        {"patient_id": "P002", "name": "ibuprofen", "dose": "200mg"}, state
    )
    ehr.update_allergies({"patient_id": "P002", "allergies": ["latex"]}, state)
    assert state == snapshot
