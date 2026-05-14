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

import copy

import pytest

from oumi.core.types.tool_call import ToolResult
from oumi.examples.ehr.executors import (
    add_diagnosis,
    get_patient,
    list_patients,
    prescribe_medication,
    record_vitals,
    update_allergies,
)


@pytest.fixture
def state():
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


# ---- read paths -----------------------------------------------------------


def test_list_patients_returns_summaries(state):
    result = list_patients({}, state)
    assert isinstance(result.output, dict)
    assert isinstance(result, ToolResult)
    assert result.updated_state is None
    assert isinstance(result.output, dict)
    assert {p["patient_id"] for p in result.output["patients"]} == {"P001", "P002"}
    # summaries omit nested fields
    assert "medications" not in result.output["patients"][0]


def test_get_patient_returns_full_record(state):
    result = get_patient({"patient_id": "P001"}, state)
    assert isinstance(result.output, dict)
    assert result.updated_state is None
    assert result.output["status"] == "ok"
    assert result.output["patient"]["name"] == "Jane Smith"
    assert result.output["patient"]["medications"][0]["name"] == "lisinopril"


def test_get_patient_unknown_returns_error_payload(state):
    result = get_patient({"patient_id": "P999"}, state)
    assert isinstance(result.output, dict)
    assert result.updated_state is None
    assert result.output == {
        "status": "error",
        "error": "not_found",
        "patient_id": "P999",
    }


# ---- write paths ----------------------------------------------------------


def test_record_vitals_appends_to_history(state):
    before = copy.deepcopy(state)
    result = record_vitals(
        {
            "patient_id": "P002",
            "timestamp": "2026-05-01T09:30",
            "bp": "120/78",
            "hr": 68,
            "temp_f": 98.6,
        },
        state,
    )
    assert isinstance(result.output, dict)
    assert result.output["status"] == "ok"
    assert result.updated_state is not None
    p002 = next(
        p for p in result.updated_state["patients"] if p["patient_id"] == "P002"
    )
    assert len(p002["vitals_history"]) == 1
    assert p002["vitals_history"][0]["bp"] == "120/78"
    # other patient untouched
    p001 = next(
        p for p in result.updated_state["patients"] if p["patient_id"] == "P001"
    )
    assert p001 == next(p for p in before["patients"] if p["patient_id"] == "P001")


def test_record_vitals_unknown_patient(state):
    result = record_vitals(
        {
            "patient_id": "PZZZ",
            "timestamp": "x",
            "bp": "120/78",
            "hr": 68,
            "temp_f": 98.6,
        },
        state,
    )
    assert isinstance(result.output, dict)
    assert result.updated_state is None
    assert result.output["error"] == "not_found"


def test_add_diagnosis_appends(state):
    result = add_diagnosis(
        {
            "patient_id": "P002",
            "code": "E11.9",
            "description": "Type 2 diabetes mellitus without complications",
            "date": "2026-05-01",
        },
        state,
    )
    assert isinstance(result.output, dict)
    assert result.output["status"] == "ok"
    assert result.updated_state is not None
    p002 = next(
        p for p in result.updated_state["patients"] if p["patient_id"] == "P002"
    )
    assert any(d["code"] == "E11.9" for d in p002["diagnoses"])


def test_add_diagnosis_rejects_duplicate(state):
    result = add_diagnosis(
        {
            "patient_id": "P001",
            "code": "I10",
            "description": "Essential hypertension",
            "date": "2026-05-01",
        },
        state,
    )
    assert isinstance(result.output, dict)
    assert result.updated_state is None
    assert result.output["error"] == "duplicate_diagnosis"


def test_prescribe_medication_appends(state):
    result = prescribe_medication(
        {"patient_id": "P002", "name": "metformin", "dose": "500mg twice daily"},
        state,
    )
    assert isinstance(result.output, dict)
    assert result.output["status"] == "ok"
    assert result.updated_state is not None
    p002 = next(
        p for p in result.updated_state["patients"] if p["patient_id"] == "P002"
    )
    assert any(m["name"] == "metformin" for m in p002["medications"])


def test_prescribe_medication_rejects_duplicate(state):
    result = prescribe_medication(
        {"patient_id": "P001", "name": "lisinopril", "dose": "10mg daily"}, state
    )
    assert isinstance(result.output, dict)
    assert result.updated_state is None
    assert result.output["error"] == "already_prescribed"


def test_prescribe_medication_blocks_allergy_conflict(state):
    result = prescribe_medication(
        {"patient_id": "P001", "name": "Penicillin", "dose": "500mg"}, state
    )
    assert isinstance(result.output, dict)
    assert result.updated_state is None
    assert result.output["error"] == "allergy_conflict"


def test_update_allergies_replaces_list(state):
    result = update_allergies(
        {"patient_id": "P001", "allergies": ["penicillin", "sulfa"]}, state
    )
    assert isinstance(result.output, dict)
    assert result.output["status"] == "ok"
    assert result.updated_state is not None
    p001 = next(
        p for p in result.updated_state["patients"] if p["patient_id"] == "P001"
    )
    assert p001["allergies"] == ["penicillin", "sulfa"]


def test_update_allergies_clears_list(state):
    result = update_allergies({"patient_id": "P001", "allergies": []}, state)
    assert isinstance(result.output, dict)
    assert result.updated_state is not None
    p001 = next(
        p for p in result.updated_state["patients"] if p["patient_id"] == "P001"
    )
    assert p001["allergies"] == []


# ---- isolation -------------------------------------------------------------


def test_executors_do_not_mutate_input_state(state):
    snapshot = copy.deepcopy(state)
    record_vitals(
        {
            "patient_id": "P001",
            "timestamp": "x",
            "bp": "1",
            "hr": 1,
            "temp_f": 1.0,
        },
        state,
    )
    add_diagnosis(
        {"patient_id": "P002", "code": "X99", "description": "x", "date": "d"},
        state,
    )
    prescribe_medication(
        {"patient_id": "P002", "name": "ibuprofen", "dose": "200mg"}, state
    )
    update_allergies({"patient_id": "P002", "allergies": ["latex"]}, state)
    assert state == snapshot
