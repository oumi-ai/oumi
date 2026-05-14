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

"""Stateful EHR tool executors.

Each executor takes a deepcopy of the env's current state and returns a
:class:`ToolResult`. Read-only executors leave ``updated_state=None``; write
executors return a fully replaced state dict, with the patient list rebuilt
to reflect the mutation. Executor-level errors that the assistant should be
able to recover from (unknown patient, etc.) are returned as structured
``{"status": "error", ...}`` payloads — *not* raised — so the model can
self-correct on the next turn.
"""

from __future__ import annotations

from typing import Any

from oumi.core.types.tool_call import ToolResult


def _find_patient(state: dict[str, Any], patient_id: str) -> dict[str, Any] | None:
    return next(
        (p for p in state["patients"] if p["patient_id"] == patient_id),
        None,
    )


def _replace_patient(
    state: dict[str, Any], patient_id: str, updated: dict[str, Any]
) -> dict[str, Any]:
    """Return a new state dict with the given patient record replaced."""
    return {
        **state,
        "patients": [
            updated if p["patient_id"] == patient_id else p for p in state["patients"]
        ],
    }


def list_patients(arguments: dict[str, Any], state: dict[str, Any]) -> ToolResult:
    """List patient summaries (read-only)."""
    summaries = [
        {
            "patient_id": p["patient_id"],
            "name": p["name"],
            "dob": p["dob"],
            "status": p["status"],
        }
        for p in state["patients"]
    ]
    return ToolResult(output={"patients": summaries})


def get_patient(arguments: dict[str, Any], state: dict[str, Any]) -> ToolResult:
    """Fetch the full record for a patient_id (read-only)."""
    patient_id = arguments["patient_id"]
    patient = _find_patient(state, patient_id)
    if patient is None:
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    return ToolResult(output={"status": "ok", "patient": patient})


def record_vitals(arguments: dict[str, Any], state: dict[str, Any]) -> ToolResult:
    """Append a vitals reading to a patient's history."""
    patient_id = arguments["patient_id"]
    patient = _find_patient(state, patient_id)
    if patient is None:
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    entry = {
        "timestamp": arguments["timestamp"],
        "bp": arguments["bp"],
        "hr": arguments["hr"],
        "temp_f": arguments["temp_f"],
    }
    updated = {
        **patient,
        "vitals_history": [*patient["vitals_history"], entry],
    }
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "vitals_recorded": entry,
        },
        updated_state=_replace_patient(state, patient_id, updated),
    )


def add_diagnosis(arguments: dict[str, Any], state: dict[str, Any]) -> ToolResult:
    """Append a diagnosis (ICD code + description + date) to a patient."""
    patient_id = arguments["patient_id"]
    patient = _find_patient(state, patient_id)
    if patient is None:
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    new_diagnosis = {
        "code": arguments["code"],
        "description": arguments["description"],
        "date": arguments["date"],
    }
    if any(d["code"] == new_diagnosis["code"] for d in patient["diagnoses"]):
        return ToolResult(
            output={
                "status": "error",
                "error": "duplicate_diagnosis",
                "patient_id": patient_id,
                "code": new_diagnosis["code"],
            }
        )
    updated = {
        **patient,
        "diagnoses": [*patient["diagnoses"], new_diagnosis],
    }
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "diagnosis_added": new_diagnosis,
        },
        updated_state=_replace_patient(state, patient_id, updated),
    )


def prescribe_medication(
    arguments: dict[str, Any], state: dict[str, Any]
) -> ToolResult:
    """Append a medication to a patient's active medication list."""
    patient_id = arguments["patient_id"]
    patient = _find_patient(state, patient_id)
    if patient is None:
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    name = arguments["name"]
    dose = arguments["dose"]
    if any(m["name"].lower() == name.lower() for m in patient["medications"]):
        return ToolResult(
            output={
                "status": "error",
                "error": "already_prescribed",
                "patient_id": patient_id,
                "medication": name,
            }
        )
    if name.lower() in (a.lower() for a in patient["allergies"]):
        return ToolResult(
            output={
                "status": "error",
                "error": "allergy_conflict",
                "patient_id": patient_id,
                "medication": name,
            }
        )
    new_med = {"name": name, "dose": dose}
    updated = {
        **patient,
        "medications": [*patient["medications"], new_med],
    }
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "medication_added": new_med,
        },
        updated_state=_replace_patient(state, patient_id, updated),
    )


def update_allergies(arguments: dict[str, Any], state: dict[str, Any]) -> ToolResult:
    """Replace a patient's allergy list with the supplied list."""
    patient_id = arguments["patient_id"]
    patient = _find_patient(state, patient_id)
    if patient is None:
        return ToolResult(
            output={"status": "error", "error": "not_found", "patient_id": patient_id}
        )
    new_allergies = list(arguments["allergies"])
    updated = {**patient, "allergies": new_allergies}
    return ToolResult(
        output={
            "status": "ok",
            "patient_id": patient_id,
            "allergies": new_allergies,
        },
        updated_state=_replace_patient(state, patient_id, updated),
    )
