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

from __future__ import annotations

from typing import Any

from oumi.core.types.tool_call import ToolResult
from oumi.examples.ehr._state import find_patient, replace_patient


def prescribe_medication(
    arguments: dict[str, Any], state: dict[str, Any]
) -> ToolResult:
    """Prescribe a medication. Rejects duplicates and allergy conflicts."""
    patient_id = arguments["patient_id"]
    patient = find_patient(state, patient_id)
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
        updated_state=replace_patient(state, patient_id, updated),
    )
