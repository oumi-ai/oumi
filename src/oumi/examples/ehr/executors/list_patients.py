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
