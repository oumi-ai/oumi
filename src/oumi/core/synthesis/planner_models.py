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

"""Pydantic models for conversation planner guided decoding.

The planner emits a structured turn-by-turn plan; ``PlannerOutput`` is
the wire shape and ``model_json_schema()`` produces the JSON schema we
hand to ``GuidedDecodingParams.json``. Defining the schema as a class
gives us a single source of truth — the Pydantic model is what we'd
also use to validate or parse planner output if we ever stop trusting
guided decoding alone.
"""

from pydantic import BaseModel, ConfigDict, Field


class PlannedTurn(BaseModel):
    """A single turn in a planned multi-turn conversation."""

    model_config = ConfigDict(extra="forbid")

    turn: int = Field(ge=1, description="1-indexed turn number.")
    instruction: str = Field(description="What should happen on this turn.")


class PlannerOutput(BaseModel):
    """Top-level planner output: a list of planned turns wrapped in an object."""

    model_config = ConfigDict(extra="forbid")

    turns: list[PlannedTurn]


PLANNER_JSON_SCHEMA: dict = PlannerOutput.model_json_schema()
