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

"""Deterministic-environment-specific tool types.

`DeterministicTool` extends `ToolParams` with a `deterministic_outputs`
field. `DeterministicToolOutput` represents a single input -> output mapping.
Both live next to `DeterministicEnvironment` because they are only meaningful
to that env type.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.tool_params import ToolParams


@dataclass
class DeterministicToolOutput(BaseParams):
    """An input-to-output mapping for a deterministic tool."""

    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    def matches(self, arguments: dict[str, Any]) -> bool:
        """Check if the input matches the given arguments."""
        return json.dumps(self.input, sort_keys=True) == json.dumps(
            arguments, sort_keys=True
        )


@dataclass
class DeterministicTool(ToolParams):
    """`ToolParams` variant for deterministic environments.

    Carries a list of pre-canned (input, output) mappings. The owning
    `DeterministicEnvironment` looks up the matching entry on each
    `step()`.
    """

    deterministic_outputs: list[DeterministicToolOutput] = field(default_factory=list)

    @classmethod
    def create(cls, raw: Any) -> DeterministicTool:
        """Create a DeterministicTool from raw config data."""
        if isinstance(raw, DeterministicTool):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, got {type(raw)}"
            )
        outputs_raw = raw.get("deterministic_outputs", [])
        outputs = [
            o
            if isinstance(o, DeterministicToolOutput)
            else DeterministicToolOutput(**o)
            for o in outputs_raw
        ]
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=dict(raw.get("parameters", {"type": "object"})),
            output_schema=(
                dict(raw["output_schema"])
                if raw.get("output_schema") is not None
                else None
            ),
            read_only=raw.get("read_only", True),
            grounding=raw.get("grounding"),
            deterministic_outputs=outputs,
        )
