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

"""Deterministic environment with fixed lookup responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import DeterministicToolOutput, Tool
from oumi.environments.tool_result import ToolResult


@dataclass
class DeterministicEnvironment(BaseEnvironment):
    """Environment that resolves tools from fixed lookups."""

    ENVIRONMENT_TYPE: ClassVar[str] = "deterministic"
    type: str = field(init=False, default=ENVIRONMENT_TYPE)

    def _coerce_tools(self, tools: list[Any]) -> list[Tool]:
        """Coerce tools and deterministic outputs into typed objects."""
        coerced_tools: list[Tool] = []
        for raw_tool in tools:
            tool = Tool.create(raw_tool)
            tool.deterministic_outputs = [
                entry
                if isinstance(entry, DeterministicToolOutput)
                else DeterministicToolOutput(**entry)
                for entry in tool.deterministic_outputs
            ]
            coerced_tools.append(tool)
        return coerced_tools

    def __post_init__(self):
        """Validate that deterministic tools have deterministic output entry."""
        super().__post_init__()
        for tool in self.tools:
            if not tool.deterministic_outputs:
                raise ValueError(
                    f"Deterministic tool '{tool.id}' must have at least one "
                    "deterministic_output entry."
                )
            seen: set[str] = set()
            for entry in tool.deterministic_outputs:
                key = json.dumps(entry.input, sort_keys=True)
                if key in seen:
                    raise ValueError(
                        f"Deterministic tool '{tool.id}' has duplicate "
                        f"deterministic input entry: {entry.input}"
                    )
                seen.add(key)

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Resolve a deterministic tool call to its output."""
        tool = self._get_tool_or_raise(tool_id)
        for entry in tool.deterministic_outputs:
            if entry.matches(arguments):
                return ToolResult(output=entry.output)
        return ToolResult(output=None)
