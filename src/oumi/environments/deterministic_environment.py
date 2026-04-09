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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.core.configs.params.base_params import BaseParams
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import BaseTool
from oumi.environments.types import ToolEnvironmentType


@dataclass
class DeterministicToolOutput(BaseParams):
    """An input-to-output mapping for a deterministic tool."""

    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the input and output fields are not empty."""
        if not self.input:
            raise ValueError("DeterministicToolOutput.input cannot be empty.")
        if not self.output:
            raise ValueError("DeterministicToolOutput.output cannot be empty.")

    def matches(self, arguments: dict[str, Any]) -> bool:
        """Check if the input matches the given arguments."""
        return json.dumps(self.input, sort_keys=True) == json.dumps(
            arguments, sort_keys=True
        )


@dataclass
class DeterministicTool(BaseTool):
    """Tool with fixed input-to-output lookup responses."""

    deterministic_outputs: list[DeterministicToolOutput] = field(default_factory=list)

    def __post_init__(self):
        """Validate deterministic tool fields."""
        super().__post_init__()
        if not self.deterministic_outputs:
            raise ValueError(
                f"DeterministicTool '{self.id}' must have at least one "
                f"deterministic_output entry."
            )
        self._check_deterministic_duplicates()

    def _check_deterministic_duplicates(self) -> None:
        seen: set[str] = set()
        for entry in self.deterministic_outputs:
            key = json.dumps(entry.input, sort_keys=True)
            if key in seen:
                raise ValueError(
                    f"DeterministicTool '{self.id}' has duplicate "
                    f"deterministic input entry: {entry.input}"
                )
            seen.add(key)

    def resolve_deterministic(
        self, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Resolve a deterministic output for the given arguments."""
        for entry in self.deterministic_outputs:
            if entry.matches(arguments):
                return entry.output
        return None

    @classmethod
    def create(cls, raw: Mapping[str, Any] | BaseTool) -> DeterministicTool:
        """Create a DeterministicTool from raw config data."""
        if isinstance(raw, DeterministicTool):
            return raw
        if isinstance(raw, BaseTool):
            raise TypeError(
                f"Cannot coerce {type(raw).__name__} to DeterministicTool. "
                f"Use a mapping with 'deterministic_outputs'."
            )
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, "
                f"got {type(raw)}"
            )
        deterministic_outputs = [
            entry
            if isinstance(entry, DeterministicToolOutput)
            else DeterministicToolOutput(**entry)
            for entry in raw.get("deterministic_outputs", [])
        ]
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=raw.get("parameters", {}),
            deterministic_outputs=deterministic_outputs,
        )


@dataclass
class DeterministicEnvironment(BaseEnvironment):
    """Environment that resolves tools from fixed lookups.

    Each tool is a ``DeterministicTool`` with a list of input-to-output
    mappings. The environment owns the resolution logic.
    """

    ENVIRONMENT_TYPE: ClassVar[ToolEnvironmentType] = ToolEnvironmentType.DETERMINISTIC
    type: ToolEnvironmentType = field(
        init=False, default=ToolEnvironmentType.DETERMINISTIC
    )
    tools: list[DeterministicTool] = field(default_factory=list)  # type: ignore[assignment]

    def _coerce_tools(self, tools: list[Any]) -> list[DeterministicTool]:
        """Coerce raw tool definitions into DeterministicTool instances."""
        return [DeterministicTool.create(t) for t in tools]

    def _validate_type_specific(self) -> None:
        return

    def resolve(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Resolve a deterministic tool call to its output.

        Raises:
            ValueError: If tool_id is not found in this environment.
        """
        for tool in self.tools:
            if tool.id == tool_id:
                return tool.resolve_deterministic(arguments)
        raise ValueError(
            f"Tool '{tool_id}' not found in environment '{self.id}'. "
            f"Available tools: {[t.id for t in self.tools]}"
        )
