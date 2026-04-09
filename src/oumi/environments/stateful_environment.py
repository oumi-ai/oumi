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

"""Stateful environment with mutable shared state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import BaseTool
from oumi.environments.types import ToolEnvironmentType


@dataclass
class StatefulTool(BaseTool):
    """Tool bound to a stateful environment."""

    output_schema: dict[str, Any] = field(default_factory=dict)
    read_only: bool = True

    @classmethod
    def create(cls, raw: Mapping[str, Any] | BaseTool) -> StatefulTool:
        """Create a StatefulTool from raw config data."""
        if isinstance(raw, StatefulTool):
            return raw
        if isinstance(raw, BaseTool):
            return cls(
                id=raw.id,
                name=raw.name,
                description=raw.description,
                parameters=raw.parameters,
            )
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, "
                f"got {type(raw)}"
            )
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=raw.get("parameters", {}),
            output_schema=raw.get("output_schema", {}),
            read_only=raw.get("read_only", True),
        )


@dataclass
class StatefulEnvironment(BaseEnvironment):
    """Environment with mutable shared state.

    Maintains a JSON state dict that tools can read and modify. The
    ``system_prompt`` instructs the LLM how to simulate this environment.
    Each tool is a ``StatefulTool`` with an output schema and read/write flag.
    """

    ENVIRONMENT_TYPE: ClassVar[ToolEnvironmentType] = ToolEnvironmentType.STATEFUL
    type: ToolEnvironmentType = field(init=False, default=ToolEnvironmentType.STATEFUL)
    system_prompt: str = ""
    state_schema: dict[str, Any] | None = None
    initial_state: dict[str, Any] | None = None
    tools: list[StatefulTool] = field(default_factory=list)  # type: ignore[assignment]

    def _coerce_tools(self, tools: list[Any]) -> list[StatefulTool]:
        """Coerce raw tool definitions into StatefulTool instances."""
        return [StatefulTool.create(t) for t in tools]

    def _validate_type_specific(self) -> None:
        """Validate stateful-specific fields."""
        if not self.system_prompt:
            raise ValueError("StatefulEnvironment.system_prompt cannot be empty.")
