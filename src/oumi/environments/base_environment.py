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

"""Abstract base class for tool environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

from oumi.core.configs.params.base_params import BaseParams
from oumi.environments.base_tool import Tool
from oumi.environments.tool_result import ToolResult


@dataclass
class BaseEnvironment(BaseParams, ABC):
    """Abstract base class for tool environments."""

    _registry: ClassVar[dict[str, type[BaseEnvironment]]] = {}

    id: str
    name: str
    description: str
    tools: list[Tool] = field(default_factory=list)

    def __init_subclass__(cls, **kwargs):
        """Register subclass in the environment type registry."""
        super().__init_subclass__(**kwargs)
        environment_type = getattr(cls, "ENVIRONMENT_TYPE", None)
        if environment_type is not None:
            cls._registry[environment_type] = cls

    def __post_init__(self):
        """Validate common fields and coerce tools."""
        if not self.id:
            raise ValueError(f"{type(self).__name__}.id cannot be empty.")
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name cannot be empty.")
        if not self.description:
            raise ValueError(f"{type(self).__name__}.description cannot be empty.")
        self.tools = self._coerce_tools(self.tools)
        self._validate_unique_tool_ids()

    def _validate_unique_tool_ids(self) -> None:
        tool_ids: set[str] = set()
        for tool in self.tools:
            if tool.id in tool_ids:
                raise ValueError(
                    f"{type(self).__name__} '{self.id}' contains duplicate "
                    f"tool id '{tool.id}'."
                )
            tool_ids.add(tool.id)

    def _coerce_tools(self, tools: list[Any]) -> list[Tool]:
        """Coerce raw tool definitions into Tool instances."""
        return [Tool.create(tool) for tool in tools]

    @abstractmethod
    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call within this environment."""

    def _get_tool(self, tool_id: str) -> Tool | None:
        """Look up a tool owned by this environment."""
        for tool in self.tools:
            if tool.id == tool_id:
                return tool
        return None

    def _get_tool_or_raise(self, tool_id: str) -> Tool:
        tool = self._get_tool(tool_id)
        if tool is None:
            raise ValueError(
                f"Tool '{tool_id}' not found in environment '{self.id}'. "
                f"Available tools: {[tool.id for tool in self.tools]}"
            )
        return tool

    @classmethod
    def create(cls, raw: Any) -> BaseEnvironment:
        """Create a concrete environment from raw config data.

        Raises:
            TypeError: If raw is not a mapping or BaseEnvironment.
            ValueError: If type is missing or unsupported.
        """
        if isinstance(raw, BaseEnvironment):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                "Environment definitions must be environment objects or mappings, "
                f"got {type(raw)}"
            )
        environment_type = raw.get("type")
        if environment_type is None:
            raise ValueError(
                "Environment definition must include a 'type' field. "
                f"Supported types: {sorted(cls._registry)}"
            )
        if not isinstance(environment_type, str):
            environment_type = str(environment_type)
        environment_cls = cls._registry.get(environment_type)
        if environment_cls is None:
            raise ValueError(f"Unsupported environment type: {environment_type}")
        init_fields = {
            field_def.name for field_def in fields(environment_cls) if field_def.init
        }
        return environment_cls(
            **{key: value for key, value in raw.items() if key in init_fields}
        )
