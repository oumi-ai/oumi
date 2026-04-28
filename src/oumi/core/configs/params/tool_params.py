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

"""Tool definitions and execution results shared by all environment types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class ToolSchema(BaseParams):
    """JSON schema for tool inputs or outputs."""

    type: str = "object"
    properties: dict[str, ToolSchema] = field(default_factory=dict)
    description: str | None = None
    required: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, raw: Any) -> ToolSchema:
        """Create a schema from raw config data."""
        if isinstance(raw, ToolSchema):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool schema definitions must be schema objects or mappings, got "
                f"{type(raw)}"
            )
        return cls(
            type=raw.get("type", "object"),
            properties={
                key: cls.create(value)
                for key, value in raw.get("properties", {}).items()
            },
            description=raw.get("description"),
            required=raw.get("required", []),
        )

    def __post_init__(self):
        """Validate schema fields."""
        if not self.type:
            raise ValueError(f"{type(self).__name__}.type cannot be empty.")
        if not isinstance(self.properties, dict):
            raise ValueError(f"{type(self).__name__}.properties must be a dict.")
        if not all(isinstance(value, ToolSchema) for value in self.properties.values()):
            raise ValueError(
                f"{type(self).__name__}.properties values must be ToolSchema."
            )
        if self.description is not None and not isinstance(self.description, str):
            raise ValueError(
                f"{type(self).__name__}.description must be a string when specified."
            )
        if not isinstance(self.required, list) or not all(
            isinstance(param, str) for param in self.required
        ):
            raise ValueError(
                f"{type(self).__name__}.required must be a list of strings."
            )
        missing_required = set(self.required) - set(self.properties)
        if missing_required:
            raise ValueError(
                f"{type(self).__name__}.required contains unknown properties: "
                f"{sorted(missing_required)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-schema-shaped dict."""
        schema: dict[str, Any] = {"type": self.type}
        if self.properties:
            schema["properties"] = {
                key: value.to_dict() for key, value in self.properties.items()
            }
        if self.description is not None:
            schema["description"] = self.description
        if self.required:
            schema["required"] = list(self.required)
        return schema


@dataclass
class ToolParams(BaseParams):
    """Tool schema owned by an environment."""

    id: str
    name: str
    description: str
    parameters: ToolSchema = field(default_factory=ToolSchema)
    output_schema: ToolSchema | None = None
    read_only: bool = True

    @classmethod
    def create(cls, raw: Any) -> ToolParams:
        """Create a tool from raw config data."""
        if isinstance(raw, ToolParams):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, got {type(raw)}"
            )
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=ToolSchema.create(raw.get("parameters", {})),
            output_schema=(
                ToolSchema.create(raw["output_schema"])
                if raw.get("output_schema") is not None
                else None
            ),
            read_only=raw.get("read_only", True),
        )

    def __post_init__(self):
        """Validate common tool fields."""
        if not self.id:
            raise ValueError(f"{type(self).__name__}.id cannot be empty.")
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name cannot be empty.")
        if not self.description:
            raise ValueError(f"{type(self).__name__}.description cannot be empty.")
        if not isinstance(self.parameters, ToolSchema):
            self.parameters = ToolSchema.create(self.parameters)
        if self.output_schema is not None and not isinstance(
            self.output_schema, ToolSchema
        ):
            self.output_schema = ToolSchema.create(self.output_schema)

    def to_llm_schema(self) -> dict[str, Any]:
        """Export a provider-agnostic schema for LLM tool registration."""
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
        }
        if self.output_schema is not None:
            schema["output_schema"] = self.output_schema.to_dict()
        return schema


@dataclass
class ToolResult(BaseParams):
    """Result returned by an environment step."""

    output: str | dict[str, Any]
    updated_state: dict[str, Any] | None = None
