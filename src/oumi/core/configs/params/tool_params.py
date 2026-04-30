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
from oumi.core.configs.params.grounding_params import ToolGroundingConfig


class ToolError(Exception):
    """Base class for tool errors surfaced back to the LLM.

    Subclasses of this exception are caught by the tool-call loop and
    re-emitted as structured ``tool`` messages so the model can self-correct
    on the next iteration.
    """


class ToolArgumentError(ToolError):
    """Raised when tool-call arguments fail schema validation."""


class ToolLookupError(ToolError):
    """Raised when a tool cannot resolve an output for the given arguments.

    Currently used by ``DeterministicEnvironment`` when no configured
    ``DeterministicToolOutput`` matches the provided arguments.
    """


def _validate_json_schema_value(
    value: Any,
    schema: dict[str, Any],
    path: str = "$",
) -> None:
    """Validate a JSON-like value against a minimal JSON-schema subset.

    Supports: object (properties + required), array (items), string,
    integer, number, boolean, null, and enum. Raises ``ValueError`` on
    the first violation, with a JSONPath-style path prefix.
    """
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object.")
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ValueError(f"{path}.{key} is required.")
        for key, child_value in value.items():
            child_schema = schema.get("properties", {}).get(key)
            if child_schema is not None:
                _validate_json_schema_value(child_value, child_schema, f"{path}.{key}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path} must be an array.")
        item_schema = schema.get("items")
        if item_schema is not None:
            for idx, item in enumerate(value):
                _validate_json_schema_value(item, item_schema, f"{path}[{idx}]")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string.")
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer.")
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must be a number.")
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be a boolean.")
    elif expected_type == "null":
        if value is not None:
            raise ValueError(f"{path} must be null.")

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise ValueError(f"{path} must be one of {enum_values}.")


@dataclass
class ToolSchema(BaseParams):
    """JSON schema for tool inputs or outputs."""

    type: str = "object"
    properties: dict[str, ToolSchema] = field(default_factory=dict)
    description: str | None = None
    required: list[str] = field(default_factory=list)
    items: ToolSchema | None = None
    enum: list[Any] | None = None

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
            items=(cls.create(raw["items"]) if raw.get("items") is not None else None),
            enum=list(raw["enum"]) if raw.get("enum") is not None else None,
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
        if self.items is not None and not isinstance(self.items, ToolSchema):
            self.items = ToolSchema.create(self.items)
        if self.enum is not None and not isinstance(self.enum, list):
            raise ValueError(
                f"{type(self).__name__}.enum must be a list when specified."
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
        if self.items is not None:
            schema["items"] = self.items.to_dict()
        if self.enum is not None:
            schema["enum"] = list(self.enum)
        return schema

    def validate(self, value: Any, *, path: str = "$") -> None:
        """Validate a value against this schema.

        Args:
            value: The value to validate.
            path: JSONPath-style prefix used in error messages. Callers
                validating tool arguments should pass ``path="arguments"``
                to produce model-friendly messages.

        Raises:
            ToolArgumentError: If ``value`` does not conform to the schema.
        """
        try:
            _validate_json_schema_value(value, self.to_dict(), path=path)
        except ValueError as e:
            raise ToolArgumentError(str(e)) from e


@dataclass
class ToolParams(BaseParams):
    """Tool schema owned by an environment."""

    id: str
    name: str
    description: str
    parameters: ToolSchema = field(default_factory=ToolSchema)
    output_schema: ToolSchema | None = None
    read_only: bool = True
    grounding: ToolGroundingConfig | None = None

    @classmethod
    def create(cls, raw: Any) -> ToolParams:
        """Create a tool from raw config data."""
        if isinstance(raw, ToolParams):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, got {type(raw)}"
            )
        grounding_raw = raw.get("grounding")
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
            grounding=(
                grounding_raw
                if grounding_raw is None
                or isinstance(grounding_raw, ToolGroundingConfig)
                else ToolGroundingConfig(**grounding_raw)
            ),
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
        if self.grounding is not None and not isinstance(
            self.grounding, ToolGroundingConfig
        ):
            self.grounding = ToolGroundingConfig(**self.grounding)

    def to_llm_schema(self) -> dict[str, Any]:
        """Export a provider-agnostic schema for LLM tool registration."""
        schema: dict[str, Any] = {
            "name": self.id,
            "display_name": self.name,
            "description": self.description,
            "parameters": self.parameters.to_dict(),
        }
        if self.output_schema is not None:
            schema["output_schema"] = self.output_schema.to_dict()
        return schema

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate call-time arguments against this tool's ``parameters`` schema.

        Raises:
            ToolArgumentError: If ``arguments`` do not conform.
        """
        self.parameters.validate(arguments, path="arguments")


@dataclass
class ToolResult(BaseParams):
    """Result returned by an environment step."""

    output: str | dict[str, Any]
    updated_state: dict[str, Any] | None = None
