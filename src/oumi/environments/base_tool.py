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

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams


class ToolError(Exception):
    """Base class for tool errors surfaced back to the LLM.

    Subclasses of this exception are caught by the tool-call loop and
    re-emitted as structured `tool` messages so the model can self-correct
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
class GroundingConfig(BaseParams):
    """Per-environment grounding configuration.

    When set on an environment, the ConversationSynthesizer samples facts from
    that environment and injects them into the planner prompt so turn plans
    reference real entities rather than hallucinated IDs.
    """

    sample_size: int = 3
    """Number of grounding facts sampled per conversation."""

    seed: int | None = None
    """If set, per-sample RNG is seeded from ``(seed + sample_index)`` for
    reproducibility. If None, grounding uses an unseeded ``random.Random``."""

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError(
                f"{type(self).__name__}.sample_size must be >= 1, "
                f"got {self.sample_size}."
            )


def _format_grounding_value(value: Any) -> str:
    """Render a fact value as a quoted string or bare literal."""
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def describe_grounding_default(facts: list[DeterministicToolOutput]) -> str:
    """Render grounding facts as a bulleted markdown block.

    Each fact's ``input`` and ``output`` dicts are flattened into a single
    key=value line. Output values win on key collisions.
    Returns "" for an empty fact list.
    """
    if not facts:
        return ""
    lines: list[str] = []
    for fact in facts:
        merged = {**fact.input, **fact.output}
        parts = [
            f"{key}={_format_grounding_value(value)}"
            for key, value in merged.items()
        ]
        lines.append(f"- {', '.join(parts)}")
    return "\n".join(lines)


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
            items=(
                cls.create(raw["items"]) if raw.get("items") is not None else None
            ),
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
class Tool(BaseParams):
    """Tool schema owned by an environment."""

    id: str
    name: str
    description: str
    parameters: ToolSchema = field(default_factory=ToolSchema)
    output_schema: ToolSchema | None = None
    read_only: bool = True
    deterministic_outputs: list[DeterministicToolOutput] = field(default_factory=list)

    @classmethod
    def create(cls, raw: Any) -> Tool:
        """Create a tool from raw config data."""
        if isinstance(raw, Tool):
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
            deterministic_outputs=[
                entry
                if isinstance(entry, DeterministicToolOutput)
                else DeterministicToolOutput(**entry)
                for entry in raw.get("deterministic_outputs", [])
            ],
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
        self.deterministic_outputs = [
            entry
            if isinstance(entry, DeterministicToolOutput)
            else DeterministicToolOutput(**entry)
            for entry in self.deterministic_outputs
        ]

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

    output: str | dict[str, Any] | None
    updated_state: dict[str, Any] | None = None
