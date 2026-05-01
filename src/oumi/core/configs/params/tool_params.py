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
from oumi.core.types.tool_call import (
    FunctionDefinition,
    JSONSchema,
    ToolDefinition,
)


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


def validate_arguments_against_schema(
    arguments: Any, schema: JSONSchema, *, path: str = "arguments"
) -> None:
    """Validate tool-call arguments against a ``JSONSchema``.

    Raises:
        ToolArgumentError: If ``arguments`` do not conform.
    """
    try:
        _validate_json_schema_value(
            arguments,
            schema.model_dump(mode="json", exclude_none=True),
            path=path,
        )
    except ValueError as e:
        raise ToolArgumentError(str(e)) from e


def _coerce_json_schema(raw: Any) -> JSONSchema:
    """Coerce a mapping, ``JSONSchema``, or ``None`` into a ``JSONSchema``.

    ``None`` resolves to ``{"type": "object"}`` — OmegaConf strips Pydantic
    defaults when round-tripping through YAML, so loaded configs may surface
    ``parameters=None``.
    """
    if raw is None:
        return JSONSchema(type="object")
    if isinstance(raw, JSONSchema):
        return raw
    if isinstance(raw, Mapping):
        return JSONSchema.model_validate(dict(raw))
    raise TypeError(
        f"Tool schema definitions must be JSONSchema or mappings, got {type(raw)}"
    )


@dataclass
class ToolParams(BaseParams):
    """Tool schema owned by an environment.

    ``parameters`` and ``output_schema`` are typed as ``Any`` so OmegaConf
    carries them as plain dicts through YAML; ``__post_init__`` coerces to
    ``JSONSchema``.
    """

    id: str
    name: str
    description: str
    parameters: Any = field(default_factory=lambda: {"type": "object"})
    output_schema: Any = None
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
            parameters=_coerce_json_schema(raw.get("parameters", {"type": "object"})),
            output_schema=(
                _coerce_json_schema(raw["output_schema"])
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
        if not isinstance(self.parameters, JSONSchema):
            self.parameters = _coerce_json_schema(self.parameters)
        if self.output_schema is not None and not isinstance(
            self.output_schema, JSONSchema
        ):
            self.output_schema = _coerce_json_schema(self.output_schema)
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
            "parameters": self.parameters.model_dump(mode="json", exclude_none=True),
        }
        if self.output_schema is not None:
            schema["output_schema"] = self.output_schema.model_dump(
                mode="json", exclude_none=True
            )
        return schema

    def to_tool_definition(self) -> ToolDefinition:
        """Project to OpenAI-wire-format ``ToolDefinition``.

        Drops chain-internal fields (``output_schema``, ``read_only``,
        ``name`` display label) that have no slot in the OpenAI contract.
        """
        return ToolDefinition(
            function=FunctionDefinition(
                name=self.id,
                description=self.description,
                parameters=self.parameters,
            ),
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate call-time arguments against this tool's ``parameters`` schema.

        Raises:
            ToolArgumentError: If ``arguments`` do not conform.
        """
        validate_arguments_against_schema(arguments, self.parameters)
