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

import jsonschema

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


@dataclass
class ToolParams(BaseParams):
    """Tool schema owned by an environment.

    ``parameters`` and ``output_schema`` are stored as plain JSON-Schema
    dicts so OmegaConf can carry them through YAML round-trips. They are
    converted to a Pydantic ``JSONSchema`` only at the wire-format boundary
    in :meth:`to_tool_definition`.
    """

    id: str
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    output_schema: dict[str, Any] | None = None
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
            parameters=dict(raw.get("parameters", {"type": "object"})),
            output_schema=(
                dict(raw["output_schema"])
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
        """Validate common tool fields.

        Accepts ``JSONSchema`` instances on ``parameters`` / ``output_schema``
        for callers that build a tool with Pydantic types directly; converts
        them to dicts so the canonical in-memory shape stays JSON-Schema-shaped.
        """
        if not self.id:
            raise ValueError(f"{type(self).__name__}.id cannot be empty.")
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name cannot be empty.")
        if not self.description:
            raise ValueError(f"{type(self).__name__}.description cannot be empty.")
        if isinstance(self.parameters, JSONSchema):
            self.parameters = self.parameters.model_dump(mode="json", exclude_none=True)
        if isinstance(self.output_schema, JSONSchema):
            self.output_schema = self.output_schema.model_dump(
                mode="json", exclude_none=True
            )
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
            "parameters": self.parameters,
        }
        if self.output_schema is not None:
            schema["output_schema"] = self.output_schema
        return schema

    def to_tool_definition(self) -> ToolDefinition:
        """Project to OpenAI-wire-format ``ToolDefinition``.

        Drops chain-internal fields (``output_schema``, ``read_only``,
        ``name`` display label) that have no slot in the OpenAI contract.
        Coerces ``parameters`` to ``JSONSchema`` at the boundary.
        """
        return ToolDefinition(
            function=FunctionDefinition(
                name=self.id,
                description=self.description,
                parameters=JSONSchema.model_validate(self.parameters),
            ),
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate call-time arguments against this tool's ``parameters`` schema.

        Raises:
            ToolArgumentError: If ``arguments`` do not conform.
        """
        try:
            jsonschema.validate(arguments, self.parameters)
        except jsonschema.ValidationError as e:
            raise ToolArgumentError(str(e)) from e
