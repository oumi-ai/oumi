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

"""Data structures for function/tool calling in LLM APIs.

These models use ``extra="allow"`` so unknown keys round-trip through
serialization unchanged. This matters for two reasons: (1) OpenAI may
add new fields to the tool wire format over time, and (2)
``transformers.utils.chat_template_utils.get_json_schema`` produces a
richer schema (e.g., a top-level ``return`` block on
``FunctionDefinition``) than the OpenAI Chat Completions API itself
defines, and chat templates may consume those extras. Dropping unknown
keys at validation time would silently lose information that
downstream code relies on.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import pydantic

# The seven primitive types defined by JSON Schema.
JSONSchemaType = Literal[
    "object", "string", "number", "integer", "boolean", "array", "null"
]


class JSONSchema(pydantic.BaseModel):
    """A JSON Schema object describing the shape of a value.

    Models the subset of JSON Schema commonly used in LLM tool
    definitions. ``extra="allow"`` lets less-common keywords
    (``$ref``, ``$defs``, ``additionalProperties``, ``anyOf``, numeric
    constraints, etc.) round-trip unchanged, matching the rest of this
    module — see the module docstring for why round-tripping matters.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")

    type: JSONSchemaType | list[JSONSchemaType] | None = None
    """JSON type(s) of this value. A list expresses a union
    (e.g., ``["string", "null"]`` for a nullable string).
    """

    description: str | None = None
    """Human-readable description, used by the model to choose values."""

    title: str | None = None
    """Short human-readable label."""

    properties: dict[str, "JSONSchema"] | None = None
    """For ``type="object"``: schema for each named property."""

    required: list[str] | None = None
    """For ``type="object"``: names of properties that must be present."""

    items: "JSONSchema | None" = None
    """For ``type="array"``: schema for array elements."""

    enum: list[Any] | None = None
    """Restricts the value to a fixed set of allowed values."""

    default: Any = None
    """Default value used when the field is omitted."""

    format: str | None = None
    """Semantic format hint (e.g., ``"date-time"``, ``"email"``)."""


class ToolType(str, Enum):
    """Type of tool available to the model."""

    FUNCTION = "function"
    """A callable function that the model can invoke."""

    def __str__(self) -> str:
        """Return the string representation of the ToolType enum."""
        return self.value


class FunctionDefinition(pydantic.BaseModel):
    """Definition of a function that can be called by the model."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")

    name: str
    """The name of the function to be called.

    Must be a-z, A-Z, 0-9, underscores and dashes, with a maximum
    length of 64.
    """

    description: str | None = None
    """A description of what the function does.

    Used by the model to choose when and how to call the function.
    """

    parameters: JSONSchema | None = None
    """The parameters the function accepts, as a JSON Schema object.

    See https://json-schema.org/understanding-json-schema/ for the
    format. To describe a function that accepts no parameters,
    provide ``{"type": "object", "properties": {}}``.
    """

    strict: bool | None = None
    """Whether to enable strict schema adherence in the generated call.

    If true, the model will follow the exact schema provided in
    ``parameters``. Only supported by OpenAI gpt-4o and later;
    ignored by other providers.
    """


class ToolDefinition(pydantic.BaseModel):
    """Definition of a tool available to the model."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")

    type: ToolType = ToolType.FUNCTION
    """The type of the tool. Currently only ``function`` is supported.

    Defaulted for ergonomics — when a second tool type lands, drop
    the default to force callers to be explicit.
    """

    function: FunctionDefinition
    """The function definition."""


class FunctionCall(pydantic.BaseModel):
    """A function call made by the model."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")

    name: str
    """The name of the function being called."""

    arguments: str
    """The arguments to call the function with, as a JSON string.

    OpenAI wire format keeps this as an unparsed JSON-encoded string
    (NOT a dict). Some providers return malformed JSON; downstream
    code is responsible for parsing and handling errors.
    """


class ToolCall(pydantic.BaseModel):
    """A tool call emitted by the model."""

    model_config = pydantic.ConfigDict(frozen=True, extra="allow")

    id: str
    """The ID of the tool call.

    Used to match a tool response message back to the call that
    requested it (via ``Message.tool_call_id``).
    """

    type: ToolType = ToolType.FUNCTION
    """The type of tool call. Defaulted; see ``ToolDefinition.type``."""

    function: FunctionCall
    """The function the model called."""


@dataclass
class ToolResult:
    """Result returned by an environment ``step()``.

    Runtime value (not an OpenAI wire-format type) — projected by the
    synthesizer into ``Message(role=TOOL, content=...)`` before output.
    ``output`` may be a string or a JSON-serializable dict; the
    synthesizer json-encodes dicts at the message boundary.
    """

    output: str | dict[str, Any]
    updated_state: dict[str, Any] | None = None
