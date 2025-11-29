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

"""Data structures for function/tool calling in LLM APIs."""

from enum import Enum
from typing import Any, Optional

import pydantic


class ToolType(str, Enum):
    """Type of tool available to the model."""

    FUNCTION = "function"
    """A callable function that the model can invoke."""

    def __str__(self) -> str:
        """Return the string representation of the ToolType enum."""
        return self.value


class ToolChoiceType(str, Enum):
    """Controls which (if any) tool is called by the model."""

    AUTO = "auto"
    """Let the model decide whether to call a tool or not."""

    NONE = "none"
    """The model will not call any tools."""

    REQUIRED = "required"
    """The model must call one or more tools."""

    def __str__(self) -> str:
        """Return the string representation of the ToolChoiceType enum."""
        return self.value


class FunctionDefinition(pydantic.BaseModel):
    """Definition of a function that can be called by the model."""

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    """The name of the function to be called.

    Must be a-z, A-Z, 0-9, underscores and dashes, with a maximum length of 64."""

    description: Optional[str] = None
    """A description of what the function does.

    This is used by the model to choose when and how to call the function."""

    parameters: Optional[dict[str, Any]] = None
    """The parameters the function accepts, described as a JSON Schema object.

    See the JSON Schema reference for documentation about the format:
    https://json-schema.org/understanding-json-schema/

    To describe a function that accepts no parameters, provide the value:
    ```json
    {"type": "object", "properties": {}}
    ```
    """

    strict: Optional[bool] = None
    """Whether to enable strict schema adherence when generating the function call.

    If set to true, the model will follow the exact schema provided in parameters.
    Only supported by OpenAI models (gpt-4o and later).
    """


class ToolDefinition(pydantic.BaseModel):
    """Definition of a tool available to the model."""

    model_config = pydantic.ConfigDict(frozen=True)

    type: ToolType
    """The type of the tool. Currently, only 'function' is supported."""

    function: FunctionDefinition
    """The function definition."""


class FunctionCall(pydantic.BaseModel):
    """A function call made by the model."""

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    """The name of the function being called."""

    arguments: str
    """The arguments to call the function with, as a JSON string."""


class ToolCall(pydantic.BaseModel):
    """A tool call made by the model."""

    model_config = pydantic.ConfigDict(frozen=True)

    id: str
    """The ID of the tool call.

    This ID is used to match tool responses to tool calls."""

    type: ToolType
    """The type of tool call."""

    function: FunctionCall
    """The function that the model called."""


class ToolChoice(pydantic.BaseModel):
    """Controls which (if any) tool is called by the model."""

    model_config = pydantic.ConfigDict(frozen=True, use_enum_values=True)

    type: ToolChoiceType = ToolChoiceType.AUTO
    """Controls whether the model calls a tool."""

    function_name: Optional[str] = None
    """The name of the specific function to call (when type is a specific function)."""

    def to_api_format(self) -> Any:
        """Convert to API format.

        Returns:
            - "auto", "none", or "required" for those types
            - {"type": "function", "function": {"name": "..."}} for specific function
        """
        if self.type in (
            ToolChoiceType.AUTO,
            ToolChoiceType.NONE,
            ToolChoiceType.REQUIRED,
        ):
            return self.type.value
        # For specific function choice
        if self.function_name:
            return {"type": "function", "function": {"name": self.function_name}}
        return self.type.value
