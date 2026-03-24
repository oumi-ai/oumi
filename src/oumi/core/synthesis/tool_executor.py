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

"""Tool executor for agentic synthesis."""

import json
import random
import re
from dataclasses import dataclass
from typing import Any

import jsonschema

from oumi.core.configs.params.tool_params import (
    DeterministicToolOutput,
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN_PATTERN = re.compile(r"<tool_call>\s*(.*)", re.DOTALL)
_TOOL_TAG_PATTERN = re.compile(r"</?tool_call>")
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")


def _parse_tool_json(raw: str) -> dict[str, Any] | None:
    """Extract and parse JSON from raw tool call content."""
    result = extract_json(raw, expected_type=dict)
    if isinstance(result, dict):
        return result
    stripped = raw.rstrip()
    for _ in range(3):
        if stripped.endswith("}"):
            stripped = stripped[:-1]
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    cleaned = _TRAILING_COMMA_PATTERN.sub(r"\1", raw)
    if cleaned != raw:
        result = extract_json(cleaned, expected_type=dict)
        if isinstance(result, dict):
            return result
    return None


_MAX_CONTEXT_MESSAGES = 20
_MAX_CONTEXT_CHARS_PER_MESSAGE = 500


@dataclass
class ToolCallParsed:
    """Successfully parsed and validated tool call."""

    tool_call: dict[str, Any]


@dataclass
class ToolCallError:
    """Structured error from parsing or validation."""

    error_json: str
    tool_name: str | None


ToolCallResult = ToolCallParsed | ToolCallError | None


class ToolExecutor:
    """Parses tool calls from LLM responses and resolves tool outputs."""

    def __init__(self, tools: list[ToolAttribute]):
        """Initialize the tool executor with a list of available tools."""
        self._tools_by_name: dict[str, ToolAttribute] = {t.name: t for t in tools}

    def get_tool_by_name(self, name: str) -> ToolAttribute | None:
        """Look up a tool by its display name."""
        return self._tools_by_name.get(name)

    def parse_and_validate_tool_call(self, response: str) -> ToolCallResult:
        """Parse <tool_call> tags from response and validate arguments."""
        match = _TOOL_CALL_PATTERN.search(response)
        if not match:
            match = _TOOL_CALL_OPEN_PATTERN.search(response)
        if not match:
            return None

        raw = match.group(1).strip()
        close_idx = raw.find("</tool_call>")
        if close_idx != -1:
            raw = raw[:close_idx].strip()
        parsed = _parse_tool_json(raw)
        if parsed is None:
            return ToolCallError(
                error_json=json.dumps(
                    {
                        "error": "malformed_json",
                        "message": "Tool call JSON could not be parsed.",
                    }
                ),
                tool_name=None,
            )

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})

        if not name or not isinstance(name, str):
            return ToolCallError(
                error_json=json.dumps(
                    {
                        "error": "malformed_json",
                        "message": "Tool call missing or invalid 'name' field.",
                    }
                ),
                tool_name=None,
            )

        if not isinstance(arguments, dict):
            return ToolCallError(
                error_json=json.dumps(
                    {
                        "error": "malformed_json",
                        "message": f"Tool call 'arguments' must be a dict, got {type(arguments).__name__}.",
                        "tool": name,
                    }
                ),
                tool_name=name,
            )

        if name not in self._tools_by_name:
            return ToolCallError(
                error_json=json.dumps(
                    {
                        "error": "unknown_tool",
                        "message": f"Tool '{name}' not found.",
                        "available_tools": sorted(self._tools_by_name.keys()),
                    }
                ),
                tool_name=name,
            )

        tool = self._tools_by_name[name]
        if tool.parameters:
            try:
                jsonschema.validate(instance=arguments, schema=tool.parameters)
            except jsonschema.ValidationError as e:
                return ToolCallError(
                    error_json=json.dumps(
                        {
                            "error": "invalid_arguments",
                            "message": e.message,
                            "tool": name,
                        }
                    ),
                    tool_name=name,
                )

        return ToolCallParsed(tool_call={"name": name, "arguments": arguments})

    def parse_tool_call(self, response: str) -> dict[str, Any] | None:
        """Deprecated: use parse_and_validate_tool_call instead."""
        result = self.parse_and_validate_tool_call(response)
        if isinstance(result, ToolCallParsed):
            return result.tool_call
        return None

    @staticmethod
    def strip_tool_tags(text: str) -> str:
        """Remove any residual <tool_call> or </tool_call> tags from text."""
        return _TOOL_TAG_PATTERN.sub("", text).strip()

    @staticmethod
    def strip_bare_tool_json(text: str) -> str:
        """Remove bare JSON objects that look like tool calls from text."""
        if not text:
            return text

        removals: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            if text[i] != "{":
                i += 1
                continue
            try:
                parsed, end_idx = json.JSONDecoder().raw_decode(text, i)
            except (json.JSONDecodeError, ValueError):
                i += 1
                continue
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                removals.append((i, end_idx))
                i = end_idx
            else:
                i += 1

        result = text
        for start, end in reversed(removals):
            result = result[:start] + result[end:]

        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")
        return result.strip()

    @staticmethod
    def extract_content_around_tool_call(text: str) -> str | None:
        """Extract natural language content before/after a <tool_call> tag.

        Returns the cleaned text, or None if nothing meaningful remains.
        """
        cleaned = _TOOL_CALL_PATTERN.sub("", text)
        cleaned = _TOOL_CALL_OPEN_PATTERN.sub("", cleaned)
        cleaned = _TOOL_TAG_PATTERN.sub("", cleaned)
        cleaned = ToolExecutor.strip_bare_tool_json(cleaned)

        cleaned = cleaned.strip()
        return cleaned if cleaned else None

    def sample_deterministic_outputs(
        self, tools: list[ToolAttribute]
    ) -> dict[str, str]:
        """Sample one deterministic output per DETERMINISTIC tool.

        Called once per conversation. Returns {tool_id: output_json_string}.
        """
        selections: dict[str, str] = {}
        for tool in tools:
            if tool.output_strategy != ToolOutputStrategy.DETERMINISTIC:
                continue
            if not tool.deterministic_outputs:
                continue

            weights = [
                o.sample_rate if o.sample_rate is not None else 1.0
                for o in tool.deterministic_outputs
            ]
            selected: DeterministicToolOutput = random.choices(
                tool.deterministic_outputs, weights=weights, k=1
            )[0]
            selections[tool.id] = json.dumps(selected.values)

        return selections

    def resolve_output(
        self,
        tool_call: dict[str, Any],
        deterministic_selections: dict[str, str],
    ) -> str | None:
        """Resolve a tool call to its output.

        Returns the output string for DETERMINISTIC tools, or None for
        GENERATED tools (caller must use build_generated_simulator_prompt).
        """
        tool = self._tools_by_name.get(tool_call["name"])
        if not tool:
            logger.warning(
                f"Cannot resolve output for unknown tool: {tool_call['name']}"
            )
            return json.dumps({"error": f"Unknown tool: {tool_call['name']}"})

        if tool.output_strategy == ToolOutputStrategy.DETERMINISTIC:
            output = deterministic_selections.get(tool.id)
            if output is None:
                logger.warning(
                    f"No deterministic output pre-selected for tool '{tool.id}'"
                )
                return json.dumps(
                    {"error": f"No output available for tool '{tool.name}'"}
                )
            return output
        return None

    def build_generated_simulator_prompt(
        self,
        tool_call: dict[str, Any],
        conversation_history: list[Message] | None = None,
    ) -> Conversation:
        """Build the LLM prompt for simulating a GENERATED tool output.

        The simulator LLM receives:
        - Tool definition (name, description, parameter schema)
        - Output schema (what the response should look like)
        - User-provided instruction for how this tool should behave
        - The actual arguments the agent passed
        - Truncated conversation context for coherence
        """
        tool = self._tools_by_name[tool_call["name"]]
        assert tool.generated_output is not None

        system_parts = [
            f'You are simulating the tool "{tool.name}".',
            f"\nDescription: {tool.description}",
        ]

        if tool.parameters:
            system_parts.append(
                f"\nParameter schema:\n{json.dumps(tool.parameters, indent=2)}"
            )

        if tool.output_schema:
            system_parts.append(
                f"\nExpected output schema:\n{json.dumps(tool.output_schema, indent=2)}"
            )

        system_parts.append(f"\nBehavior: {tool.generated_output.instruction}")
        system_parts.append(
            "\nProduce a realistic JSON response matching the output schema. "
            "Output ONLY valid JSON."
        )

        messages = [Message(role=Role.SYSTEM, content="\n".join(system_parts))]

        user_parts: list[str] = []
        if conversation_history:
            context = self._truncate_history(conversation_history)
            if context:
                user_parts.append(f"Conversation so far:\n{context}\n")

        user_parts.append(
            f"The agent called {tool.name} with:\n"
            f"{json.dumps(tool_call['arguments'], indent=2)}\n\n"
            f"Generate the tool's JSON output."
        )
        messages.append(Message(role=Role.USER, content="\n".join(user_parts)))

        return Conversation(messages=messages)

    @staticmethod
    def build_capability_summary(tools: list[ToolAttribute]) -> str:
        """Build a capability summary for the conversation planner.

        Lists what the assistant can do (descriptions only) without exposing
        tool names or parameters. This gives the planner enough context to
        create feasible plans while keeping tool selection up to the agent.

        Returns an empty string if the tools list is empty.
        """
        seen: set[str] = set()
        lines: list[str] = []
        for tool in tools:
            if tool.description not in seen:
                seen.add(tool.description)
                lines.append(f"- {tool.description}")
        return "\n".join(lines)

    @staticmethod
    def build_tool_catalog(tools: list[ToolAttribute]) -> str:
        """Build a formatted tool catalog with full JSON Schema for each tool."""
        lines: list[str] = []
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                schema_str = json.dumps(tool.parameters, indent=2)
                indented = "\n".join(
                    f"  {line}" for line in schema_str.split("\n")
                )
                lines.append(f"  Parameters:\n{indented}")
        return "\n".join(lines)

    @staticmethod
    def build_tool_definitions(tools: list[ToolAttribute]) -> list[dict[str, Any]]:
        """Convert ToolAttributes to standard tool definitions for output."""
        definitions: list[dict[str, Any]] = []
        for tool in tools:
            definition: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                },
            }
            definition["function"]["parameters"] = tool.parameters or {
                "type": "object",
                "properties": {},
            }
            definitions.append(definition)
        return definitions

    @staticmethod
    def format_tool_call_message(
        tool_call: dict[str, Any],
        call_id: str,
    ) -> dict[str, Any]:
        """Format a parsed tool call as a standard OpenAI assistant message."""
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call["name"],
                        "arguments": json.dumps(tool_call["arguments"]),
                    },
                }
            ],
        }

    @staticmethod
    def format_tool_result_message(
        call_id: str,
        content: str,
        name: str,
    ) -> dict[str, Any]:
        """Format a tool result as a standard OpenAI tool message."""
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        }

    @staticmethod
    def _truncate_history(messages: list[Message]) -> str:
        """Truncate conversation history to fit context constraints."""
        recent = messages[-_MAX_CONTEXT_MESSAGES:]
        lines: list[str] = []
        for msg in recent:
            content = msg.content or ""
            if (
                isinstance(content, str)
                and len(content) > _MAX_CONTEXT_CHARS_PER_MESSAGE
            ):
                content = content[:_MAX_CONTEXT_CHARS_PER_MESSAGE] + "..."
            lines.append(f"[{msg.role.value}]: {content}")
        return "\n".join(lines)
