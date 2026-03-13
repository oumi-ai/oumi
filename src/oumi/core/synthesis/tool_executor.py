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

from oumi.core.configs.params.tool_params import (
    DeterministicToolOutput,
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

_MAX_CONTEXT_MESSAGES = 20
_MAX_CONTEXT_CHARS_PER_MESSAGE = 500


@dataclass
class ToolCallValidationError:
    """Structured error from validating tool call arguments against a schema."""

    tool_name: str
    error_type: str
    message: str
    details: dict[str, Any]


class ToolExecutor:
    """Parses tool calls from LLM responses and resolves tool outputs."""

    def __init__(self, tools: list[ToolAttribute]):
        """Initialize the tool executor with a list of available tools."""
        self._tools_by_name: dict[str, ToolAttribute] = {t.name: t for t in tools}
        self._tools_by_id: dict[str, ToolAttribute] = {t.id: t for t in tools}

    def get_tool_by_name(self, name: str) -> ToolAttribute | None:
        """Look up a tool by its display name."""
        return self._tools_by_name.get(name)

    def parse_tool_call(self, response: str) -> dict[str, Any] | None:
        """Parse <tool_call> tags from response."""
        match = _TOOL_CALL_PATTERN.search(response)
        if not match:
            return None

        raw = match.group(1)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            fixed = raw
            for _ in range(3):
                if fixed.endswith("}"):
                    fixed = fixed[:-1]
                    try:
                        parsed = json.loads(fixed)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                logger.warning(f"Failed to parse tool call JSON: {raw[:200]}")
                return None

        if not isinstance(parsed, dict):
            logger.warning(f"Tool call is not a dict: {type(parsed)}")
            return None

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})

        if not name or not isinstance(name, str):
            logger.warning(f"Tool call missing or invalid 'name': {parsed}")
            return None

        if not isinstance(arguments, dict):
            logger.warning(f"Tool call 'arguments' is not a dict: {type(arguments)}")
            return None

        if name not in self._tools_by_name:
            logger.warning(
                f"Tool call references unknown tool '{name}'. "
                f"Available: {sorted(self._tools_by_name.keys())}"
            )
            return None

        return {"name": name, "arguments": arguments}

    def validate_arguments(
        self, tool_call: dict[str, Any]
    ) -> ToolCallValidationError | None:
        """Validate tool call arguments against the tool's parameter schema.

        Returns None if valid, or a ToolCallValidationError describing the problem.
        """
        tool = self._tools_by_name.get(tool_call["name"])
        if not tool:
            return ToolCallValidationError(
                tool_name=tool_call["name"],
                error_type="unknown_tool",
                message=f"Unknown tool: {tool_call['name']}",
                details={},
            )

        if not tool.parameters:
            return None

        schema_props = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        provided = set(tool_call["arguments"].keys())

        missing = required - provided
        if missing:
            return ToolCallValidationError(
                tool_name=tool.name,
                error_type="missing_required",
                message=f"Missing required parameters: {sorted(missing)}",
                details={"missing": sorted(missing), "provided": sorted(provided)},
            )

        unknown = provided - set(schema_props.keys())
        if unknown:
            return ToolCallValidationError(
                tool_name=tool.name,
                error_type="unknown_parameters",
                message=f"Unknown parameters: {sorted(unknown)}",
                details={
                    "unknown": sorted(unknown),
                    "allowed": sorted(schema_props.keys()),
                },
            )

        return None

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
        """Build a formatted tool catalog for injection into synthesis prompts."""
        lines: list[str] = []
        for tool in tools:
            lines.append(f"- {tool.name}: {tool.description}")
            if tool.parameters:
                props = tool.parameters.get("properties", {})
                required = set(tool.parameters.get("required", []))
                for param_name, param_info in props.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = ", required" if param_name in required else ""
                    line = f"    {param_name} ({param_type}{req_marker})"
                    if param_desc:
                        line += f" - {param_desc}"
                    lines.append(line)
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
