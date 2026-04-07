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

from dataclasses import dataclass
from typing import Any

from oumi.core.configs.params.tool_params import ToolAttribute
from oumi.core.types.conversation import Conversation, Message


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
        """Initialize the tool executor with available tools."""
        raise NotImplementedError

    def get_tool_by_name(self, name: str) -> ToolAttribute | None:
        """Look up a tool by its display name."""
        raise NotImplementedError

    def parse_and_validate_tool_call(self, response: str) -> ToolCallResult:
        """Parse <tool_call> tags from response and validate arguments."""
        raise NotImplementedError

    def sample_deterministic_outputs(
        self, tools: list[ToolAttribute]
    ) -> dict[str, str]:
        """Sample one deterministic output per DETERMINISTIC tool."""
        raise NotImplementedError

    def resolve_output(
        self,
        tool_call: dict[str, Any],
        deterministic_selections: dict[str, str],
    ) -> str | None:
        """Resolve a tool call to its output. None for GENERATED tools."""
        raise NotImplementedError

    def build_generated_simulator_prompt(
        self,
        tool_call: dict[str, Any],
        conversation_history: list[Message] | None = None,
    ) -> Conversation:
        """Build LLM prompt for simulating a GENERATED tool's output."""
        raise NotImplementedError

    @staticmethod
    def build_capability_summary(tools: list[ToolAttribute]) -> str:
        """Build a planner-facing capability summary."""
        raise NotImplementedError

    @staticmethod
    def build_tool_catalog(tools: list[ToolAttribute]) -> str:
        """Build a formatted tool catalog with schemas and usage examples."""
        raise NotImplementedError

    @staticmethod
    def build_tool_definitions(
        tools: list[ToolAttribute],
    ) -> list[dict[str, Any]]:
        """Convert ToolAttributes to standard tool definitions for output."""
        raise NotImplementedError

    @staticmethod
    def format_tool_call_message(
        tool_call: dict[str, Any],
        call_id: str,
    ) -> dict[str, Any]:
        """Format a parsed tool call as a standard OpenAI assistant message."""
        raise NotImplementedError

    @staticmethod
    def format_tool_result_message(
        call_id: str,
        content: str,
        name: str,
    ) -> dict[str, Any]:
        """Format a tool result as a standard OpenAI tool message."""
        raise NotImplementedError

    @staticmethod
    def strip_tool_tags(text: str) -> str:
        """Remove any residual <tool_call> or </tool_call> tags."""
        raise NotImplementedError

    @staticmethod
    def sanitize_assistant_content(text: str) -> str:
        """Strip tool-call artifacts from assistant prose before export."""
        raise NotImplementedError

    @staticmethod
    def build_tool_few_shot(tools: list[ToolAttribute]) -> list[Message]:
        """Build few-shot messages demonstrating a correct tool-call exchange."""
        raise NotImplementedError

    @staticmethod
    def build_tool_turn_info(
        current_turn: int,
        target_turns: int,
        turn_instruction: str,
        max_calls_reached: bool,
    ) -> str:
        """Build the turn-level user message for assistant tool turns."""
        raise NotImplementedError

    @staticmethod
    def build_prose_turn_info(
        current_turn: int,
        target_turns: int,
        role: str,
        turn_instruction: str,
    ) -> str:
        """Build turn-level user message for non-tool turns."""
        raise NotImplementedError

    @staticmethod
    def record_tool_result(
        idx: int,
        raw_text: str,
        tool_call: dict,
        call_id: str,
        result: str,
        turn_tool_msgs: dict[int, list[Message]],
        output_messages: list[list[dict]],
        env_state: dict | None = None,
    ) -> None:
        """Append a tool call + result to conversation history and output."""
        raise NotImplementedError
