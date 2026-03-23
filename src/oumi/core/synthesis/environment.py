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

"""Stateful environment for agentic tool synthesis."""

import copy
import json
from typing import Any

import jsonschema

from oumi.core.configs.params.tool_params import ToolAttribute, ToolEnvironmentAttribute
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.json_patch import (
    JsonPatchError,
    JsonPatchValidationError,
    apply_json_patch,
    parse_patch_response,
)
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json

_MAX_STATE_UPDATE_RETRIES = 2
"""Maximum retries for state updates that fail schema validation."""

_MAX_RESULT_RETRIES = 2
"""Maximum retries for tool results that fail JSON parsing."""


class GeneratedToolEnvironment:
    """Stateful environment for tool synthesis using a build/apply pattern.

    Maintains a JSON state document that evolves as tools read from and
    write to it. Each tool call is processed via two LLM calls:
    1. build_result_prompt / apply_result: produces the tool's output given
       current state
    2. build_state_update_prompt / apply_state_update: updates state to
       reflect the tool call (if not read_only)

    The environment does not call inference itself — it only builds prompts
    and applies responses. The caller (e.g. ConversationSynthesizer) is
    responsible for batching inference calls across environments.

    State is validated against a JSON Schema after each update.
    """

    def __init__(self, config: ToolEnvironmentAttribute):
        """Initialize the environment with config."""
        self._config = config
        self._state: dict[str, Any] = copy.deepcopy(config.initial_state or {})
        self._state_schema: dict[str, Any] | None = (
            copy.deepcopy(config.state_schema) if config.state_schema else None
        )

    @property
    def state(self) -> dict[str, Any]:
        """Current state of the environment."""
        return self._state

    def build_result_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        retry: bool = False,
    ) -> Conversation:
        """Build the prompt for generating a tool result."""
        system_parts = [
            self._config.system_prompt,
            f"\nCurrent state:\n{json.dumps(self._state, indent=2)}",
        ]

        user_parts = [
            f"Tool '{tool.name}' called with arguments:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            "Given the current state, produce the tool's output. "
            "Output ONLY valid JSON — no markdown fences, no explanation, "
            "no code blocks. The first character of your response must be "
            "{ or [.",
        ]
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous output was not valid JSON "
                "(it may have been truncated or wrapped in markdown). "
                "Output ONLY a complete, valid JSON object or array."
            )
        if tool.output_schema:
            user_parts.insert(
                0,
                f"Expected output schema:\n"
                f"{json.dumps(tool.output_schema, indent=2)}\n\n",
            )

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def build_state_update_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        result: str,
        retry: bool = False,
    ) -> Conversation:
        """Build a few-shot prompt for generating a JSON Patch state update."""
        system_parts = [
            self._config.system_prompt,
        ]
        if self._state_schema:
            system_parts.append(
                f"\nState schema:\n{json.dumps(self._state_schema, indent=2)}"
            )

        example_user = (
            'Current state:\n{"users": {"1": {"name": "Alice", "role": "admin"}, '
            '"2": {"name": "Bob", "role": "viewer"}}}\n\n'
            "Tool 'UpdateRole' was called with:\n"
            '{"user_id": "2", "new_role": "editor"}\n\n'
            'Tool returned:\n{"status": "success", "rows_affected": 1}\n\n'
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. /users/2/role), "
            "and value (for add/replace). Output ONLY the JSON array."
        )
        example_assistant = (
            '[{"op": "replace", "path": "/users/2/role", "value": "editor"}]'
        )

        example_path = self._build_example_path()

        user_parts = [
            f"Current state:\n{json.dumps(self._state, indent=2)}\n\n"
            f"Tool '{tool.name}' was called with:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            f"Tool returned:\n{result}\n\n"
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. "
            f"{example_path}), "
            "and value (for add/replace). Output ONLY the JSON array.",
        ]
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous output was not a valid JSON Patch "
                "array. Output ONLY a JSON array of RFC 6902 patch operations."
            )

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content=example_user),
            Message(role=Role.ASSISTANT, content=example_assistant),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def _build_example_path(self) -> str:
        """Build a dynamic JSON Pointer example from current state keys.

        Walks the first key at each level up to depth 2 to produce something
        like "/users/1/name". Falls back to "/key/value" if state is empty.
        """
        parts: list[str] = []
        current: Any = self._state
        for _ in range(3):
            if isinstance(current, dict) and current:
                key = next(iter(current))
                parts.append(str(key))
                current = current[key]
            else:
                break
        if not parts:
            return "/key/value"
        return "/" + "/".join(parts)

    def apply_result(self, response: Conversation) -> str:
        """Extract the tool result text from an inference response."""
        return self._extract_text(response)

    def apply_state_update(self, response: Conversation) -> bool:
        """Parse, validate, and apply a JSON Patch from an inference response.

        Returns True if the state was successfully updated, False otherwise.
        """
        text = self._extract_text(response)
        patch = parse_patch_response(text)
        if patch is None:
            logger.warning(f"Failed to parse JSON Patch: {text[:200]}")
            return False
        try:
            self._state = apply_json_patch(
                self._state, patch, schema=self._state_schema
            )
            return True
        except JsonPatchError as e:
            logger.warning(f"JSON Patch application failed: {e}")
            return False
        except JsonPatchValidationError as e:
            logger.warning(f"Patched state failed schema validation: {e}")
            return False

    def _validate_state(self, state: dict[str, Any]) -> bool:
        """Validate state against the schema. Returns True if valid."""
        if not self._state_schema:
            return True
        try:
            jsonschema.validate(instance=state, schema=self._state_schema)
            return True
        except jsonschema.ValidationError as e:
            logger.warning(f"State validation failed: {e.message}")
            return False

    def build_schema_prompt(
        self,
        tools: list[ToolAttribute],
        scenario_context: str | None = None,
    ) -> Conversation:
        """Build the prompt for generating a JSON Schema from tool definitions.

        Args:
            tools: All tools bound to this environment.
            scenario_context: Optional conversation persona/scenario context.

        Returns:
            A Conversation with system and user messages.
        """
        tool_descriptions = []
        for tool in tools:
            desc = (
                f"- {tool.name} ({tool.description})\n"
                f"  read_only: {tool.read_only}\n"
                f"  parameters: {json.dumps(tool.parameters)}"
            )
            if tool.output_schema:
                desc += f"\n  output_schema: {json.dumps(tool.output_schema)}"
            tool_descriptions.append(desc)

        system_msg = (
            "You are designing a JSON Schema for the state of an environment.\n\n"
            f"Environment: {self._config.name}\n"
            f"Description: {self._config.description}\n\n"
            f"{self._config.system_prompt}"
        )

        user_parts = [
            "The following tools operate on this environment:\n\n"
            + "\n\n".join(tool_descriptions)
            + "\n\nDesign a JSON Schema for the state. "
            "Every field that any tool reads or writes must be represented.\n\n"
            "IMPORTANT: For collections of records (e.g., tables, lists of "
            "entities), use dictionaries keyed by the record's primary "
            "identifier — NOT arrays. For example, use "
            '{"users": {"1": {...}, "2": {...}}} instead of '
            '{"users": [{...}, {...}]}. This ensures stable key-based '
            "lookups.\n\n"
            "Output ONLY a valid JSON Schema object.",
        ]
        if scenario_context:
            user_parts.insert(0, f"Scenario context: {scenario_context}\n\n")

        messages = [
            Message(role=Role.SYSTEM, content=system_msg),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def apply_schema(self, response: Conversation) -> bool:
        """Parse JSON from the response and store it as the state schema.

        Args:
            response: Inference response containing JSON schema.

        Returns:
            True if the schema was successfully parsed and stored,
            False otherwise. On failure, _state_schema is unchanged.
        """
        text = self._extract_text(response)
        parsed = extract_json(text, expected_type=dict)
        if isinstance(parsed, dict):
            self._state_schema = parsed
            return True
        logger.warning(f"Failed to parse schema JSON: {text[:200]}")
        return False

    def build_initial_state_prompt(
        self,
        scenario_context: str | None = None,
    ) -> Conversation:
        """Build the prompt for generating an initial state.

        Args:
            scenario_context: Optional conversation persona/scenario context.

        Returns:
            A Conversation with system and user messages.
        """
        system_msg = (
            f"You are initializing the state of an environment.\n\n"
            f"Environment: {self._config.name}\n"
            f"Description: {self._config.description}\n\n"
            f"{self._config.system_prompt}"
        )

        user_parts = [
            f"State schema:\n{json.dumps(self._state_schema, indent=2)}\n\n"
            "Generate a realistic initial state conforming to this schema. "
            "Output ONLY valid JSON.",
        ]
        if scenario_context:
            user_parts.insert(0, f"Scenario context: {scenario_context}\n\n")

        messages = [
            Message(role=Role.SYSTEM, content=system_msg),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def apply_initial_state(self, response: Conversation) -> bool:
        """Parse JSON from response, validate against schema, and store as state.

        Args:
            response: Inference response containing the initial state JSON.

        Returns:
            True if the state was successfully parsed, validated, and stored.
            False otherwise. On failure, _state is unchanged.
        """
        text = self._extract_text(response)
        parsed = extract_json(text, expected_type=dict)
        if not isinstance(parsed, dict):
            logger.warning(f"Failed to parse initial state JSON: {text[:200]}")
            return False
        if not self._validate_state(parsed):
            return False
        self._state = parsed
        return True

    def summarize_state(self) -> str:
        """Produce a concise rule-based text summary of the current state.

        Example output: "files: a.txt: (str), b.txt: (str)"

        Returns:
            A human-readable string summary of the current state.
        """
        if not self._state:
            return ""
        parts = []
        for key, value in self._state.items():
            child_summary = self._summarize_value(value, depth=1)
            if child_summary:
                parts.append(f"{key}: {child_summary}")
            else:
                parts.append(key)
        return ", ".join(parts)

    @staticmethod
    def _summarize_value(value: Any, depth: int, max_depth: int = 3) -> str:
        """Recursively produce a concise summary of a JSON value.

        Args:
            value: The value to summarize.
            depth: Current recursion depth.
            max_depth: Maximum recursion depth before truncating.

        Returns:
            A concise string summary of the value.
        """
        if depth > max_depth:
            return "..."

        if isinstance(value, dict):
            if not value:
                return "{}"
            parts = []
            for k, v in value.items():
                child = GeneratedToolEnvironment._summarize_value(
                    v, depth + 1, max_depth
                )
                if child:
                    parts.append(f"{k}: {child}")
                else:
                    parts.append(str(k))
            return ", ".join(parts)

        if isinstance(value, list):
            count = len(value)
            if count == 0:
                return "[]"
            first = value[0] if value else None
            if isinstance(first, dict) and first:
                keys = ", ".join(str(k) for k in first.keys())
                return f"[{count} items, keys: {keys}]"
            return f"[{count} items]"
        return f"({type(value).__name__})"

    @staticmethod
    def _extract_text(conversation: Conversation) -> str:
        """Extract text content from the last message of a conversation."""
        last_msg = conversation.messages[-1]
        content = last_msg.content
        if isinstance(content, str):
            return content
        return ""
