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

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.tool_params import ToolEnvironmentAttribute, ToolAttribute
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json

_MAX_STATE_UPDATE_RETRIES = 2
"""Maximum retries for state updates that fail schema validation."""


class GeneratedToolEnvironment:
    """LLM-powered stateful environment for tool synthesis.

    Maintains a JSON state document that evolves as tools read from and
    write to it. Each tool call is processed via two LLM calls:
    1. _generate_result: produces the tool's output given current state
    2. _update_state: updates state to reflect the tool call (if not read_only)

    State is validated against a JSON Schema after each update.
    """

    def __init__(
        self,
        config: ToolEnvironmentAttribute,
        inference_engine: Any,
        inference_config: InferenceConfig | None = None,
    ):
        """Initialize the environment with config and inference engine."""
        self._config = config
        self._inference_engine = inference_engine
        self._inference_config = inference_config
        self._state: dict[str, Any] = copy.deepcopy(config.initial_state or {})
        self._state_schema: dict[str, Any] | None = (
            copy.deepcopy(config.state_schema) if config.state_schema else None
        )

    @property
    def state(self) -> dict[str, Any]:
        """Current state of the environment."""
        return self._state

    def step(self, tool: ToolAttribute, arguments: dict[str, Any]) -> str:
        """Execute a tool call against this environment.

        Args:
            tool: The tool being called. Uses tool.read_only to determine
                whether state is updated after the call.
            arguments: The arguments passed to the tool.

        Returns:
            The tool result as a string.
        """
        result = self._generate_result(tool, arguments)

        if not tool.read_only:
            for attempt in range(_MAX_STATE_UPDATE_RETRIES):
                new_state = self._update_state(
                    tool, arguments, result, retry=(attempt > 0)
                )
                if self._validate_state(new_state):
                    self._state = new_state
                    break
            else:
                logger.warning(
                    f"State validation failed {_MAX_STATE_UPDATE_RETRIES} times "
                    f"for tool '{tool.name}'. Keeping previous state."
                )

        return result

    def reset(self) -> None:
        """Reset to initial state."""
        self._state = copy.deepcopy(self._config.initial_state or {})

    def initialize(
        self,
        tools: list[ToolAttribute],
        scenario_context: str | None = None,
    ) -> None:
        """Generate state_schema and/or initial_state if not provided.

        Args:
            tools: All tools bound to this environment.
            scenario_context: Optional conversation persona/scenario context.
        """
        if self._state_schema is None:
            self._state_schema = self._generate_schema(tools, scenario_context)

        if not self._state:
            self._state = self._generate_initial_state(scenario_context)
            if not self._validate_state(self._state):
                # Retry up to 2 times
                for _ in range(_MAX_STATE_UPDATE_RETRIES - 1):
                    self._state = self._generate_initial_state(scenario_context)
                    if self._validate_state(self._state):
                        break
                else:
                    logger.warning(
                        f"Initial state generation failed validation "
                        f"for environment '{self._config.id}'. Using empty dict."
                    )
                    self._state = {}
        elif self._state_schema:
            if not self._validate_state(self._state):
                raise ValueError(
                    f"Initial state for environment '{self._config.id}' "
                    f"does not conform to the provided state schema."
                )

    def _generate_result(self, tool: ToolAttribute, arguments: dict[str, Any]) -> str:
        """LLM call: given current state + tool call, produce tool output."""
        system_parts = [
            self._config.system_prompt,
            f"\nCurrent state:\n{json.dumps(self._state, indent=2)}",
        ]

        user_parts = [
            f"Tool '{tool.name}' called with arguments:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            "Given the current state, produce the tool's output. "
            "Output ONLY valid JSON.",
        ]
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
        conversation = Conversation(messages=messages)

        results = self._inference_engine.infer(
            [conversation], inference_config=self._inference_config
        )
        return self._extract_text(results[0])

    def _update_state(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        result: str,
        retry: bool = False,
    ) -> dict[str, Any]:
        """LLM call: given state + tool call + result, produce new state."""
        system_parts = [
            self._config.system_prompt,
        ]
        if self._state_schema:
            system_parts.append(
                f"\nState schema:\n{json.dumps(self._state_schema, indent=2)}"
            )

        user_parts = [
            f"Current state:\n{json.dumps(self._state, indent=2)}\n\n"
            f"Tool '{tool.name}' was called with:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            f"Tool returned:\n{result}\n\n"
            "Update the state to reflect this tool call. "
            "Output ONLY valid JSON conforming to the state schema.",
        ]
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous state update was invalid. "
                "Ensure the output is valid JSON and conforms to the schema."
            )

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        conversation = Conversation(messages=messages)

        results = self._inference_engine.infer(
            [conversation], inference_config=self._inference_config
        )
        text = self._extract_text(results[0])

        parsed = extract_json(text, expected_type=dict)
        if isinstance(parsed, dict):
            return parsed

        logger.warning(f"Failed to parse state update JSON: {text[:200]}")
        return self._state

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

    def _generate_schema(
        self,
        tools: list[ToolAttribute],
        scenario_context: str | None = None,
    ) -> dict[str, Any]:
        """LLM call: generate a JSON Schema from tool definitions."""
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
            "Every field that any tool reads or writes must be represented. "
            "Output ONLY a valid JSON Schema object.",
        ]
        if scenario_context:
            user_parts.insert(0, f"Scenario context: {scenario_context}\n\n")

        messages = [
            Message(role=Role.SYSTEM, content=system_msg),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        results = self._inference_engine.infer(
            [Conversation(messages=messages)],
            inference_config=self._inference_config,
        )
        text = self._extract_text(results[0])

        parsed = extract_json(text, expected_type=dict)
        if isinstance(parsed, dict):
            return parsed

        logger.warning(
            f"Failed to generate valid schema for '{self._config.id}'. "
            "Using permissive schema."
        )
        return {"type": "object"}

    def _generate_initial_state(
        self, scenario_context: str | None = None
    ) -> dict[str, Any]:
        """LLM call: generate initial state from schema."""
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
        results = self._inference_engine.infer(
            [Conversation(messages=messages)],
            inference_config=self._inference_config,
        )
        text = self._extract_text(results[0])

        parsed = extract_json(text, expected_type=dict)
        if isinstance(parsed, dict):
            return parsed

        logger.warning(
            f"Failed to generate valid initial state for '{self._config.id}'."
        )
        return {}

    @staticmethod
    def _extract_text(conversation: Conversation) -> str:
        """Extract text content from the last message of a conversation."""
        last_msg = conversation.messages[-1]
        content = last_msg.content
        if isinstance(content, str):
            return content
        return ""
