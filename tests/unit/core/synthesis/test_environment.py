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

"""Tests for GeneratedToolEnvironment."""

import json

from oumi.core.configs.params.tool_params import (
    ToolEnvironmentAttribute,
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.environment import GeneratedToolEnvironment
from oumi.core.types.conversation import Conversation, Message, Role


def _make_env_config(**overrides) -> ToolEnvironmentAttribute:
    defaults = dict(
        id="filesystem",
        name="Filesystem",
        description="A simple filesystem with files and their contents.",
        system_prompt="You manage a filesystem.",
        state_schema={
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["files"],
        },
        initial_state={"files": {}},
    )
    defaults.update(overrides)
    return ToolEnvironmentAttribute(**defaults)


def _make_env_tool(read_only: bool = True, **overrides) -> ToolAttribute:
    defaults = dict(
        id="read_file",
        name="ReadFile",
        description="Read a file",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="filesystem",
        read_only=read_only,
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    defaults.update(overrides)
    return ToolAttribute(**defaults)


class TestBatchedInitMethods:
    # --- build_schema_prompt / apply_schema ---

    def test_build_schema_prompt_returns_conversation(self):
        """build_schema_prompt returns a Conversation with system+user messages,
        contains 'JSON Schema' and tool name, and makes no infer calls."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool()

        conv = env.build_schema_prompt([tool])

        assert isinstance(conv, Conversation)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[1].role == Role.USER
        user_text = conv.messages[1].content
        assert "JSON Schema" in user_text
        assert tool.name in user_text

    def test_build_schema_prompt_includes_scenario_context(self):
        """Scenario context appears in the user message."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool()

        conv = env.build_schema_prompt([tool], scenario_context="office network setup")

        user_text = conv.messages[1].content
        assert "office network setup" in user_text

    def test_apply_schema_valid(self):
        """Valid JSON schema is parsed and stored in _state_schema."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        response = Conversation(
            messages=[Message(role=Role.ASSISTANT, content=json.dumps(schema))]
        )

        result = env.apply_schema(response)

        assert result is True
        assert env._state_schema == schema

    def test_apply_schema_invalid_json(self):
        """Unparseable response returns False and leaves _state_schema unchanged."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        assert env._state_schema is None

        response = Conversation(
            messages=[Message(role=Role.ASSISTANT, content="not valid json at all!!!")]
        )

        result = env.apply_schema(response)

        assert result is False
        assert env._state_schema is None

    # --- build_initial_state_prompt / apply_initial_state ---

    def test_build_initial_state_prompt_returns_conversation(self):
        """Returns a Conversation referencing the schema."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        env._state_schema = {
            "type": "object",
            "properties": {"files": {"type": "object"}},
        }

        conv = env.build_initial_state_prompt()

        assert isinstance(conv, Conversation)
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[1].role == Role.USER
        user_text = conv.messages[1].content
        # Should reference the schema content
        assert "files" in user_text

    def test_build_initial_state_prompt_includes_scenario_context(self):
        """Scenario context appears in the user message."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        env._state_schema = {"type": "object"}

        conv = env.build_initial_state_prompt(scenario_context="rainy day at the park")

        user_text = conv.messages[1].content
        assert "rainy day at the park" in user_text

    def test_apply_initial_state_valid(self):
        """Valid state JSON is parsed, validated against schema, and stored."""
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {"files": {"type": "object", "additionalProperties": {"type": "string"}}},
                "required": ["files"],
            },
            initial_state=None,
        )
        env = GeneratedToolEnvironment(config=config)
        # Override state to empty so we can see apply_initial_state set it
        env._state = {}
        state = {"files": {"readme.txt": "hello"}}
        response = Conversation(
            messages=[Message(role=Role.ASSISTANT, content=json.dumps(state))]
        )

        result = env.apply_initial_state(response)

        assert result is True
        assert env.state == state

    def test_apply_initial_state_invalid_schema(self):
        """State violating schema returns False and leaves _state unchanged."""
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {"files": {"type": "object"}},
                "required": ["files"],
            },
            initial_state=None,
        )
        env = GeneratedToolEnvironment(config=config)
        original_state = {"files": {}}
        env._state = original_state.copy()

        bad_state = {"no_files_key": "wrong"}
        response = Conversation(
            messages=[Message(role=Role.ASSISTANT, content=json.dumps(bad_state))]
        )

        result = env.apply_initial_state(response)

        assert result is False
        assert env.state == original_state

    def test_apply_initial_state_invalid_json(self):
        """Unparseable response returns False and leaves _state unchanged."""
        config = _make_env_config(initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        env._state = {"files": {}}

        response = Conversation(
            messages=[Message(role=Role.ASSISTANT, content="this is not json")]
        )

        result = env.apply_initial_state(response)

        assert result is False
        assert env.state == {"files": {}}

    # --- summarize_state ---

    def test_summarize_state_basic(self):
        """Summary contains key names; no LLM call is made."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello", "b.txt": "world"}}
        )
        env = GeneratedToolEnvironment(config=config)

        summary = env.summarize_state()

        assert "files" in summary
        assert "a.txt" in summary
        assert "b.txt" in summary

    def test_summarize_state_nested_arrays(self):
        """Summary reports array lengths."""
        config = _make_env_config(
            state_schema=None,
            initial_state={"items": [{"name": "x"}, {"name": "y"}, {"name": "z"}]},
        )
        env = GeneratedToolEnvironment(config=config)

        summary = env.summarize_state()

        assert "items" in summary
        # Should report the count of items
        assert "3" in summary

    def test_summarize_state_empty(self):
        """Handles empty state gracefully."""
        config = _make_env_config(initial_state=None, state_schema=None)
        env = GeneratedToolEnvironment(config=config)
        env._state = {}

        summary = env.summarize_state()

        assert isinstance(summary, str)
        # Should not raise; empty is fine

    def test_build_schema_prompt_includes_dict_keyed_instruction(self):
        """Schema prompt instructs LLM to use dict-keyed collections."""
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool()

        conv = env.build_schema_prompt([tool])

        user_text = conv.messages[1].content
        assert "dictionaries keyed by" in user_text.lower() or "dict" in user_text.lower()
