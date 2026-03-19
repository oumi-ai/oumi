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
from unittest.mock import MagicMock

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


def _mock_inference_engine(responses: list[str]) -> MagicMock:
    """Create a mock inference engine that returns given responses in sequence."""
    engine = MagicMock()
    call_count = 0

    def infer_side_effect(conversations, **kwargs):
        nonlocal call_count
        results = []
        for _ in conversations:
            text = responses[call_count] if call_count < len(responses) else "{}"
            call_count += 1
            results.append(
                Conversation(messages=[Message(role=Role.ASSISTANT, content=text)])
            )
        return results

    engine.infer.side_effect = infer_side_effect
    return engine


class TestGeneratedToolEnvironmentStep:
    def test_step_read_only_returns_result_without_state_change(self):
        """Read-only step should return result but not modify state."""
        config = _make_env_config(initial_state={"files": {"notes.txt": "hello"}})
        engine = _mock_inference_engine([
            '{"content": "hello"}',  # _generate_result
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)
        tool = _make_env_tool(read_only=True)

        result = env.step(tool, arguments={"path": "notes.txt"})

        assert result == '{"content": "hello"}'
        assert env.state == {"files": {"notes.txt": "hello"}}
        # Only 1 LLM call (generate_result), no update_state
        assert engine.infer.call_count == 1

    def test_step_mutating_returns_result_and_updates_state(self):
        """Non-read-only step should return result AND update state."""
        config = _make_env_config(initial_state={"files": {}})
        new_state = {"files": {"notes.txt": "hello"}}
        engine = _mock_inference_engine([
            '"File created successfully."',   # _generate_result
            json.dumps(new_state),             # _update_state
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)
        tool = _make_env_tool(
            id="create_file",
            name="CreateFile",
            description="Create a file",
            read_only=False,
        )

        result = env.step(
            tool, arguments={"path": "notes.txt", "content": "hello"}
        )

        assert result == '"File created successfully."'
        assert env.state == new_state
        # 2 LLM calls: generate_result + update_state
        assert engine.infer.call_count == 2

    def test_step_state_validation_failure_retries(self):
        """Invalid state from _update_state should retry."""
        config = _make_env_config(initial_state={"files": {}})
        valid_state = {"files": {"a.txt": "x"}}
        engine = _mock_inference_engine([
            '"Created."',               # _generate_result
            '{"bad": "no files key"}',  # _update_state attempt 1 (invalid)
            json.dumps(valid_state),    # _update_state attempt 2 (valid)
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)
        tool = _make_env_tool(
            read_only=False, id="create", name="Create", description="c"
        )

        env.step(tool, arguments={"path": "a.txt", "content": "x"})

        assert env.state == valid_state
        assert engine.infer.call_count == 3  # result + bad state + good state

    def test_step_state_validation_failure_twice_keeps_old_state(self):
        """Two invalid state updates should keep old state."""
        old_state = {"files": {}}
        config = _make_env_config(initial_state=old_state)
        engine = _mock_inference_engine([
            '"Created."',               # _generate_result
            '{"bad": 1}',              # _update_state attempt 1
            '{"also_bad": 2}',         # _update_state attempt 2
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)
        tool = _make_env_tool(
            read_only=False, id="create", name="Create", description="c"
        )

        env.step(tool, arguments={"path": "a.txt", "content": "x"})

        assert env.state == old_state  # unchanged


class TestGeneratedToolEnvironmentReset:
    def test_reset_restores_initial_state(self):
        config = _make_env_config(initial_state={"files": {"a.txt": "hello"}})
        new_state = {"files": {"a.txt": "hello", "b.txt": "world"}}
        engine = _mock_inference_engine([
            '"Created."',
            json.dumps(new_state),
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)
        tool = _make_env_tool(
            read_only=False, id="create", name="Create", description="c"
        )

        env.step(tool, arguments={"path": "b.txt", "content": "world"})
        assert env.state == new_state

        env.reset()
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_reset_is_deep_copy(self):
        """Resetting should produce an independent copy."""
        config = _make_env_config(initial_state={"files": {}})
        engine = _mock_inference_engine([])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)

        env.reset()
        env._state["files"]["hack.txt"] = "oops"
        env.reset()
        assert env.state == {"files": {}}


class TestGeneratedToolEnvironmentInitialize:
    def test_initialize_generates_schema_and_state(self):
        """When both are None, initialize should generate both."""
        config = _make_env_config(state_schema=None, initial_state=None)
        schema = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["files"],
        }
        state = {"files": {}}
        engine = _mock_inference_engine([
            json.dumps(schema),  # schema generation
            json.dumps(state),   # initial state generation
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)

        tools = [_make_env_tool(read_only=True)]
        env.initialize(tools)

        assert env._state_schema == schema
        assert env.state == state
        assert engine.infer.call_count == 2

    def test_initialize_generates_only_state_when_schema_provided(self):
        """When schema is provided but initial_state is None, only generate state."""
        schema = {
            "type": "object",
            "properties": {"files": {"type": "object"}},
            "required": ["files"],
        }
        config = _make_env_config(state_schema=schema, initial_state=None)
        state = {"files": {}}
        engine = _mock_inference_engine([
            json.dumps(state),  # only initial state generation
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)

        env.initialize([_make_env_tool()])

        assert env._state_schema == schema
        assert env.state == state
        assert engine.infer.call_count == 1

    def test_initialize_validates_when_both_provided(self):
        """When both are provided, initialize should validate state against schema."""
        config = _make_env_config()  # has valid schema and state
        engine = _mock_inference_engine([])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)

        env.initialize([_make_env_tool()])

        # No LLM calls needed, just validation
        assert engine.infer.call_count == 0

    def test_initialize_raises_when_initial_state_violates_schema(self):
        """When initial_state doesn't match schema, initialize should raise."""
        config = _make_env_config(initial_state={"bad_key": "no files"})
        engine = _mock_inference_engine([])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)

        import pytest

        with pytest.raises(ValueError, match="does not conform"):
            env.initialize([_make_env_tool()])

    def test_initialize_with_scenario_context(self):
        """Scenario context should be included in LLM prompts."""
        config = _make_env_config(state_schema=None, initial_state=None)
        schema = {
            "type": "object",
            "properties": {"files": {"type": "object"}},
            "required": ["files"],
        }
        engine = _mock_inference_engine([
            json.dumps(schema),
            json.dumps({"files": {}}),
        ])
        env = GeneratedToolEnvironment(config=config, inference_engine=engine)

        env.initialize(
            [_make_env_tool()],
            scenario_context="Tenant calling about a broken window.",
        )

        # Verify scenario context was included in the prompt
        first_call_args = engine.infer.call_args_list[0]
        prompt_conversations = first_call_args[0][0]
        prompt_text = prompt_conversations[0].messages[-1].content
        assert "broken window" in prompt_text
