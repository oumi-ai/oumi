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

from typing import Any

from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolEnvironmentAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.environment import GeneratedToolEnvironment
from oumi.core.types.conversation import Conversation, Message, Role


def _make_env_config(**overrides: Any) -> ToolEnvironmentAttribute:
    defaults: dict[str, Any] = dict(
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


def _make_env_tool(read_only: bool = True, **overrides: Any) -> ToolAttribute:
    defaults: dict[str, Any] = dict(
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


class TestSummarizeState:
    def test_summarize_state_basic(self):
        """Summary contains key names; no LLM call is made."""
        config = _make_env_config(
            initial_state={
                "files": {"a.txt": "hello", "b.txt": "world"}
            }
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
            initial_state={
                "items": [
                    {"name": "x"},
                    {"name": "y"},
                    {"name": "z"},
                ]
            },
        )
        env = GeneratedToolEnvironment(config=config)
        summary = env.summarize_state()
        assert "items" in summary
        assert "3" in summary

    def test_summarize_state_empty(self):
        """Handles empty state gracefully."""
        config = _make_env_config(
            initial_state=None, state_schema=None
        )
        env = GeneratedToolEnvironment(config=config)
        env._state = {}
        summary = env.summarize_state()
        assert isinstance(summary, str)


class TestStateUpdatePrompt:
    def test_prompt_is_eight_messages(self):
        """State update prompt has system, 3 user/assistant pairs, then actual user."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )

        assert len(conv.messages) == 8
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[1].role == Role.USER
        assert conv.messages[2].role == Role.ASSISTANT
        assert conv.messages[3].role == Role.USER

    def test_few_shot_example_contains_patch_format(self):
        """The assistant example message contains a JSON Patch array."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )

        assistant_text = conv.messages[2].content
        assert '"op"' in assistant_text
        assert '"path"' in assistant_text
        assert '"replace"' in assistant_text
        # Also verify indices 4 and 6 are assistant messages
        assert conv.messages[4].role == Role.ASSISTANT
        assert conv.messages[6].role == Role.ASSISTANT

    def test_actual_request_contains_current_state(self):
        """The actual user message includes current state and tool details."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )

        user_text = conv.messages[7].content
        assert "a.txt" in user_text
        assert "WriteFile" in user_text
        assert "JSON Patch" in user_text

    def test_actual_request_contains_dynamic_example_path(self):
        """The actual user message includes an example path from current state."""
        config = _make_env_config(
            initial_state={"files": {"readme.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "readme.txt", "content": "world"},
            result='{"status": "success"}',
        )

        user_text = conv.messages[7].content
        assert "/files/" in user_text

    def test_retry_prompt_references_json_patch(self):
        """Retry=True appends a message referencing JSON Patch format."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
            retry=True,
        )

        last_text = conv.messages[-1].content
        assert isinstance(last_text, str)
        assert "JSON Patch" in last_text or "json patch" in last_text.lower()


class TestApplyStateUpdate:
    def _make_response(self, text: str) -> Conversation:
        return Conversation(messages=[Message(role=Role.ASSISTANT, content=text)])

    def test_valid_patch_updates_state(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "old"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "replace", "path": "/files/a.txt", "value": "new"}]'
        )
        result = env.apply_state_update(response)

        assert result is True
        assert env.state["files"]["a.txt"] == "new"

    def test_add_op_updates_state(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "add", "path": "/files/b.txt", "value": "world"}]'
        )
        result = env.apply_state_update(response)

        assert result is True
        assert env.state["files"]["b.txt"] == "world"
        assert env.state["files"]["a.txt"] == "hello"

    def test_invalid_patch_returns_false(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "replace", "path": "/nonexistent/path", "value": "x"}]'
        )
        result = env.apply_state_update(response)

        assert result is False
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_malformed_json_returns_false(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response("not json at all")
        result = env.apply_state_update(response)

        assert result is False
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_schema_validation_failure_returns_false(self):
        """Patch succeeds but result violates schema -> returns False."""
        config = _make_env_config(
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
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "replace", "path": "/files/a.txt", "value": 12345}]'
        )
        result = env.apply_state_update(response)

        assert result is False
        assert env.state["files"]["a.txt"] == "hello"

    def test_empty_patch_is_valid(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response("[]")
        result = env.apply_state_update(response)

        assert result is True
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_markdown_fenced_patch(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "old"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '```json\n[{"op": "replace", "path": "/files/a.txt", "value": "new"}]\n```'
        )
        result = env.apply_state_update(response)

        assert result is True
        assert env.state["files"]["a.txt"] == "new"


class TestPublicSetters:
    def test_set_state_valid(self):
        config = _make_env_config(initial_state={"files": {}})
        env = GeneratedToolEnvironment(config=config)
        new_state = {"files": {"a.txt": "hello"}}
        result = env.set_state(new_state)
        assert result is True
        assert env.state == new_state

    def test_set_state_deep_copies(self):
        config = _make_env_config(initial_state={"files": {}})
        env = GeneratedToolEnvironment(config=config)
        new_state = {"files": {"a.txt": "hello"}}
        env.set_state(new_state)
        new_state["files"]["a.txt"] = "mutated"
        assert env.state["files"]["a.txt"] == "hello"

    def test_set_state_validates_by_default(self):
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {"files": {"type": "object"}},
                "required": ["files"],
            },
            initial_state={"files": {}},
        )
        env = GeneratedToolEnvironment(config=config)
        result = env.set_state({"wrong_key": "bad"})
        assert result is False
        assert env.state == {"files": {}}

    def test_set_state_skip_validation(self):
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {"files": {"type": "object"}},
                "required": ["files"],
            },
            initial_state={"files": {}},
        )
        env = GeneratedToolEnvironment(config=config)
        result = env.set_state({"wrong_key": "bad"}, validate=False)
        assert result is True
        assert env.state == {"wrong_key": "bad"}

    def test_set_schema(self):
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        env.set_schema(schema)
        assert env._state_schema == schema

    def test_set_schema_deep_copies(self):
        config = _make_env_config(state_schema=None, initial_state=None)
        env = GeneratedToolEnvironment(config=config)
        schema = {"type": "object"}
        env.set_schema(schema)
        schema["type"] = "array"
        assert env._state_schema is not None
        assert env._state_schema["type"] == "object"


class TestStateUpdatePromptThreeShot:
    def test_prompt_has_eight_messages(self):
        """State update prompt: system + 3 user/assistant pairs + actual user = 8."""
        config = _make_env_config(initial_state={"files": {"a.txt": "hello"}})
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")
        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )
        assert len(conv.messages) == 8
        assert conv.messages[0].role == Role.SYSTEM
        for i in range(1, 7, 2):
            assert conv.messages[i].role == Role.USER
            assert conv.messages[i + 1].role == Role.ASSISTANT
        assert conv.messages[7].role == Role.USER

    def test_examples_cover_add_replace_remove(self):
        """The three examples demonstrate replace, add, and remove operations."""
        config = _make_env_config(initial_state={"files": {"a.txt": "hello"}})
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")
        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )
        assistant_texts = [str(conv.messages[i].content) for i in range(2, 7, 2)]
        all_examples = " ".join(assistant_texts)
        assert '"replace"' in all_examples
        assert '"add"' in all_examples
        assert '"remove"' in all_examples
