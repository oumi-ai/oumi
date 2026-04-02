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

"""Tests for GeneratedToolEnvironment and EnvironmentRegistry."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolEnvironmentAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.environment import (
    EnvironmentRegistry,
    GeneratedToolEnvironment,
    _pluralize,
    build_collection_prompt,
    build_dependency_graph,
    build_state_generation_prompt,
    serialize_env_states,
    sort_into_waves,
    validate_collection,
)
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


def _make_write_state_response(text: str) -> Conversation:
    return Conversation(messages=[Message(role=Role.ASSISTANT, content=text)])


def test_serialize_env_states_returns_full_state_for_small_envs():
    initial_state = {
        "files": {
            "a.txt": "hello",
            "b.txt": "world",
        },
        "metadata": {"status": "ready", "locked": False},
    }
    env = GeneratedToolEnvironment(_make_env_config(initial_state=initial_state))

    serialized = serialize_env_states({"filesystem": env})
    parsed = json.loads(serialized)

    assert parsed["filesystem"]["name"] == "Filesystem"
    assert parsed["filesystem"]["state"] == initial_state


def test_write_state_update_prompt_mentions_patch_and_error():
    config = _make_env_config(initial_state={"files": {"a.txt": "hello"}})
    env = GeneratedToolEnvironment(config=config)
    tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

    conv = env.build_write_state_update_prompt(
        tool,
        arguments={"path": "a.txt", "content": "world"},
    )

    last_text = str(conv.messages[-1].content)
    assert '"patch"' in last_text
    assert '"error"' in last_text
    assert "before the tool result is generated" in last_text


def test_apply_state_update_returning_patch_success():
    config = _make_env_config(initial_state={"files": {"a.txt": "old"}})
    env = GeneratedToolEnvironment(config=config)

    response = _make_write_state_response(
        '{"patch": [{"op": "replace", "path": "/files/a.txt", '
        '"value": "new"}], "error": null}'
    )
    succeeded, patch_ops, error = env.apply_state_update_returning_patch(response)

    assert succeeded is True
    assert error is None
    assert patch_ops == [{"op": "replace", "path": "/files/a.txt", "value": "new"}]
    assert env.state["files"]["a.txt"] == "new"


def test_apply_state_update_returning_patch_rejected_keeps_state():
    config = _make_env_config(initial_state={"files": {"a.txt": "hello"}})
    env = GeneratedToolEnvironment(config=config)

    response = _make_write_state_response(
        '{"patch": [], "error": "File does not exist."}'
    )
    succeeded, patch_ops, error = env.apply_state_update_returning_patch(response)

    assert succeeded is False
    assert patch_ops == []
    assert error == "File does not exist."
    assert env.state == {"files": {"a.txt": "hello"}}


def test_write_result_prompt_includes_before_after_diff():
    config = _make_env_config(initial_state={"files": {"a.txt": "updated"}})
    env = GeneratedToolEnvironment(config=config)
    tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")
    pre_patch_state = {"files": {"a.txt": "old"}}

    conv = env.build_write_result_prompt(
        tool,
        arguments={"path": "a.txt", "content": "updated"},
        patch_ops=[{"op": "replace", "path": "/files/a.txt", "value": "updated"}],
        patch_succeeded=True,
        pre_patch_state=pre_patch_state,
    )

    last_text = str(conv.messages[-1].content)
    assert "State BEFORE the tool call" in last_text
    assert "State AFTER the tool call" in last_text
    assert '"old"' in last_text
    assert '"updated"' in last_text
    assert "SUCCESS" in last_text
    assert "Do not re-check preconditions" in last_text


def test_write_result_prompt_fallback_without_pre_patch_state():
    config = _make_env_config(initial_state={"files": {"a.txt": "updated"}})
    env = GeneratedToolEnvironment(config=config)
    tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

    conv = env.build_write_result_prompt(
        tool,
        arguments={"path": "a.txt", "content": "updated"},
        patch_ops=[{"op": "replace", "path": "/files/a.txt", "value": "updated"}],
        patch_succeeded=True,
    )

    last_text = str(conv.messages[-1].content)
    assert "Current state (after update)" in last_text
    assert "Applied state update" in last_text
    assert "SUCCESS" in last_text


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


def _make_registry_env_config(**overrides: Any) -> ToolEnvironmentAttribute:
    defaults: dict[str, Any] = dict(
        id="db",
        name="PropertyDB",
        description="A property management database.",
        system_prompt="You are a property management database.",
    )
    defaults.update(overrides)
    return ToolEnvironmentAttribute(**defaults)


def _make_tool(**overrides: Any) -> ToolAttribute:
    defaults: dict[str, Any] = dict(
        id="get_tenant",
        name="GetTenant",
        description="Look up a tenant",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
        parameters={
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
        },
    )
    defaults.update(overrides)
    return ToolAttribute(**defaults)


class TestPluralize:
    def test_regular_noun(self):
        assert _pluralize("tenant") == "tenants"

    def test_already_plural(self):
        assert _pluralize("tenants") == "tenants"

    def test_noun_ending_in_s(self):
        assert _pluralize("status") == "statuses"

    def test_noun_ending_in_y(self):
        assert _pluralize("category") == "categories"

    def test_noun_ending_in_e(self):
        assert _pluralize("file") == "files"


class TestBuildStateGenerationPrompt:
    def test_includes_environment_description(self):
        config = _make_registry_env_config()
        conv = build_state_generation_prompt(config=config, tools=[])
        system_text = conv.messages[0].content
        assert "PropertyDB" in system_text
        assert "property management database" in system_text

    def test_includes_tool_definitions(self):
        config = _make_registry_env_config()
        tool = _make_tool(
            output_schema={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        )
        conv = build_state_generation_prompt(config=config, tools=[tool])
        user_text = conv.messages[1].content
        assert "GetTenant" in user_text
        assert "tenant_id" in user_text

    def test_includes_scenario_context(self):
        config = _make_registry_env_config()
        conv = build_state_generation_prompt(
            config=config,
            tools=[],
            scenario_context="Residential apartment in Seattle",
        )
        user_text = conv.messages[1].content
        assert "Seattle" in user_text

    def test_no_tools_still_produces_prompt(self):
        config = _make_registry_env_config()
        conv = build_state_generation_prompt(config=config, tools=[])
        assert len(conv.messages) == 2
        user_text = conv.messages[1].content
        assert isinstance(user_text, str)
        assert "initial state" in user_text.lower()

    def test_retry_error_included(self):
        config = _make_registry_env_config()
        conv = build_state_generation_prompt(
            config=config,
            tools=[],
            retry_error="Could not parse JSON",
        )
        user_text = conv.messages[1].content
        assert "Could not parse JSON" in user_text


class TestBuildDependencyGraph:
    def test_no_foreign_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "units": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"number": {"type": "string"}},
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph == {"tenants": set(), "units": set()}

    def test_foreign_key_creates_dependency(self):
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "leases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "start_date": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["leases"] == {"tenants"}
        assert graph["tenants"] == set()

    def test_multiple_foreign_keys(self):
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "units": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"number": {"type": "string"}},
                    },
                },
                "leases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "unit_id": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["leases"] == {"tenants", "units"}

    def test_self_referencing_id_ignored(self):
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["tenants"] == set()


class TestSortIntoWaves:
    def test_independent_collections_in_one_wave(self):
        graph = {"tenants": set(), "units": set()}
        waves = sort_into_waves(graph)
        assert len(waves) == 1
        assert set(waves[0]) == {"tenants", "units"}

    def test_dependent_collection_in_later_wave(self):
        graph = {
            "tenants": set(),
            "units": set(),
            "leases": {"tenants", "units"},
        }
        waves = sort_into_waves(graph)
        assert len(waves) == 2
        assert set(waves[0]) == {"tenants", "units"}
        assert waves[1] == ["leases"]

    def test_three_wave_chain(self):
        graph = {
            "tenants": set(),
            "units": set(),
            "leases": {"tenants", "units"},
            "payments": {"leases"},
        }
        waves = sort_into_waves(graph)
        assert len(waves) == 3
        assert set(waves[0]) == {"tenants", "units"}
        assert waves[1] == ["leases"]
        assert waves[2] == ["payments"]

    def test_empty_graph(self):
        waves = sort_into_waves({})
        assert waves == []

    def test_cycle_broken(self):
        graph = {
            "tenants": {"leases"},
            "leases": {"tenants"},
        }
        waves = sort_into_waves(graph)
        all_collections = [c for wave in waves for c in wave]
        assert set(all_collections) == {"tenants", "leases"}


class TestBuildCollectionPrompt:
    def test_wave_0_prompt_has_no_existing_state(self):
        config = _make_registry_env_config()
        sub_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
        )
        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.SYSTEM
        user_text = conv.messages[1].content
        assert "tenants" in user_text
        assert "name" in user_text
        assert "Existing state" not in user_text

    def test_wave_1_prompt_includes_existing_state(self):
        config = _make_registry_env_config()
        sub_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "start_date": {"type": "string"},
            },
        }
        existing = {"tenants": {"T-001": {"name": "Alice"}}}
        conv = build_collection_prompt(
            config=config,
            collection_name="leases",
            sub_schema=sub_schema,
            existing_state=existing,
        )
        user_text = conv.messages[1].content
        assert "T-001" in user_text
        assert "Alice" in user_text
        assert "tenant_id" in user_text

    def test_prompt_includes_id_format_hint(self):
        config = _make_registry_env_config()
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
        )
        user_text = conv.messages[1].content
        assert isinstance(user_text, str)
        assert "string ids" in user_text.lower() or "keyed by" in user_text.lower()

    def test_prompt_requests_json_only(self):
        config = _make_registry_env_config()
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
        )
        user_text = conv.messages[1].content
        assert isinstance(user_text, str)
        assert "no markdown" in user_text.lower()

    def test_scenario_context_included(self):
        config = _make_registry_env_config()
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
            scenario_context="Residential apartment complex in Seattle",
        )
        user_text = conv.messages[1].content
        assert "Seattle" in user_text


class TestValidateCollection:
    def test_valid_collection(self):
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"1": {"name": "Alice"}, "2": {"name": "Bob"}}
        ok, error = validate_collection("tenants", data, sub_schema, {})
        assert ok is True
        assert error is None

    def test_invalid_json_type(self):
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        ok, error = validate_collection("tenants", [1, 2], sub_schema, {})
        assert ok is False
        assert error is not None
        assert "dict" in error.lower() or "object" in error.lower()

    def test_record_schema_violation(self):
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        data = {"1": {"name": 12345}}
        ok, _error = validate_collection("tenants", data, sub_schema, {})
        assert ok is False

    def test_referential_integrity_pass(self):
        sub_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "start_date": {"type": "string"},
            },
        }
        data = {
            "L-001": {
                "tenant_id": "T-001",
                "start_date": "2024-01-01",
            }
        }
        existing = {"tenants": {"T-001": {"name": "Alice"}}}
        ok, _error = validate_collection("leases", data, sub_schema, existing)
        assert ok is True

    def test_referential_integrity_fail(self):
        sub_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "start_date": {"type": "string"},
            },
        }
        data = {
            "L-001": {
                "tenant_id": "T-999",
                "start_date": "2024-01-01",
            }
        }
        existing = {"tenants": {"T-001": {"name": "Alice"}}}
        ok, error = validate_collection("leases", data, sub_schema, existing)
        assert ok is False
        assert error is not None
        assert "T-999" in error

    def test_empty_collection_valid(self):
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        ok, _error = validate_collection("tenants", {}, sub_schema, {})
        assert ok is True


class TestEnvironmentRegistryStatic:
    def test_register_static_and_create_copies(self):
        config = _make_registry_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {"type": "object"},
                },
            },
            initial_state={
                "tenants": {"1": {"name": "Alice"}},
            },
        )
        registry = EnvironmentRegistry()
        registry.register_static(config)

        copies = registry.create_copies("db", 3)

        assert len(copies) == 3
        for env in copies:
            assert isinstance(env, GeneratedToolEnvironment)
            assert env.state == {
                "tenants": {"1": {"name": "Alice"}},
            }

        copies[0].state["tenants"]["1"]["name"] = "Mutated"
        assert copies[1].state["tenants"]["1"]["name"] == "Alice"

    def test_create_copies_unknown_env_raises(self):
        registry = EnvironmentRegistry()
        with pytest.raises(KeyError):
            registry.create_copies("nonexistent", 1)


class TestEnvironmentRegistryBuild:
    def _make_mock_engine(self, responses: list[str]):
        engine = MagicMock()
        call_count = 0

        def mock_infer(input, inference_config=None):
            nonlocal call_count
            results = []
            for _ in input:
                text = responses[call_count % len(responses)]
                call_count += 1
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content=text,
                            )
                        ]
                    )
                )
            return results

        engine.infer = mock_infer
        return engine

    def test_build_with_config_provided_schema(self):
        config = _make_registry_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                    },
                },
            },
        )
        tenant_data = json.dumps({"T-001": {"name": "Alice"}, "T-002": {"name": "Bob"}})
        engine = self._make_mock_engine([tenant_data])
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        tools = [_make_tool()]
        registry.build(config, tools, engine, inference_config)

        copies = registry.create_copies("db", 2)
        assert len(copies) == 2
        assert "T-001" in copies[0].state["tenants"]

    def test_build_generates_state_from_description(self):
        config = _make_registry_env_config(
            state_schema=None,
            initial_state=None,
        )
        state_data = json.dumps(
            {
                "tenants": {"T-001": {"name": "Alice"}},
                "units": {"U-101": {"number": "101", "status": "occupied"}},
            }
        )
        engine = self._make_mock_engine([state_data])
        inference_config = MagicMock()

        tools = [_make_tool()]

        registry = EnvironmentRegistry()
        registry.build(config, tools, engine, inference_config)

        copies = registry.create_copies("db", 1)
        assert copies[0].state.get("tenants") is not None
        assert copies[0].state["tenants"]["T-001"]["name"] == "Alice"

    def test_build_multi_wave(self):
        config = _make_registry_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "leases": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "tenant_id": {"type": "string"},
                                "start_date": {"type": "string"},
                            },
                        },
                    },
                },
            },
        )
        tenant_data = json.dumps({"T-001": {"name": "Alice"}})
        lease_data = json.dumps(
            {
                "L-001": {
                    "tenant_id": "T-001",
                    "start_date": "2024-01-01",
                }
            }
        )
        engine = self._make_mock_engine([tenant_data, lease_data])
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        registry.build(config, [_make_tool()], engine, inference_config)

        copies = registry.create_copies("db", 1)
        state = copies[0].state
        assert "T-001" in state["tenants"]
        assert "L-001" in state["leases"]
        assert state["leases"]["L-001"]["tenant_id"] == "T-001"

    def test_build_partial_failure_keeps_successful_collections(self):
        config = _make_registry_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                    },
                    "units": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "number": {"type": "string"},
                            },
                        },
                    },
                },
            },
        )
        tenant_data = json.dumps({"T-001": {"name": "Alice"}})
        engine = self._make_mock_engine(
            [
                tenant_data,
                "not valid json at all",
                "not valid json at all",
                "not valid json at all",
            ]
        )
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        registry.build(config, [_make_tool()], engine, inference_config)

        copies = registry.create_copies("db", 1)
        state = copies[0].state
        assert "T-001" in state.get("tenants", {})
        assert "units" not in state or state.get("units") == {}

    def test_build_raises_on_response_count_mismatch(self):
        config = _make_registry_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                        },
                    }
                },
            },
        )
        engine = MagicMock()
        engine.infer.return_value = []
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        with pytest.raises(RuntimeError, match="collection generation"):
            registry.build(config, [_make_tool()], engine, inference_config)
