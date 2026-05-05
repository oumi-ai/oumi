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

from typing import Any

import jsonschema
import pytest

from oumi.core.configs.params.tool_params import ToolArgumentError, ToolParams
from oumi.core.types.tool_call import JSONSchema, ToolDefinition, ToolResult
from oumi.environments.synthetic_environment import SyntheticStateParams


def _make_state_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        },
        "required": ["files"],
    }


@pytest.mark.parametrize("field,value", [("id", ""), ("name", ""), ("description", "")])
def test_tool_params_empty_field_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        ToolParams(**{"id": "t", "name": "T", "description": "d", **{field: value}})


def test_tool_params_to_llm_schema():
    tool = ToolParams(
        id="search",
        name="Search",
        description="Search the catalog.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    assert tool.to_llm_schema() == {
        "name": "search",
        "display_name": "Search",
        "description": "Search the catalog.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def test_tool_params_to_llm_schema_includes_output_schema():
    tool = ToolParams(
        id="search",
        name="Search",
        description="Search the catalog.",
        parameters={"type": "object", "properties": {}},
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
    )
    assert tool.to_llm_schema() == {
        "name": "search",
        "display_name": "Search",
        "description": "Search the catalog.",
        "parameters": {"type": "object", "properties": {}},
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
    }


def test_tool_params_create_reads_extended_fields():
    tool = ToolParams.create(
        {
            "id": "policy",
            "name": "Policy",
            "description": "Look up policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {"type": "string"},
                },
                "required": ["policy_id"],
            },
            "output_schema": {"type": "object", "properties": {}},
        }
    )
    assert tool.parameters["required"] == ["policy_id"]
    assert tool.output_schema == {"type": "object", "properties": {}}


def test_tool_params_accepts_jsonschema_pydantic_instance():
    """Direct construction with a Pydantic JSONSchema is supported via post_init."""
    tool = ToolParams(
        id="search",
        name="Search",
        description="Search.",
        parameters=JSONSchema(  # type: ignore[arg-type]
            type="object",
            properties={"q": JSONSchema(type="string")},
            required=["q"],
        ),
    )
    assert isinstance(tool.parameters, dict)
    assert tool.parameters == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }


def test_tool_params_to_tool_definition_drops_chain_internals():
    tool = ToolParams(
        id="search",
        name="Search Display",
        description="Search the catalog.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        output_schema={"type": "object"},
        read_only=False,
    )
    td = tool.to_tool_definition()
    assert isinstance(td, ToolDefinition)
    # name/description/parameters preserved; display name and output_schema dropped.
    assert td.function.name == "search"
    assert td.function.description == "Search the catalog."
    assert td.function.parameters == JSONSchema(
        type="object",
        properties={"query": JSONSchema(type="string")},
        required=["query"],
    )


def test_tool_params_grounding_default_none():
    tool = ToolParams(id="t", name="T", description="d")
    assert tool.grounding is None


def test_tool_params_grounding_from_dict_via_create():
    tool = ToolParams.create(
        {
            "id": "lookup_book_status",
            "name": "Lookup",
            "description": "lookup",
            "grounding": {
                "key": "book_id",
                "fields": ["book_id", "title", "status"],
            },
        }
    )
    assert tool.grounding is not None
    assert tool.grounding.key == "book_id"
    assert tool.grounding.fields == ["book_id", "title", "status"]


def test_tool_params_grounding_from_post_init_dict():
    """ToolParams coerces raw grounding dicts from direct dataclass construction."""
    tool = ToolParams(
        id="t",
        name="T",
        description="d",
        grounding={"key": "k", "fields": ["k", "v"]},  # type: ignore[arg-type]
    )
    assert tool.grounding is not None
    assert tool.grounding.key == "k"


def test_tool_result_round_trip():
    result = ToolResult(output={"msg": "ok"})
    assert result.output == {"msg": "ok"}
    assert result.updated_state is None


def test_validate_arguments_rejects_wrong_item_type():
    tool = ToolParams(
        id="t",
        name="T",
        description="d",
        parameters={"type": "array", "items": {"type": "string"}},
    )
    with pytest.raises(ToolArgumentError):
        tool.validate_arguments(["ok", 2])  # type: ignore[arg-type]


def test_validate_arguments_rejects_value_outside_enum():
    tool = ToolParams(
        id="t",
        name="T",
        description="d",
        parameters={"type": "string", "enum": ["a", "b"]},
    )
    with pytest.raises(ToolArgumentError):
        tool.validate_arguments("c")  # type: ignore[arg-type]


def _policy_tool() -> ToolParams:
    return ToolParams(
        id="policy",
        name="Policy",
        description="Look up policy.",
        parameters={
            "type": "object",
            "properties": {
                "policy_id": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["policy_id"],
        },
    )


def test_validate_arguments_accepts_conforming_value():
    _policy_tool().validate_arguments({"policy_id": "abc", "limit": 5})


def test_validate_arguments_missing_required_raises():
    with pytest.raises(ToolArgumentError, match=r"policy_id"):
        _policy_tool().validate_arguments({"limit": 5})


def test_validate_arguments_wrong_type_raises():
    with pytest.raises(ToolArgumentError, match=r"integer"):
        _policy_tool().validate_arguments({"policy_id": "abc", "limit": "five"})


def test_validate_arguments_empty_schema_accepts_any_object():
    # Tools that don't declare parameters shouldn't force callers to pass {}.
    tool = ToolParams(id="ping", name="Ping", description="No args.")
    tool.validate_arguments({})
    tool.validate_arguments({"extra": "ignored"})


def test_synthetic_state_params_validates_initial_state_against_schema():
    with pytest.raises(jsonschema.ValidationError):
        SyntheticStateParams(
            state_schema=_make_state_schema(),
            initial_state={"files": {"count": "bad"}},
        )


def test_synthetic_state_params_accepts_partial_inputs():
    assert (
        SyntheticStateParams(state_schema=_make_state_schema()).state_schema is not None
    )
    assert SyntheticStateParams(
        initial_state={"files": {"count": 1}}
    ).initial_state == {"files": {"count": 1}}
