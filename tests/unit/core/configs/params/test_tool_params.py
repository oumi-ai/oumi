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

import pytest

from oumi.core.configs.params.tool_params import (
    ToolParams,
    ToolResult,
    ToolSchema,
)
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
        parameters=ToolSchema(
            type="object",
            properties={"query": ToolSchema(type="string")},
            required=["query"],
        ),
    )
    assert tool.to_llm_schema() == {
        "name": "Search",
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
        parameters=ToolSchema(type="object", properties={}),
        output_schema=ToolSchema(
            type="object",
            properties={"result": ToolSchema(type="string")},
        ),
    )
    assert tool.to_llm_schema() == {
        "name": "Search",
        "description": "Search the catalog.",
        "parameters": {"type": "object"},
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
    assert tool.parameters.required == ["policy_id"]
    assert tool.output_schema == ToolSchema(type="object", properties={})


def test_tool_result_round_trip():
    result = ToolResult(output={"msg": "ok"})
    assert result.output == {"msg": "ok"}
    assert result.updated_state is None


def test_tool_schema_to_dict():
    schema = ToolSchema(
        type="object",
        properties={
            "query": ToolSchema(type="string"),
        },
        description="Tool input schema.",
        required=["query"],
    )
    assert schema.to_dict() == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "description": "Tool input schema.",
        "required": ["query"],
    }


def test_tool_schema_create_recursively_coerces_nested_properties():
    schema = ToolSchema.create(
        {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Customer email."}
                    },
                    "required": ["email"],
                }
            },
            "required": ["customer"],
        }
    )
    assert isinstance(schema.properties["customer"], ToolSchema)
    assert isinstance(schema.properties["customer"].properties["email"], ToolSchema)
    assert schema.to_dict() == {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Customer email.",
                    }
                },
                "required": ["email"],
            }
        },
        "required": ["customer"],
    }


def test_tool_schema_required_must_exist_in_properties():
    with pytest.raises(ValueError, match="required contains unknown properties"):
        ToolSchema(
            type="object",
            properties={"query": ToolSchema(type="string")},
            required=["missing"],
        )


def test_synthetic_state_params_validates_initial_state_against_schema():
    with pytest.raises(ValueError, match=r"\$\.files\.count must be an integer"):
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
