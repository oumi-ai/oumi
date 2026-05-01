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
    ToolArgumentError,
    ToolParams,
    ToolResult,
    validate_arguments_against_schema,
)
from oumi.core.types.tool_call import JSONSchema
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
        parameters=JSONSchema(
            type="object",
            properties={"query": JSONSchema(type="string")},
            required=["query"],
        ),
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
        parameters=JSONSchema(type="object"),
        output_schema=JSONSchema(
            type="object",
            properties={"result": JSONSchema(type="string")},
        ),
    )
    assert tool.to_llm_schema() == {
        "name": "search",
        "display_name": "Search",
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
    assert tool.output_schema == JSONSchema.model_validate(
        {"type": "object", "properties": {}}
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


def test_tool_schema_round_trip():
    schema = JSONSchema(
        type="object",
        properties={
            "query": JSONSchema(type="string"),
        },
        description="Tool input schema.",
        required=["query"],
    )
    assert schema.model_dump(mode="json", exclude_none=True) == {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "description": "Tool input schema.",
        "required": ["query"],
    }


def test_tool_schema_create_recursively_coerces_nested_properties():
    schema = JSONSchema.model_validate(
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
    assert schema.properties is not None
    customer = schema.properties["customer"]
    assert isinstance(customer, JSONSchema)
    assert customer.properties is not None
    assert isinstance(customer.properties["email"], JSONSchema)
    assert schema.model_dump(mode="json", exclude_none=True) == {
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


def test_tool_schema_construction_does_not_validate_required_against_properties():
    # Pydantic-backed JSONSchema permits ``required`` keys that aren't in
    # ``properties`` — JSON Schema itself allows this (extensions, $ref'd
    # properties, etc.). Argument validation will surface the mismatch at
    # call time via `validate_arguments_against_schema` if the caller omits
    # the required key.
    schema = JSONSchema(
        type="object",
        properties={"query": JSONSchema(type="string")},
        required=["missing"],
    )
    assert schema.required == ["missing"]


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


# --- JSONSchema items / enum ---


def test_tool_schema_create_coerces_items_and_enum():
    schema = JSONSchema.model_validate(
        {
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b"]},
        }
    )
    assert isinstance(schema.items, JSONSchema)
    assert schema.items.enum == ["a", "b"]
    assert schema.model_dump(mode="json", exclude_none=True) == {
        "type": "array",
        "items": {"type": "string", "enum": ["a", "b"]},
    }


def test_tool_schema_validate_rejects_wrong_item_type():
    schema = JSONSchema.model_validate({"type": "array", "items": {"type": "string"}})
    with pytest.raises(ToolArgumentError, match=r"arguments\[1\] must be a string"):
        validate_arguments_against_schema(["ok", 2], schema)


def test_tool_schema_validate_rejects_value_outside_enum():
    schema = JSONSchema.model_validate({"type": "string", "enum": ["a", "b"]})
    with pytest.raises(ToolArgumentError, match="must be one of"):
        validate_arguments_against_schema("c", schema)


def test_tool_schema_enum_must_be_list():
    # Pydantic surfaces the type mismatch via its own validation error.
    with pytest.raises(ValueError, match="Input should be a valid list"):
        JSONSchema(type="string", enum="a")  # type: ignore[arg-type]


# --- JSONSchema.validate / ToolParams.validate_arguments ---


def _policy_tool() -> ToolParams:
    return ToolParams(
        id="policy",
        name="Policy",
        description="Look up policy.",
        parameters=JSONSchema(
            type="object",
            properties={
                "policy_id": JSONSchema(type="string"),
                "limit": JSONSchema(type="integer"),
            },
            required=["policy_id"],
        ),
    )


def test_tool_schema_validate_accepts_conforming_value():
    schema = _policy_tool().parameters
    validate_arguments_against_schema({"policy_id": "abc", "limit": 5}, schema)


def test_tool_schema_validate_missing_required_raises():
    schema = _policy_tool().parameters
    with pytest.raises(ToolArgumentError, match=r"arguments\.policy_id is required"):
        validate_arguments_against_schema({"limit": 5}, schema)


def test_tool_schema_validate_wrong_type_raises():
    schema = _policy_tool().parameters
    with pytest.raises(ToolArgumentError, match=r"arguments\.limit must be an integer"):
        validate_arguments_against_schema({"policy_id": "abc", "limit": "five"}, schema)


def test_tool_schema_validate_nested_object():
    schema = JSONSchema.model_validate(
        {
            "type": "object",
            "properties": {
                "customer": {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                }
            },
            "required": ["customer"],
        }
    )
    with pytest.raises(
        ToolArgumentError, match=r"arguments\.customer\.email is required"
    ):
        validate_arguments_against_schema({"customer": {}}, schema)


def test_tool_schema_validate_empty_schema_accepts_any_object():
    # Tools that don't declare parameters shouldn't force callers to pass {}.
    tool = ToolParams(id="ping", name="Ping", description="No args.")
    tool.validate_arguments({})
    tool.validate_arguments({"extra": "ignored"})


def test_tool_params_validate_arguments_delegates_to_parameters():
    with pytest.raises(ToolArgumentError, match="arguments.policy_id is required"):
        _policy_tool().validate_arguments({})


# --- GroundingConfig ---


def test_grounding_config_defaults():
    from oumi.environments import GroundingConfig

    cfg = GroundingConfig()
    assert cfg.sample_size == 3
    assert cfg.seed is None


def test_grounding_config_accepts_valid_values():
    from oumi.environments import GroundingConfig

    cfg = GroundingConfig(sample_size=5, seed=42)
    assert cfg.sample_size == 5
    assert cfg.seed == 42


def test_grounding_config_rejects_sample_size_below_one():
    from oumi.environments import GroundingConfig

    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        GroundingConfig(sample_size=0)
    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        GroundingConfig(sample_size=-3)


# --- describe_grounding_default ---


def test_describe_grounding_default_empty():
    from oumi.environments.utils import describe_grounding_default

    assert describe_grounding_default([]) == ""


def test_describe_grounding_default_single_fact():
    from oumi.core.configs.params.grounding_params import GroundingFact
    from oumi.environments.utils import describe_grounding_default

    facts = [GroundingFact(data={"id": "42", "title": "Dune", "year": 1965})]
    rendered = describe_grounding_default(facts)
    assert rendered == '- id="42", title="Dune", year=1965'


def test_describe_grounding_default_multi_fact_preserves_order():
    from oumi.core.configs.params.grounding_params import GroundingFact
    from oumi.environments.utils import describe_grounding_default

    facts = [
        GroundingFact(data={"id": "7", "title": "LotR"}),
        GroundingFact(data={"id": "42", "title": "Dune"}),
    ]
    rendered = describe_grounding_default(facts)
    assert rendered == ('- id="7", title="LotR"\n- id="42", title="Dune"')


def test_describe_grounding_default_handles_non_string_values():
    from oumi.core.configs.params.grounding_params import GroundingFact
    from oumi.environments.utils import describe_grounding_default

    facts = [
        GroundingFact(data={"id": 42, "available": True, "count": 3, "rating": 4.5})
    ]
    rendered = describe_grounding_default(facts)
    assert "id=42" in rendered
    assert "available=True" in rendered
    assert "count=3" in rendered
    assert "rating=4.5" in rendered
