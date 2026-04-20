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

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.environments import (
    BaseEnvironment,
    DeterministicEnvironment,
    DeterministicToolOutput,
    GroundingFact,
    SyntheticEnvironment,
    SyntheticStateParams,
    Tool,
    ToolArgumentError,
    ToolLookupError,
    ToolResult,
    ToolSchema,
)


def _make_deterministic_tool(**overrides: Any) -> Tool:
    defaults: dict[str, Any] = dict(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    defaults.update(overrides)
    return Tool(**defaults)


def _make_synthetic_tool(**overrides: Any) -> Tool:
    defaults: dict[str, Any] = dict(
        id="tool2",
        name="GenTool",
        description="A generated tool",
    )
    defaults.update(overrides)
    return Tool(**defaults)


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


def test_deterministic_tool_output_matches_exact():
    entry = DeterministicToolOutput(
        input={"id": "01", "status": "pending"},
        output={"message": "Order is pending"},
    )
    assert entry.matches({"id": "01", "status": "pending"}) is True
    assert entry.matches({"status": "pending", "id": "01"}) is True


def test_deterministic_tool_output_no_match():
    entry = DeterministicToolOutput(
        input={"id": "01"},
        output={"message": "ok"},
    )
    assert entry.matches({"id": "02"}) is False
    assert entry.matches({"id": "01", "extra": "arg"}) is False


@pytest.mark.parametrize("field,value", [("id", ""), ("name", ""), ("description", "")])
def test_tool_empty_field_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        Tool(**{"id": "t", "name": "T", "description": "d", **{field: value}})


def test_tool_to_llm_schema():
    tool = Tool(
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
        "name": "search",
        "display_name": "Search",
        "description": "Search the catalog.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }


def test_tool_to_llm_schema_includes_output_schema():
    tool = Tool(
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
        "name": "search",
        "display_name": "Search",
        "description": "Search the catalog.",
        "parameters": {"type": "object"},
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
        },
    }


def test_tool_create_coerces_deterministic_outputs():
    env = BaseEnvironment.create(
        {
            "id": "lookup",
            "type": "deterministic",
            "name": "Lookup",
            "description": "Lookup tools",
            "tools": [
                {
                    "id": "policy",
                    "name": "Policy",
                    "description": "Look up policy.",
                    "deterministic_outputs": [
                        {"input": {"id": "1"}, "output": {"result": "ok"}}
                    ],
                }
            ],
        }
    )
    assert isinstance(env, DeterministicEnvironment)
    assert isinstance(env.tools[0].deterministic_outputs[0], DeterministicToolOutput)


def test_tool_create_reads_extended_tool_fields():
    tool = Tool.create(
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


def test_tool_schema_create_coerces_items_and_enum():
    schema = ToolSchema.create(
        {
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b"]},
        }
    )
    assert isinstance(schema.items, ToolSchema)
    assert schema.items.enum == ["a", "b"]
    assert schema.to_dict() == {
        "type": "array",
        "items": {"type": "string", "enum": ["a", "b"]},
    }


def test_tool_schema_validate_rejects_wrong_item_type():
    schema = ToolSchema.create({"type": "array", "items": {"type": "string"}})
    with pytest.raises(ToolArgumentError, match=r"arguments\[1\] must be a string"):
        schema.validate(["ok", 2], path="arguments")


def test_tool_schema_validate_rejects_value_outside_enum():
    schema = ToolSchema.create({"type": "string", "enum": ["a", "b"]})
    with pytest.raises(ToolArgumentError, match="must be one of"):
        schema.validate("c", path="arguments")


def test_tool_schema_enum_must_be_list():
    with pytest.raises(ValueError, match="enum must be a list"):
        ToolSchema(type="string", enum="a")  # type: ignore[arg-type]


def test_tool_post_init_coerces_deterministic_outputs_from_direct_construction():
    tool = Tool(
        id="policy",
        name="Policy",
        description="Look up policy.",
        deterministic_outputs=[
            {"input": {"id": "1"}, "output": {"result": "ok"}},  # type: ignore[list-item]
        ],
    )
    assert isinstance(tool.deterministic_outputs[0], DeterministicToolOutput)


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


def test_synthetic_environment_valid_stateful():
    env = SyntheticEnvironment(
        id="filesystem",
        name="Filesystem",
        description="A simple filesystem",
        system_prompt="You manage a filesystem.",
        state_params=SyntheticStateParams(
            state_schema=_make_state_schema(),
            initial_state={"files": {"count": 1}},
        ),
        cache_by_input=False,
        tools=[Tool(id="read", name="Read", description="Read files.")],
    )
    assert env.current_state == {"files": {"count": 1}}


def test_synthetic_environment_coerces_dict_tools():
    env = SyntheticEnvironment(
        id="fs",
        name="FS",
        description="d",
        system_prompt="p",
        tools=[{"id": "read", "name": "Read", "description": "Read files."}],  # type: ignore[list-item]
    )
    assert isinstance(env.tools[0], Tool)


def test_synthetic_environment_empty_system_prompt_raises():
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        SyntheticEnvironment(id="x", name="n", description="d", system_prompt="")


def test_synthetic_environment_rejects_deterministic_outputs():
    with pytest.raises(ValueError, match="cannot define deterministic_outputs"):
        SyntheticEnvironment(
            id="x",
            name="n",
            description="d",
            system_prompt="p",
            tools=[_make_deterministic_tool()],
        )


def test_synthetic_environment_rejects_cache_when_stateful():
    with pytest.raises(ValueError, match="cache_by_input must be False"):
        SyntheticEnvironment(
            id="x",
            name="n",
            description="d",
            system_prompt="p",
            state_params=SyntheticStateParams(),
            cache_by_input=True,
        )


def test_synthetic_environment_cache_round_trip():
    env = SyntheticEnvironment(
        id="weather",
        name="Weather",
        description="Weather API",
        system_prompt="Simulate weather.",
        cache_by_input=True,
        tools=[_make_synthetic_tool(id="get_weather")],
    )
    result = ToolResult(output={"temp": 72})
    env._cache_result("get_weather", {"city": "SF"}, result)
    cached = env._resolve_cached("get_weather", {"city": "SF"})
    assert cached == result
    assert cached is not result


def test_synthetic_environment_step_unknown_tool_raises():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step("missing", {})


def test_synthetic_environment_step_known_tool_is_stub():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        env.step("answer", {})


def test_deterministic_environment_valid():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            Tool(
                id="policy",
                name="Policy",
                description="Look up policy.",
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "1"},
                        output={"result": "ok"},
                    )
                ],
            )
        ],
    )
    assert env.type == "deterministic"
    assert isinstance(env.tools[0], Tool)


def test_deterministic_environment_requires_outputs_on_tool():
    with pytest.raises(ValueError, match="must have at least one"):
        DeterministicEnvironment(
            id="det_env",
            name="Deterministic",
            description="d",
            tools=[_make_deterministic_tool(deterministic_outputs=[])],
        )


def test_deterministic_environment_duplicate_inputs_raises():
    outputs = [
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "a"}),
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "b"}),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        DeterministicEnvironment(
            id="det_env",
            name="Deterministic",
            description="d",
            tools=[_make_deterministic_tool(deterministic_outputs=outputs)],
        )


def test_deterministic_environment_step_match():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            _make_deterministic_tool(
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "01"}, output={"msg": "pending"}
                    ),
                    DeterministicToolOutput(
                        input={"id": "02"}, output={"msg": "delivered"}
                    ),
                ]
            )
        ],
    )
    assert env.step("tool1", {"id": "01"}) == ToolResult(output={"msg": "pending"})
    assert env.step("tool1", {"id": "02"}) == ToolResult(output={"msg": "delivered"})


def test_deterministic_environment_step_no_match_raises_with_hint():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[_make_deterministic_tool()],
    )
    with pytest.raises(ToolLookupError) as excinfo:
        env.step("tool1", {"id": "99"})
    message = str(excinfo.value)
    assert "No deterministic output matches" in message
    assert "tool1" in message
    # The configured inputs are surfaced so the LLM can self-correct.
    assert '"id": "01"' in message


def test_deterministic_environment_supports_empty_argument_match():
    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        tools=[
            Tool(
                id="ping",
                name="Ping",
                description="Zero-arg tool.",
                deterministic_outputs=[
                    DeterministicToolOutput(input={}, output={}),
                ],
            )
        ],
    )
    assert env.step("ping", {}) == ToolResult(output={})


def test_environment_empty_id_raises():
    with pytest.raises(ValueError, match="id cannot be empty"):
        SyntheticEnvironment(id="", name="n", description="d", system_prompt="p")


def test_environment_empty_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        SyntheticEnvironment(id="x", name="", description="d", system_prompt="p")


def test_environment_empty_description_raises():
    with pytest.raises(ValueError, match="description cannot be empty"):
        SyntheticEnvironment(id="x", name="n", description="", system_prompt="p")


def test_environment_duplicate_tool_ids_raises():
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        SyntheticEnvironment(
            id="env2",
            name="Env 2",
            description="d",
            system_prompt="p",
            tools=[
                Tool(id="dup", name="Read", description="Read files."),
                Tool(id="dup", name="Write", description="Write files."),
            ],
        )


def test_environment_config_duplicate_tool_ids_across_envs_raises():
    env1 = SyntheticEnvironment(
        id="env1",
        name="Env 1",
        description="d",
        system_prompt="p",
        tools=[Tool(id="dup", name="Read", description="Read files.")],
    )
    env2 = SyntheticEnvironment(
        id="env2",
        name="Env 2",
        description="d",
        system_prompt="p",
        tools=[Tool(id="dup", name="Write", description="Write files.")],
    )
    with pytest.raises(ValueError, match="duplicate tool id 'dup'"):
        EnvironmentConfig(environments=[env1, env2])


def test_environment_config_tool_environment_map():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer_faq")],
    )
    config = EnvironmentConfig(environments=[env])
    assert config.tool_environment_map == {"answer_faq": "faq"}


def test_base_environment_create_routes_synthetic():
    env = BaseEnvironment.create(
        {
            "id": "faq",
            "type": "synthetic",
            "name": "FAQ",
            "description": "FAQ tools",
            "system_prompt": "Answer FAQs.",
            "tools": [{"id": "answer", "name": "Answer", "description": "Answer."}],
        }
    )
    assert isinstance(env, SyntheticEnvironment)


def test_base_environment_create_routes_deterministic():
    env = BaseEnvironment.create(
        {
            "id": "lookup",
            "type": "deterministic",
            "name": "Lookup",
            "description": "Lookup tools",
            "tools": [
                {
                    "id": "policy",
                    "name": "Policy",
                    "description": "Look up policy.",
                    "deterministic_outputs": [
                        {"input": {"id": "1"}, "output": {"result": "ok"}}
                    ],
                }
            ],
        }
    )
    assert isinstance(env, DeterministicEnvironment)


def test_base_environment_create_missing_type_raises():
    with pytest.raises(ValueError, match="must include a 'type' field"):
        BaseEnvironment.create({"id": "faq"})


def test_base_environment_create_unsupported_type_raises():
    with pytest.raises(ValueError, match="Unsupported environment type"):
        BaseEnvironment.create({"id": "faq", "type": "unknown"})


# --- ToolSchema.validate / Tool.validate_arguments ---


def _policy_tool() -> Tool:
    return Tool(
        id="policy",
        name="Policy",
        description="Look up policy.",
        parameters=ToolSchema(
            type="object",
            properties={
                "policy_id": ToolSchema(type="string"),
                "limit": ToolSchema(type="integer"),
            },
            required=["policy_id"],
        ),
    )


def test_tool_schema_validate_missing_required_raises():
    schema = _policy_tool().parameters
    with pytest.raises(ToolArgumentError, match=r"arguments\.policy_id is required"):
        schema.validate({"limit": 5}, path="arguments")


def test_tool_schema_validate_wrong_type_raises():
    schema = _policy_tool().parameters
    with pytest.raises(ToolArgumentError, match=r"arguments\.limit must be an integer"):
        schema.validate({"policy_id": "abc", "limit": "five"}, path="arguments")


def test_tool_schema_validate_nested_object():
    schema = ToolSchema.create(
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
        schema.validate({"customer": {}}, path="arguments")


def test_tool_schema_validate_empty_schema_accepts_any_object():
    # Tools that don't declare parameters shouldn't force callers to pass {}.
    tool = Tool(id="ping", name="Ping", description="No args.")
    tool.validate_arguments({})
    tool.validate_arguments({"extra": "ignored"})


def test_tool_validate_arguments_delegates_to_parameters():
    with pytest.raises(ToolArgumentError, match="arguments.policy_id is required"):
        _policy_tool().validate_arguments({})


# --- GroundingConfig ---


def test_grounding_config_rejects_sample_size_below_one():
    from oumi.environments import GroundingConfig

    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        GroundingConfig(sample_size=0)
    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        GroundingConfig(sample_size=-3)


# --- describe_grounding_default ---


def test_describe_grounding_default_single_fact():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [GroundingFact(data={"id": "42", "title": "Dune", "year": 1965})]
    rendered = describe_grounding_default(facts)
    assert rendered == '- id="42", title="Dune", year=1965'


def test_describe_grounding_default_multi_fact_preserves_order():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [
        GroundingFact(data={"id": "7", "title": "LotR"}),
        GroundingFact(data={"id": "42", "title": "Dune"}),
    ]
    rendered = describe_grounding_default(facts)
    assert rendered == ('- id="7", title="LotR"\n- id="42", title="Dune"')


def test_describe_grounding_default_handles_non_string_values():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [
        GroundingFact(data={"id": 42, "available": True, "count": 3, "rating": 4.5})
    ]
    rendered = describe_grounding_default(facts)
    assert "id=42" in rendered
    assert "available=True" in rendered
    assert "count=3" in rendered
    assert "rating=4.5" in rendered


# --- BaseEnvironment grounding defaults ---


# --- DeterministicEnvironment.sample_grounding ---


def _det_env_with_n_entries(n: int) -> DeterministicEnvironment:
    """Build a DeterministicEnvironment with a single tool containing n entries."""
    outputs = [
        DeterministicToolOutput(input={"id": str(i)}, output={"title": f"title-{i}"})
        for i in range(n)
    ]
    return DeterministicEnvironment(
        id="books",
        name="Books",
        description="Book lookup",
        tools=[
            Tool(
                id="lookup",
                name="Lookup",
                description="Look up a book.",
                deterministic_outputs=outputs,
            )
        ],
    )


def test_deterministic_sample_grounding_returns_n_facts():
    import random

    env = _det_env_with_n_entries(10)
    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    for fact in facts:
        assert isinstance(fact, GroundingFact)


def test_deterministic_sample_grounding_no_replacement_within_call():
    import random

    env = _det_env_with_n_entries(10)
    facts = env.sample_grounding(n=5, rng=random.Random(0))
    ids = [fact.data["id"] for fact in facts]
    assert len(set(ids)) == len(ids)


def test_deterministic_sample_grounding_truncates_when_n_exceeds_pool():
    import random

    env = _det_env_with_n_entries(3)
    facts = env.sample_grounding(n=10, rng=random.Random(0))
    assert len(facts) == 3


def test_deterministic_sample_grounding_seeded_rng_is_reproducible():
    import random

    env = _det_env_with_n_entries(20)
    facts_a = env.sample_grounding(n=4, rng=random.Random(42))
    facts_b = env.sample_grounding(n=4, rng=random.Random(42))
    ids_a = [fact.data["id"] for fact in facts_a]
    ids_b = [fact.data["id"] for fact in facts_b]
    assert ids_a == ids_b


def test_deterministic_sample_grounding_pools_across_tools():
    env = DeterministicEnvironment(
        id="multi",
        name="Multi",
        description="Two tools",
        tools=[
            Tool(
                id="tool_a",
                name="A",
                description="Tool A",
                deterministic_outputs=[
                    DeterministicToolOutput(input={"k": "a1"}, output={"v": "a1"})
                ],
            ),
            Tool(
                id="tool_b",
                name="B",
                description="Tool B",
                deterministic_outputs=[
                    DeterministicToolOutput(input={"k": "b1"}, output={"v": "b1"}),
                    DeterministicToolOutput(input={"k": "b2"}, output={"v": "b2"}),
                ],
            ),
        ],
    )
    import random

    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    keys = sorted(fact.data["k"] for fact in facts)
    assert keys == ["a1", "b1", "b2"]


def test_deterministic_sample_grounding_output_wins_on_key_conflict():
    import random

    env = DeterministicEnvironment(
        id="lookup",
        name="Lookup",
        description="Look up an id.",
        tools=[
            Tool(
                id="lookup",
                name="Lookup",
                description="Look up an id.",
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "1", "note": "input-note"},
                        output={"note": "output-note"},
                    )
                ],
            )
        ],
    )
    facts = env.sample_grounding(n=1, rng=random.Random(0))
    assert len(facts) == 1
    assert facts[0].data == {"id": "1", "note": "output-note"}
