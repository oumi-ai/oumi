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

import logging
import random

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import (
    GroundingConfig,
    GroundingFact,
    ToolGroundingConfig,
)
from oumi.core.configs.params.tool_params import ToolLookupError, ToolParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.deterministic_environment import (
    DeterministicEnvironment,
    DeterministicEnvironmentKwargs,
    ToolLookupEntry,
)


def _make_tool(
    tool_id: str = "tool1",
    parameters: dict | None = None,
) -> ToolParams:
    return ToolParams(
        id=tool_id,
        name=tool_id,
        description="A tool",
        parameters=parameters if parameters is not None else {"type": "object"},
    )


def _make_params(
    tools: list[ToolParams] | None = None,
    lookup_table: dict[str, list[ToolLookupEntry]] | None = None,
    grounding: GroundingConfig | None = None,
    **overrides,
) -> EnvironmentParams:
    """Build EnvironmentParams with defaults that pass validation."""
    if tools is None:
        tools = [_make_tool()]
    if lookup_table is None:
        lookup_table = {
            "tool1": [ToolLookupEntry(input={"id": "01"}, output={"msg": "ok"})]
        }
    defaults: dict = dict(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        env_type="deterministic",
        tools=tools,
        env_kwargs={"lookup_table": lookup_table},
        grounding=grounding,
    )
    defaults.update(overrides)
    return EnvironmentParams(**defaults)


# --- from_params + lookup_table validation ---


def test_from_params_constructs_runtime_instance():
    env = DeterministicEnvironment.from_params(_make_params())
    assert isinstance(env, DeterministicEnvironment)
    assert isinstance(env._kwargs, DeterministicEnvironmentKwargs)


def test_from_params_coerces_raw_lookup_entries():
    """Raw dict entries in lookup_table are coerced to ToolLookupEntry."""
    env = DeterministicEnvironment.from_params(
        _make_params(
            lookup_table={  # type: ignore[arg-type]
                "tool1": [{"input": {"id": "1"}, "output": {"msg": "ok"}}],
            }
        )
    )
    entry = env._kwargs.lookup_table["tool1"][0]
    assert isinstance(entry, ToolLookupEntry)
    assert entry.input == {"id": "1"}


def test_tool_without_entries_raises():
    """Hard error: tool declared but lookup_table has no entries for it."""
    with pytest.raises(ValueError, match="has no entries in lookup_table"):
        DeterministicEnvironment.from_params(
            _make_params(
                tools=[_make_tool("tool1"), _make_tool("tool2")],
                lookup_table={
                    "tool1": [ToolLookupEntry(input={"id": "01"}, output={"msg": "ok"})]
                    # tool2 missing
                },
            )
        )


def test_stale_lookup_table_keys_warn(caplog):
    """Warning (not error) when lookup_table has entries for unknown tool."""
    with caplog.at_level(logging.WARNING, logger="oumi"):
        DeterministicEnvironment.from_params(
            _make_params(
                lookup_table={
                    "tool1": [
                        ToolLookupEntry(input={"id": "01"}, output={"msg": "ok"})
                    ],
                    "ghost_tool": [
                        ToolLookupEntry(input={"id": "x"}, output={"msg": "y"})
                    ],
                }
            )
        )
    assert any(
        "ghost_tool" in rec.getMessage() and "unknown tool" in rec.getMessage()
        for rec in caplog.records
    )


def test_unknown_env_kwargs_raises_with_known_keys():
    """Typos in env_kwargs surface as a clear ValueError naming known keys."""
    params = _make_params()
    params.env_kwargs = {
        "lookup_table": {"tool1": [{"input": {}, "output": {}}]},
        "lookup_tabel": {},
    }
    with pytest.raises(ValueError, match="unknown env_kwargs.*lookup_tabel"):
        DeterministicEnvironment.from_params(params)


def test_duplicate_inputs_raises():
    with pytest.raises(ValueError, match="duplicate input"):
        DeterministicEnvironment.from_params(
            _make_params(
                lookup_table={
                    "tool1": [
                        ToolLookupEntry(input={"id": "01"}, output={"msg": "a"}),
                        ToolLookupEntry(input={"id": "01"}, output={"msg": "b"}),
                    ]
                }
            )
        )


def test_inputs_with_omitted_and_explicit_defaults_are_duplicates():
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "default": "fahrenheit"},
            },
            "required": ["location"],
        }
    )

    with pytest.raises(ValueError, match="duplicate input"):
        DeterministicEnvironment.from_params(
            _make_params(
                tools=[tool],
                lookup_table={
                    "tool1": [
                        ToolLookupEntry(
                            input={"location": "sf"},
                            output={"temperature": 65},
                        ),
                        ToolLookupEntry(
                            input={"location": "sf", "unit": "fahrenheit"},
                            output={"temperature": 65},
                        ),
                    ]
                },
            )
        )


# --- unreachable schema defaults ---


@pytest.mark.parametrize(
    "parameters,expected_path",
    [
        pytest.param(
            {
                "type": "object",
                "properties": {
                    "unit": {
                        "anyOf": [
                            {"type": "string", "default": "fahrenheit"},
                            {"type": "null"},
                        ]
                    }
                },
            },
            "parameters.properties.unit.anyOf[0].default",
            id="anyOf",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "default": "all"},
                    }
                },
            },
            "parameters.properties.tags.items.default",
            id="items",
        ),
        pytest.param(
            {
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {"enum": {"type": "string", "default": "x"}},
                    }
                },
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
                },
            },
            "parameters.properties.children.items.properties.enum.default",
            id="property-named-like-a-keyword",
        ),
        pytest.param(
            # `list[Node]` on a recursive pydantic model: the same `$ref` is
            # reached both fillably and not, so the cycle guard must key on both.
            {
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "default": "leaf"},
                            "children": {
                                "type": "array",
                                "default": [],
                                "items": {"$ref": "#/$defs/Node"},
                            },
                        },
                    }
                },
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/Node"}},
            },
            "parameters.properties.root.properties.children.items"
            ".properties.label.default",
            id="recursive-model-under-items",
        ),
    ],
)
def test_unreachable_schema_default_raises(parameters, expected_path):
    """A default off the `properties` chain errors at construction, not at lookup."""
    with pytest.raises(ValueError, match="never applied") as excinfo:
        DeterministicEnvironment.from_params(
            _make_params(tools=[_make_tool(parameters=parameters)])
        )
    assert expected_path in str(excinfo.value)
    assert "tool1" in str(excinfo.value)


def test_combinator_without_defaults_is_accepted():
    """Only a hidden default is an error; `anyOf` on its own stays usable."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        }
    )

    env = DeterministicEnvironment.from_params(_make_params(tools=[tool]))

    assert env.step([("tool1", {"id": "01"})]) == [ToolResult(output={"msg": "ok"})]


def test_pydantic_style_ref_schema_fills_nested_defaults():
    """`model_json_schema()` emits `$ref` + `$defs`; defaults must still fill."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "$defs": {
                "Opts": {
                    "type": "object",
                    "properties": {"unit": {"type": "string", "default": "fahrenheit"}},
                }
            },
            "properties": {
                "location": {"type": "string"},
                "opts": {"$ref": "#/$defs/Opts", "default": {"unit": "fahrenheit"}},
            },
            "required": ["location"],
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(input={"location": "sf"}, output={"msg": "ok"})
                ]
            },
        )
    )

    # Both the absent `opts` and an explicit empty one resolve to the same row.
    assert env.step([("tool1", {"location": "sf"})]) == [
        ToolResult(output={"msg": "ok"})
    ]
    assert env.step([("tool1", {"location": "sf", "opts": {}})]) == [
        ToolResult(output={"msg": "ok"})
    ]


def test_ref_without_sibling_default_fills_through_target():
    """A bare `$ref` parent fills from the target's own `default`."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "$defs": {
                "Opts": {
                    "type": "object",
                    "default": {},
                    "properties": {"unit": {"type": "string", "default": "celsius"}},
                }
            },
            "properties": {"opts": {"$ref": "#/$defs/Opts"}},
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={"tool1": [ToolLookupEntry(input={}, output={"msg": "ok"})]},
        )
    )

    assert env._kwargs.lookup_table["tool1"][0].input == {"opts": {"unit": "celsius"}}
    assert env.step([("tool1", {})]) == [ToolResult(output={"msg": "ok"})]


def test_self_referential_ref_terminates():
    """A cyclic `$ref` must not hang construction or lookup."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "child": {"$ref": "#/$defs/Node"},
                        "label": {"type": "string", "default": "leaf"},
                    },
                }
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(input={"root": {"child": {}}}, output={"msg": "ok"})
                ]
            },
        )
    )

    assert env._kwargs.lookup_table["tool1"][0].input == {
        "root": {"label": "leaf", "child": {"label": "leaf"}}
    }


def test_dangling_ref_is_left_alone():
    """An unresolvable `$ref` contributes no defaults and is not an error."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {"opts": {"$ref": "#/$defs/Missing"}},
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [ToolLookupEntry(input={"opts": {}}, output={"msg": "ok"})]
            },
        )
    )

    assert env.step([("tool1", {"opts": {}})]) == [ToolResult(output={"msg": "ok"})]


def test_default_value_containing_schema_keywords_is_not_flagged():
    """A default's own value is data, so keywords inside it are not declarations."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "opts": {
                    "type": "object",
                    "default": {"$ref": "literal", "default": "literal"},
                }
            },
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={"opts": {"$ref": "literal", "default": "literal"}},
                        output={"msg": "ok"},
                    )
                ]
            },
        )
    )

    assert env.step([("tool1", {})]) == [ToolResult(output={"msg": "ok"})]


@pytest.mark.parametrize(
    "parameters,arguments",
    [
        pytest.param(
            {
                "$defs": {
                    "Item": {
                        "type": "object",
                        "properties": {"default": {"type": "string"}},
                        "required": ["default"],
                    }
                },
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Item"}}
                },
            },
            {"children": [{"default": "given"}]},
            id="property-named-default",
        ),
        pytest.param(
            # What `Field(json_schema_extra={"example": ...})` emits.
            {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "example": {"default": "data"}}
                },
            },
            {"value": "ok"},
            id="annotation-holding-data",
        ),
    ],
)
def test_non_subschema_positions_are_not_scanned(parameters, arguments):
    """Only subschema positions are walked, so data never reads as a declaration."""
    tool = _make_tool(parameters=parameters)

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [ToolLookupEntry(input=arguments, output={"msg": "ok"})]
            },
        )
    )

    assert env.step([("tool1", arguments)]) == [ToolResult(output={"msg": "ok"})]


def test_self_referential_default_does_not_expand_forever():
    """A `$ref` cycle whose recursive field defaults to `{}` must still terminate."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node", "default": {}}},
                }
            },
            "properties": {"root": {"$ref": "#/$defs/Node"}},
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [ToolLookupEntry(input={"root": {}}, output={"msg": "ok"})]
            },
        )
    )

    # The default is dropped at the cycle rather than nested forever, and both
    # sides of the match drop it identically.
    assert env._kwargs.lookup_table["tool1"][0].input == {"root": {}}
    assert env.step([("tool1", {"root": {}})]) == [ToolResult(output={"msg": "ok"})]


def test_whole_document_ref_resolves():
    """`{"$ref": "#"}` points at the root schema, not nowhere."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string", "default": "leaf"},
                "child": {"$ref": "#"},
            },
        }
    )

    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [ToolLookupEntry(input={"child": {}}, output={"msg": "ok"})]
            },
        )
    )

    assert env._kwargs.lookup_table["tool1"][0].input == {
        "label": "leaf",
        "child": {"label": "leaf"},
    }


def test_unreferenced_defs_are_not_flagged():
    """`$defs` entries nobody references are inert, not dead defaults."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "$defs": {"Unused": {"type": "object", "default": {"x": 1}}},
            "properties": {"id": {"type": "string"}},
        }
    )

    env = DeterministicEnvironment.from_params(_make_params(tools=[tool]))

    assert env.step([("tool1", {"id": "01"})]) == [ToolResult(output={"msg": "ok"})]


# --- step ---


def test_step_returns_matching_output():
    env = DeterministicEnvironment.from_params(
        _make_params(
            lookup_table={
                "tool1": [
                    ToolLookupEntry(input={"id": "01"}, output={"msg": "pending"}),
                    ToolLookupEntry(input={"id": "02"}, output={"msg": "delivered"}),
                ]
            }
        )
    )
    assert env.step([("tool1", {"id": "01"})]) == [
        ToolResult(output={"msg": "pending"})
    ]
    assert env.step([("tool1", {"id": "02"})]) == [
        ToolResult(output={"msg": "delivered"})
    ]
    # Batched: order preserved across multiple calls in one invocation.
    assert env.step([("tool1", {"id": "01"}), ("tool1", {"id": "02"})]) == [
        ToolResult(output={"msg": "pending"}),
        ToolResult(output={"msg": "delivered"}),
    ]


def test_step_no_match_raises_with_hint():
    env = DeterministicEnvironment.from_params(_make_params())
    with pytest.raises(ToolLookupError) as excinfo:
        env.step([("tool1", {"id": "99"})])
    msg = str(excinfo.value)
    assert "No deterministic output matches" in msg
    assert "tool1" in msg
    assert '"id": "01"' in msg  # configured inputs surfaced for self-correction


def test_step_supports_zero_arg_tool():
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("ping")],
            lookup_table={"ping": [ToolLookupEntry(input={}, output={})]},
        )
    )
    assert env.step([("ping", {})]) == [ToolResult(output={})]


def test_step_unknown_tool_raises():
    env = DeterministicEnvironment.from_params(_make_params())
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step([("missing", {"id": "01"})])


def test_step_matches_omitted_argument_to_explicit_default():
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "default": "fahrenheit"},
            },
            "required": ["location"],
        }
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={"location": "sf", "unit": "fahrenheit"},
                        output={"temperature": 65},
                    )
                ]
            },
        )
    )

    assert env.step([("tool1", {"location": "sf"})]) == [
        ToolResult(output={"temperature": 65})
    ]


def test_step_matches_explicit_argument_to_omitted_entry_default():
    """The entry side is filled too, so terse lookup rows match verbose calls."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "default": "fahrenheit"},
            },
            "required": ["location"],
        }
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={"location": "sf"},
                        output={"temperature": 65},
                    )
                ]
            },
        )
    )

    assert env.step([("tool1", {"location": "sf", "unit": "fahrenheit"})]) == [
        ToolResult(output={"temperature": 65})
    ]


def test_filled_list_default_is_not_aliased_to_schema():
    """Filled defaults are deep-copied, so a filled entry can't corrupt the schema."""
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {"tags": {"type": "array", "default": ["a"]}},
        }
    )
    entry = ToolLookupEntry(input={}, output={"ok": True})
    DeterministicEnvironment.from_params(
        _make_params(tools=[tool], lookup_table={"tool1": [entry]})
    )
    assert entry.input == {"tags": ["a"]}

    entry.input["tags"].append("MUTATED")

    assert tool.parameters["properties"]["tags"]["default"] == ["a"]


def test_step_recursively_fills_defaults_in_existing_object():
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "opts": {
                    "type": "object",
                    "properties": {
                        "unit": {"type": "string", "default": "fahrenheit"},
                        "verbose": {"type": "boolean", "default": False},
                    },
                },
            },
            "required": ["location"],
        }
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={
                            "location": "sf",
                            "opts": {"unit": "fahrenheit", "verbose": False},
                        },
                        output={"temperature": 65},
                    )
                ]
            },
        )
    )

    assert env.step([("tool1", {"location": "sf", "opts": {}})]) == [
        ToolResult(output={"temperature": 65})
    ]


def test_step_does_not_create_missing_object_without_default():
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "opts": {
                    "type": "object",
                    "properties": {
                        "unit": {"type": "string", "default": "fahrenheit"},
                    },
                },
            },
            "required": ["location"],
        }
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={"location": "sf"},
                        output={"source": "no-opts"},
                    ),
                    ToolLookupEntry(
                        input={
                            "location": "sf",
                            "opts": {"unit": "fahrenheit"},
                        },
                        output={"source": "opts"},
                    ),
                ]
            },
        )
    )

    assert env.step([("tool1", {"location": "sf"})]) == [
        ToolResult(output={"source": "no-opts"})
    ]
    assert env.step([("tool1", {"location": "sf", "opts": {}})]) == [
        ToolResult(output={"source": "opts"})
    ]


def test_step_creates_and_fills_missing_object_with_default():
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "opts": {
                    "type": "object",
                    "default": {},
                    "properties": {
                        "unit": {"type": "string", "default": "fahrenheit"},
                        "verbose": {"type": "boolean", "default": False},
                    },
                },
            },
            "required": ["location"],
        }
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={
                            "location": "sf",
                            "opts": {"unit": "fahrenheit", "verbose": False},
                        },
                        output={"temperature": 65},
                    )
                ]
            },
        )
    )

    assert env.step([("tool1", {"location": "sf"})]) == [
        ToolResult(output={"temperature": 65})
    ]


def test_step_default_filling_does_not_mutate_call_arguments():
    tool = _make_tool(
        parameters={
            "type": "object",
            "properties": {
                "opts": {
                    "type": "object",
                    "properties": {
                        "verbose": {"type": "boolean", "default": False},
                    },
                }
            },
        }
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "tool1": [
                    ToolLookupEntry(
                        input={"opts": {"verbose": False}},
                        output={"ok": True},
                    )
                ]
            },
        )
    )
    arguments = {"opts": {}}

    env.step([("tool1", arguments)])

    assert arguments == {"opts": {}}


@pytest.mark.parametrize(
    "output",
    [191.23, 42, ["a", "b"], "text", True, None],
)
def test_step_returns_non_dict_output(output):
    """Scalars, lists, and None round-trip through step() unchanged."""
    env = DeterministicEnvironment.from_params(
        _make_params(
            lookup_table={"tool1": [ToolLookupEntry(input={"id": "01"}, output=output)]}
        )
    )
    assert env.step([("tool1", {"id": "01"})]) == [ToolResult(output=output)]


# --- sample_grounding ---


def _grounded_env(
    n_entries: int = 10,
    sample_size: int = 3,
    seed: int | None = None,
) -> DeterministicEnvironment:
    """Build a DeterministicEnvironment with one grounded tool."""
    return DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("lookup")],
            lookup_table={
                "lookup": [
                    ToolLookupEntry(input={"id": str(i)}, output={"title": f"t-{i}"})
                    for i in range(n_entries)
                ]
            },
            grounding=GroundingConfig(
                sample_size=sample_size,
                seed=seed,
                tools={
                    "lookup": ToolGroundingConfig(fields=["id", "title"]),
                },
            ),
        )
    )


def test_sample_grounding_returns_facts():
    env = _grounded_env(n_entries=10, sample_size=3)
    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    for fact in facts:
        assert isinstance(fact, GroundingFact)
        assert set(fact.data.keys()) == {"id", "title"}


def test_sample_grounding_no_grounding_returns_empty():
    env = DeterministicEnvironment.from_params(_make_params())
    assert env.sample_grounding(n=5, rng=random.Random(0)) == []


def test_sample_grounding_only_grounded_tools_contribute():
    """Tools without an entry in grounding.tools contribute nothing."""
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("grounded"), _make_tool("plain")],
            lookup_table={
                "grounded": [
                    ToolLookupEntry(input={"id": "G1"}, output={"v": "g"}),
                ],
                "plain": [
                    ToolLookupEntry(input={"id": "P1"}, output={"v": "p"}),
                ],
            },
            grounding=GroundingConfig(
                sample_size=10,
                seed=0,
                tools={
                    "grounded": ToolGroundingConfig(fields=["id", "v"]),
                },
            ),
        )
    )
    facts = env.sample_grounding(n=10, rng=random.Random(0))
    assert len(facts) == 1
    assert facts[0].data == {"id": "G1", "v": "g"}


def test_sample_grounding_respects_tool_ids_filter():
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("a"), _make_tool("b")],
            lookup_table={
                "a": [ToolLookupEntry(input={"id": "A1"}, output={"v": "from_a"})],
                "b": [ToolLookupEntry(input={"id": "B1"}, output={"v": "from_b"})],
            },
            grounding=GroundingConfig(
                sample_size=10,
                tools={
                    "a": ToolGroundingConfig(fields=["id", "v"]),
                    "b": ToolGroundingConfig(fields=["id", "v"]),
                },
            ),
        )
    )
    facts = env.sample_grounding(n=10, rng=random.Random(0), tool_ids={"a"})
    assert len(facts) == 1
    assert facts[0].data == {"id": "A1", "v": "from_a"}


def test_sample_grounding_field_missing_in_row_is_dropped():
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("t")],
            lookup_table={
                "t": [ToolLookupEntry(input={"id": "X1"}, output={"v": "ok"})]
            },
            grounding=GroundingConfig(
                sample_size=1,
                tools={
                    "t": ToolGroundingConfig(fields=["id", "v", "missing"]),
                },
            ),
        )
    )
    facts = env.sample_grounding(n=1, rng=random.Random(0))
    assert facts[0].data == {"id": "X1", "v": "ok"}
    assert "missing" not in facts[0].data


def test_sample_grounding_merges_input_and_output():
    """Output values win over input values on key collision."""
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("lookup")],
            lookup_table={
                "lookup": [
                    ToolLookupEntry(
                        input={"id": "1", "note": "input-note"},
                        output={"note": "output-note", "title": "Dune"},
                    ),
                ]
            },
            grounding=GroundingConfig(
                sample_size=1,
                tools={
                    "lookup": ToolGroundingConfig(fields=["id", "note", "title"]),
                },
            ),
        )
    )
    facts = env.sample_grounding(n=1, rng=random.Random(0))
    assert facts[0].data == {"id": "1", "note": "output-note", "title": "Dune"}


def test_sample_grounding_includes_filled_defaults():
    """Entry inputs are normalized before projection, so defaults reach facts."""
    tool = _make_tool(
        "lookup",
        parameters={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "unit": {"type": "string", "default": "fahrenheit"},
                "locale": {"type": "string", "default": "en"},
            },
        },
    )
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[tool],
            lookup_table={
                "lookup": [ToolLookupEntry(input={"id": "1"}, output={"title": "Dune"})]
            },
            grounding=GroundingConfig(
                sample_size=1,
                tools={"lookup": ToolGroundingConfig(fields=["id", "unit", "title"])},
            ),
        )
    )
    facts = env.sample_grounding(n=1, rng=random.Random(0))

    assert facts[0].data == {"id": "1", "unit": "fahrenheit", "title": "Dune"}


def test_grounding_key_collision_warns(caplog):
    """Warn when a whitelisted grounding field is in both input and output."""
    with caplog.at_level(logging.WARNING, logger="oumi"):
        DeterministicEnvironment.from_params(
            _make_params(
                tools=[_make_tool("lookup")],
                lookup_table={
                    "lookup": [
                        ToolLookupEntry(
                            input={"id": "1", "note": "in"},
                            output={"note": "out", "title": "Dune"},
                        ),
                    ]
                },
                grounding=GroundingConfig(
                    sample_size=1,
                    tools={
                        "lookup": ToolGroundingConfig(fields=["id", "note", "title"]),
                    },
                ),
            )
        )
    assert any(
        "shadows the input" in rec.getMessage() and "note" in rec.getMessage()
        for rec in caplog.records
    )


def test_grounding_key_collision_outside_whitelist_no_warn(caplog):
    """A collision on a non-whitelisted key never reaches a fact, so no warning."""
    with caplog.at_level(logging.WARNING, logger="oumi"):
        DeterministicEnvironment.from_params(
            _make_params(
                tools=[_make_tool("lookup")],
                lookup_table={
                    "lookup": [
                        ToolLookupEntry(
                            input={"id": "1", "note": "in"},
                            output={"note": "out"},
                        ),
                    ]
                },
                grounding=GroundingConfig(
                    sample_size=1,
                    tools={"lookup": ToolGroundingConfig(fields=["id"])},
                ),
            )
        )
    assert not any("shadows the input" in rec.getMessage() for rec in caplog.records)


def test_sample_grounding_scalar_output_projects_input_only():
    """Non-dict outputs have no fields to project; ground on input alone."""
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("price")],
            lookup_table={
                "price": [
                    ToolLookupEntry(input={"id": "AAPL"}, output=191.23),
                    ToolLookupEntry(input={"id": "GOOG"}, output=2801.5),
                ]
            },
            grounding=GroundingConfig(
                sample_size=10,
                tools={"price": ToolGroundingConfig(fields=["id"])},
            ),
        )
    )
    facts = env.sample_grounding(n=10, rng=random.Random(0))
    assert len(facts) == 2
    assert sorted(f.data["id"] for f in facts) == ["AAPL", "GOOG"]
    for fact in facts:
        assert set(fact.data.keys()) == {"id"}


def test_sample_grounding_seeded_is_reproducible():
    env = _grounded_env(n_entries=20)
    a = env.sample_grounding(n=4, rng=random.Random(42))
    b = env.sample_grounding(n=4, rng=random.Random(42))
    assert [f.data["id"] for f in a] == [f.data["id"] for f in b]


def test_sample_grounding_truncates_when_n_exceeds_pool():
    env = _grounded_env(n_entries=3)
    facts = env.sample_grounding(n=10, rng=random.Random(0))
    assert len(facts) == 3


def test_sample_grounding_no_replacement_within_call():
    env = _grounded_env(n_entries=10)
    facts = env.sample_grounding(n=5, rng=random.Random(0))
    ids = [f.data["id"] for f in facts]
    assert len(set(ids)) == len(ids)


def test_sample_grounding_pools_across_tools():
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[_make_tool("a"), _make_tool("b")],
            lookup_table={
                "a": [ToolLookupEntry(input={"k": "a1"}, output={"v": "a1"})],
                "b": [
                    ToolLookupEntry(input={"k": "b1"}, output={"v": "b1"}),
                    ToolLookupEntry(input={"k": "b2"}, output={"v": "b2"}),
                ],
            },
            grounding=GroundingConfig(
                sample_size=3,
                tools={
                    "a": ToolGroundingConfig(fields=["k", "v"]),
                    "b": ToolGroundingConfig(fields=["k", "v"]),
                },
            ),
        )
    )
    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    assert sorted(f.data["k"] for f in facts) == ["a1", "b1", "b2"]
