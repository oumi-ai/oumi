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


def _make_tool(tool_id: str = "tool1") -> ToolParams:
    return ToolParams(id=tool_id, name=tool_id, description="A tool")


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
