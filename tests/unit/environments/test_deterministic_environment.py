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

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.configs.params.tool_params import ToolLookupError, ToolResult
from oumi.environments.deterministic_environment import (
    DeterministicEnvironment,
    DeterministicEnvironmentKwargs,
)
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)


def _make_tool(**overrides) -> DeterministicTool:
    defaults: dict = dict(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    defaults.update(overrides)
    return DeterministicTool(**defaults)


def _make_params(**overrides) -> EnvironmentParams:
    defaults: dict = dict(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        env_type="deterministic",
        tools=[_make_tool()],
    )
    defaults.update(overrides)
    return EnvironmentParams(**defaults)


def test_from_params_constructs_runtime_instance():
    env = DeterministicEnvironment.from_params(_make_params())
    assert isinstance(env, DeterministicEnvironment)
    assert isinstance(env._kwargs, DeterministicEnvironmentKwargs)


def test_rejects_env_kwargs():
    params = _make_params(env_kwargs={"unexpected": True})
    with pytest.raises(ValueError, match="does not accept env_kwargs"):
        DeterministicEnvironment.from_params(params)


def test_requires_deterministic_outputs_on_tool():
    params = _make_params(
        tools=[_make_tool(deterministic_outputs=[])],
    )
    with pytest.raises(ValueError, match="must have at least one"):
        DeterministicEnvironment.from_params(params)


def test_duplicate_deterministic_inputs_raises():
    outputs = [
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "a"}),
        DeterministicToolOutput(input={"id": "01"}, output={"msg": "b"}),
    ]
    params = _make_params(tools=[_make_tool(deterministic_outputs=outputs)])
    with pytest.raises(ValueError, match="duplicate"):
        DeterministicEnvironment.from_params(params)


def test_step_returns_matching_output():
    params = _make_params(
        tools=[
            _make_tool(
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "01"}, output={"msg": "pending"}
                    ),
                    DeterministicToolOutput(
                        input={"id": "02"}, output={"msg": "delivered"}
                    ),
                ]
            )
        ]
    )
    env = DeterministicEnvironment.from_params(params)
    assert env.step("tool1", {"id": "01"}) == ToolResult(output={"msg": "pending"})
    assert env.step("tool1", {"id": "02"}) == ToolResult(output={"msg": "delivered"})


def test_step_no_match_raises_with_hint():
    env = DeterministicEnvironment.from_params(_make_params())
    with pytest.raises(ToolLookupError) as excinfo:
        env.step("tool1", {"id": "99"})
    message = str(excinfo.value)
    assert "No deterministic output matches" in message
    assert "tool1" in message
    # The configured inputs are surfaced so the LLM can self-correct.
    assert '"id": "01"' in message


def test_step_supports_zero_arg_tool():
    params = _make_params(
        tools=[
            DeterministicTool(
                id="ping",
                name="Ping",
                description="Zero-arg tool.",
                deterministic_outputs=[
                    DeterministicToolOutput(input={}, output={}),
                ],
            )
        ]
    )
    env = DeterministicEnvironment.from_params(params)
    assert env.step("ping", {}) == ToolResult(output={})


def test_step_unknown_tool_raises():
    env = DeterministicEnvironment.from_params(_make_params())
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step("missing", {"id": "01"})


def test_from_params_coerces_raw_deterministic_outputs():
    tool = DeterministicTool(
        id="tool1",
        name="MyTool",
        description="A tool",
        deterministic_outputs=[{"input": {"id": "1"}, "output": {"msg": "ok"}}],  # type: ignore[list-item]
    )
    env = DeterministicEnvironment.from_params(_make_params(tools=[tool]))
    assert isinstance(
        env._params.tools[0].deterministic_outputs[0], DeterministicToolOutput
    )


# --- DeterministicEnvironment.sample_grounding ---


def _det_env_with_n_entries(n: int) -> DeterministicEnvironment:
    """Build a DeterministicEnvironment with a single tool containing n entries."""
    outputs = [
        DeterministicToolOutput(input={"id": str(i)}, output={"title": f"title-{i}"})
        for i in range(n)
    ]
    return DeterministicEnvironment.from_params(
        _make_params(
            tools=[
                DeterministicTool(
                    id="lookup",
                    name="Lookup",
                    description="Look up a book.",
                    deterministic_outputs=outputs,
                )
            ]
        )
    )


def test_sample_grounding_returns_n_facts():
    import random

    env = _det_env_with_n_entries(10)
    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    for fact in facts:
        assert isinstance(fact, GroundingFact)
        assert "id" in fact.data
        assert "title" in fact.data


def test_sample_grounding_merges_input_and_output_into_data():
    import random

    # Override-on-conflict: output values win over input values for matching keys.
    env = DeterministicEnvironment.from_params(
        _make_params(
            tools=[
                ToolParams(
                    id="lookup",
                    name="Lookup",
                    description="Look up.",
                    deterministic_outputs=[
                        DeterministicToolOutput(
                            input={"id": "1", "note": "input-note"},
                            output={"note": "output-note", "title": "Dune"},
                        ),
                    ],
                )
            ]
        )
    )
    facts = env.sample_grounding(n=1, rng=random.Random(0))
    assert len(facts) == 1
    assert facts[0].data == {
        "id": "1",
        "note": "output-note",  # output wins on key collision
        "title": "Dune",
    }


def test_sample_grounding_no_replacement_within_call():
    import random

    env = _det_env_with_n_entries(10)
    facts = env.sample_grounding(n=5, rng=random.Random(0))
    ids = [fact.data["id"] for fact in facts]
    assert len(set(ids)) == len(ids)


def test_sample_grounding_truncates_when_n_exceeds_pool():
    import random

    env = _det_env_with_n_entries(3)
    facts = env.sample_grounding(n=10, rng=random.Random(0))
    assert len(facts) == 3


def test_sample_grounding_seeded_rng_is_reproducible():
    import random

    env = _det_env_with_n_entries(20)
    facts_a = env.sample_grounding(n=4, rng=random.Random(42))
    facts_b = env.sample_grounding(n=4, rng=random.Random(42))
    ids_a = [fact.data["id"] for fact in facts_a]
    ids_b = [fact.data["id"] for fact in facts_b]
    assert ids_a == ids_b


def test_sample_grounding_different_seeds_differ():
    import random

    env = _det_env_with_n_entries(20)
    facts_a = env.sample_grounding(n=4, rng=random.Random(1))
    facts_b = env.sample_grounding(n=4, rng=random.Random(999))
    ids_a = sorted(fact.data["id"] for fact in facts_a)
    ids_b = sorted(fact.data["id"] for fact in facts_b)
    # With 20 entries and 4 picks, collision on both sets is vanishingly small.
    assert ids_a != ids_b


def test_sample_grounding_pools_across_tools():
    import random

    params = _make_params(
        tools=[
            DeterministicTool(
                id="tool_a",
                name="A",
                description="Tool A",
                deterministic_outputs=[
                    DeterministicToolOutput(input={"k": "a1"}, output={"v": "a1"})
                ],
            ),
            DeterministicTool(
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
    env = DeterministicEnvironment.from_params(params)
    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    keys = sorted(fact.data["k"] for fact in facts)
    assert keys == ["a1", "b1", "b2"]
