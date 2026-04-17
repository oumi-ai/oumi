# Deterministic Environment Grounding v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in grounding for `DeterministicEnvironment` that injects per-sample entity facts into the planner prompt so conversations reference real env state instead of hallucinated IDs.

**Architecture:** three-way contract — planner is grounded with a random subset of real entities, user persona inherits grounding through the planner's turn plans, assistant stays un-grounded. Each `BaseEnvironment` owns `GroundingConfig` and the `sample_grounding`/`describe_grounding` primitives. `ConversationSynthesizer` attaches facts onto each sample once upstream of planning, then renders a single block in `_create_planner_prompt`.

**Tech Stack:** Python 3.10+, dataclasses, pytest, `uv` for test execution.

**Spec reference:** `docs/superpowers/specs/2026-04-17-deterministic-env-grounding-design.md`.

---

## File map

| File | Change |
|------|--------|
| `src/oumi/environments/base_tool.py` | **Add** `GroundingConfig` dataclass. **Add** module-level `describe_grounding_default` rendering helper. |
| `src/oumi/environments/base_environment.py` | **Add** `grounding: GroundingConfig \| None = None` field. **Add** default `sample_grounding` (returns `[]`) and `describe_grounding` (delegates to the helper) methods. |
| `src/oumi/environments/deterministic_environment.py` | **Add** `sample_grounding` override that pools tool outputs and samples via `rng`. |
| `src/oumi/environments/__init__.py` | **Export** `GroundingConfig`. |
| `src/oumi/core/synthesis/conversation_synthesizer.py` | **Add** `_make_grounding_rng` and `_attach_grounding_facts` helpers. **Call** `_attach_grounding_facts` from `synthesize()`. **Add** grounding-block rendering in `_create_planner_prompt`. **Add** `{grounding_facts}` placeholder misuse warning in `_validate_tool_configuration`. |
| `tests/unit/core/configs/params/test_tool_params.py` | Unit tests for `GroundingConfig`, `BaseEnvironment.sample_grounding`/`describe_grounding` defaults, `DeterministicEnvironment.sample_grounding`, `describe_grounding_default` helper. |
| `tests/unit/core/synthesis/test_conversation_synthesizer.py` | Unit tests for `_make_grounding_rng`, `_attach_grounding_facts`, planner-prompt integration, placeholder misuse warning, end-to-end regression. |

## Test command reference

All `pytest` calls must be run via `uv run pytest ...` with `dangerouslyDisableSandbox: true` on the Bash invocation (the `uv` cache is outside the sandbox allowlist).

Default test command for this plan (deselects two pre-existing, unrelated failures):

```bash
uv run pytest \
  tests/unit/core/configs/params/test_tool_params.py \
  tests/unit/core/synthesis/test_conversation_synthesizer.py \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_format_persona_injects_tools_for_assistant_only \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_build_role_context_includes_tools_for_assistant \
  -q
```

Pre-plan baseline: **109 passed, 2 deselected**.

---

## Task 1: `GroundingConfig` dataclass

**Files:**
- Modify: `src/oumi/environments/base_tool.py` (add class at end of file, before the module footer)
- Test: `tests/unit/core/configs/params/test_tool_params.py` (append tests)

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/configs/params/test_tool_params.py`:

```python
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
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k grounding_config -v
```

Expected: `ImportError` or `AttributeError` — `GroundingConfig` does not exist.

- [ ] **Step 3: Implement `GroundingConfig`**

Add to `src/oumi/environments/base_tool.py`, placed after the existing `DeterministicToolOutput` class and before `ToolSchema`:

```python
@dataclass
class GroundingConfig(BaseParams):
    """Per-environment grounding configuration.

    When set on an environment, the ConversationSynthesizer samples facts from
    that environment and injects them into the planner prompt so turn plans
    reference real entities rather than hallucinated IDs.
    """

    sample_size: int = 3
    """Number of grounding facts sampled per conversation."""

    seed: int | None = None
    """If set, per-sample RNG is seeded from ``(seed + sample_index)`` for
    reproducibility. If None, grounding uses an unseeded ``random.Random``."""

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError(
                f"{type(self).__name__}.sample_size must be >= 1, "
                f"got {self.sample_size}."
            )
```

**Do NOT yet export it from `__init__.py`** — that happens in Task 5. For now, the import in the tests will need to succeed via the environments package; see Step 4.

- [ ] **Step 4: Make `GroundingConfig` importable so the tests run**

In `src/oumi/environments/__init__.py`, add `GroundingConfig` to the `from oumi.environments.base_tool import (...)` block and to `__all__`:

```python
from oumi.environments.base_tool import (
    DeterministicToolOutput,
    GroundingConfig,
    Tool,
    ToolArgumentError,
    ToolError,
    ToolLookupError,
    ToolResult,
    ToolSchema,
)
```

```python
__all__ = [
    "BaseEnvironment",
    "GroundingConfig",
    "Tool",
    "ToolArgumentError",
    "ToolError",
    "ToolLookupError",
    "ToolSchema",
    "ToolResult",
    "SyntheticEnvironment",
    "SyntheticStateParams",
    "DeterministicEnvironment",
    "DeterministicToolOutput",
]
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k grounding_config -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add src/oumi/environments/base_tool.py src/oumi/environments/__init__.py tests/unit/core/configs/params/test_tool_params.py
git commit -m "Add GroundingConfig dataclass for environment grounding"
```

---

## Task 2: `describe_grounding_default` rendering helper

**Files:**
- Modify: `src/oumi/environments/base_tool.py` (add helper function)
- Test: `tests/unit/core/configs/params/test_tool_params.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/configs/params/test_tool_params.py`:

```python
# --- describe_grounding_default ---


def test_describe_grounding_default_empty():
    from oumi.environments.base_tool import describe_grounding_default

    assert describe_grounding_default([]) == ""


def test_describe_grounding_default_single_fact():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [
        DeterministicToolOutput(
            input={"id": "42"}, output={"title": "Dune", "year": 1965}
        )
    ]
    rendered = describe_grounding_default(facts)
    assert rendered == '- id="42", title="Dune", year=1965'


def test_describe_grounding_default_multi_fact_preserves_order():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [
        DeterministicToolOutput(input={"id": "7"}, output={"title": "LotR"}),
        DeterministicToolOutput(input={"id": "42"}, output={"title": "Dune"}),
    ]
    rendered = describe_grounding_default(facts)
    assert rendered == (
        '- id="7", title="LotR"\n- id="42", title="Dune"'
    )


def test_describe_grounding_default_output_wins_on_key_conflict():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [
        DeterministicToolOutput(
            input={"id": "1", "note": "input-note"},
            output={"note": "output-note"},
        )
    ]
    rendered = describe_grounding_default(facts)
    assert 'note="output-note"' in rendered
    assert 'input-note' not in rendered


def test_describe_grounding_default_handles_non_string_values():
    from oumi.environments.base_tool import describe_grounding_default

    facts = [
        DeterministicToolOutput(
            input={"id": 42},
            output={"available": True, "count": 3, "rating": 4.5},
        )
    ]
    rendered = describe_grounding_default(facts)
    assert "id=42" in rendered
    assert "available=True" in rendered
    assert "count=3" in rendered
    assert "rating=4.5" in rendered
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k describe_grounding_default -v
```

Expected: `ImportError` on `describe_grounding_default`.

- [ ] **Step 3: Implement helper**

Add to `src/oumi/environments/base_tool.py` just after the `GroundingConfig` class:

```python
def _format_grounding_value(value: Any) -> str:
    """Render a fact value as a quoted string or bare literal."""
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def describe_grounding_default(facts: list[DeterministicToolOutput]) -> str:
    """Render grounding facts as a bulleted markdown block.

    Each fact's ``input`` and ``output`` dicts are flattened into a single
    key=value line. Output values win on key collisions.
    Returns "" for an empty fact list.
    """
    if not facts:
        return ""
    lines: list[str] = []
    for fact in facts:
        merged = {**fact.input, **fact.output}
        parts = [
            f"{key}={_format_grounding_value(value)}"
            for key, value in merged.items()
        ]
        lines.append(f"- {', '.join(parts)}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k describe_grounding_default -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oumi/environments/base_tool.py tests/unit/core/configs/params/test_tool_params.py
git commit -m "Add describe_grounding_default helper for bulleted fact rendering"
```

---

## Task 3: `BaseEnvironment.grounding` field + default methods

**Files:**
- Modify: `src/oumi/environments/base_environment.py`
- Test: `tests/unit/core/configs/params/test_tool_params.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/configs/params/test_tool_params.py`:

```python
# --- BaseEnvironment grounding defaults ---


def test_base_environment_grounding_field_defaults_to_none():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    assert env.grounding is None


def test_base_environment_default_sample_grounding_returns_empty():
    import random

    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    assert env.sample_grounding(n=5, rng=random.Random(0)) == []


def test_base_environment_default_describe_grounding_empty_list():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    assert env.describe_grounding([]) == ""


def test_base_environment_default_describe_grounding_delegates_to_helper():
    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        tools=[_make_synthetic_tool(id="answer")],
    )
    facts = [
        DeterministicToolOutput(input={"id": "42"}, output={"title": "Dune"})
    ]
    assert env.describe_grounding(facts) == '- id="42", title="Dune"'


def test_base_environment_accepts_grounding_in_constructor():
    from oumi.environments import GroundingConfig

    env = SyntheticEnvironment(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        system_prompt="Answer FAQs.",
        grounding=GroundingConfig(sample_size=2, seed=7),
        tools=[_make_synthetic_tool(id="answer")],
    )
    assert env.grounding is not None
    assert env.grounding.sample_size == 2
    assert env.grounding.seed == 7
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k "base_environment_grounding or base_environment_default_sample or base_environment_default_describe or base_environment_accepts_grounding" -v
```

Expected: failures — `grounding` is not a field on `BaseEnvironment`, methods don't exist.

- [ ] **Step 3: Implement field and default methods**

In `src/oumi/environments/base_environment.py`:

Update the imports block:

```python
from oumi.environments.base_tool import (
    DeterministicToolOutput,
    GroundingConfig,
    Tool,
    ToolResult,
    describe_grounding_default,
)
```

Add `import random` at the top (alongside other stdlib imports).

Add the `grounding` field to the dataclass body, right after `tools`:

```python
@dataclass
class BaseEnvironment(BaseParams, ABC):
    """Abstract base class for tool environments."""

    _registry: ClassVar[dict[str, type[BaseEnvironment]]] = {}

    id: str
    name: str
    description: str
    tools: list[Tool] = field(default_factory=list)
    grounding: GroundingConfig | None = None
```

Add two methods at the end of the class, just before `create` (or wherever they fit cleanly):

```python
    def sample_grounding(
        self, n: int, *, rng: random.Random
    ) -> list[DeterministicToolOutput]:
        """Sample n grounding facts from this environment.

        Default: returns an empty list. Subclasses that support grounding
        (currently only ``DeterministicEnvironment``) override this.
        """
        return []

    def describe_grounding(
        self, facts: list[DeterministicToolOutput]
    ) -> str:
        """Render grounding facts as a bulleted markdown block.

        Default implementation flattens each fact's input and output dicts
        (output wins on key collisions) into a single bullet line. Suitable
        for any dict-shaped fact. Subclasses may override for custom
        rendering.
        """
        return describe_grounding_default(facts)
```

Coercion: if raw config dicts include `grounding`, it needs to be converted to a `GroundingConfig`. Add that to `BaseEnvironment.create` inside the filter that extracts fields into the constructor call. Locate the existing block:

```python
        return environment_cls(
            **{key: value for key, value in raw.items() if key in init_fields}
        )
```

Replace with:

```python
        coerced: dict[str, Any] = {
            key: value for key, value in raw.items() if key in init_fields
        }
        if isinstance(coerced.get("grounding"), dict):
            coerced["grounding"] = GroundingConfig(**coerced["grounding"])
        return environment_cls(**coerced)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k "base_environment_grounding or base_environment_default_sample or base_environment_default_describe or base_environment_accepts_grounding" -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full file to confirm no regression**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -q
```

Expected: all tests pass (previous task count + 5).

- [ ] **Step 6: Commit**

```bash
git add src/oumi/environments/base_environment.py tests/unit/core/configs/params/test_tool_params.py
git commit -m "Add grounding field and default methods to BaseEnvironment"
```

---

## Task 4: `DeterministicEnvironment.sample_grounding` override

**Files:**
- Modify: `src/oumi/environments/deterministic_environment.py`
- Test: `tests/unit/core/configs/params/test_tool_params.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/configs/params/test_tool_params.py`:

```python
# --- DeterministicEnvironment.sample_grounding ---


def _det_env_with_n_entries(n: int) -> DeterministicEnvironment:
    """Build a DeterministicEnvironment with a single tool containing n entries."""
    outputs = [
        DeterministicToolOutput(
            input={"id": str(i)}, output={"title": f"title-{i}"}
        )
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
        assert isinstance(fact, DeterministicToolOutput)


def test_deterministic_sample_grounding_no_replacement_within_call():
    import random

    env = _det_env_with_n_entries(10)
    facts = env.sample_grounding(n=5, rng=random.Random(0))
    ids = [fact.input["id"] for fact in facts]
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
    ids_a = [fact.input["id"] for fact in facts_a]
    ids_b = [fact.input["id"] for fact in facts_b]
    assert ids_a == ids_b


def test_deterministic_sample_grounding_different_seeds_differ():
    import random

    env = _det_env_with_n_entries(20)
    facts_a = env.sample_grounding(n=4, rng=random.Random(1))
    facts_b = env.sample_grounding(n=4, rng=random.Random(999))
    ids_a = sorted(fact.input["id"] for fact in facts_a)
    ids_b = sorted(fact.input["id"] for fact in facts_b)
    # With 20 entries and 4 picks, collision on both sets is vanishingly small.
    assert ids_a != ids_b


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
                    DeterministicToolOutput(
                        input={"k": "a1"}, output={"v": "a1"}
                    )
                ],
            ),
            Tool(
                id="tool_b",
                name="B",
                description="Tool B",
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"k": "b1"}, output={"v": "b1"}
                    ),
                    DeterministicToolOutput(
                        input={"k": "b2"}, output={"v": "b2"}
                    ),
                ],
            ),
        ],
    )
    import random

    facts = env.sample_grounding(n=3, rng=random.Random(0))
    assert len(facts) == 3
    keys = sorted(fact.input["k"] for fact in facts)
    assert keys == ["a1", "b1", "b2"]
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k deterministic_sample_grounding -v
```

Expected: all fail — the default `sample_grounding` returns `[]`.

- [ ] **Step 3: Implement override**

In `src/oumi/environments/deterministic_environment.py`:

Update imports:

```python
import json
import random
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import (
    DeterministicToolOutput,
    Tool,
    ToolLookupError,
    ToolResult,
)
```

Add the override method anywhere inside the `DeterministicEnvironment` class (e.g., just after `step`):

```python
    def sample_grounding(
        self, n: int, *, rng: random.Random
    ) -> list[DeterministicToolOutput]:
        """Sample grounding facts from the pool of deterministic outputs.

        Pools every ``DeterministicToolOutput`` across every tool owned by
        this environment, then draws ``min(n, len(pool))`` entries without
        replacement using the supplied RNG. Silent truncation — the
        synthesizer is responsible for surfacing a warning when applicable.
        """
        pool: list[DeterministicToolOutput] = [
            entry
            for tool in self.tools
            for entry in tool.deterministic_outputs
        ]
        return rng.sample(pool, min(n, len(pool)))
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -k deterministic_sample_grounding -v
```

Expected: 6 passed.

- [ ] **Step 5: Run full tool_params file to confirm no regression**

```bash
uv run pytest tests/unit/core/configs/params/test_tool_params.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/oumi/environments/deterministic_environment.py tests/unit/core/configs/params/test_tool_params.py
git commit -m "Override sample_grounding in DeterministicEnvironment"
```

---

## Task 5: `_make_grounding_rng` helper on `ConversationSynthesizer`

**Files:**
- Modify: `src/oumi/core/synthesis/conversation_synthesizer.py`
- Test: `tests/unit/core/synthesis/test_conversation_synthesizer.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/synthesis/test_conversation_synthesizer.py` (before the final batch of `synthesize_raises_*` tests — location doesn't strictly matter, just keep it near other single-method tests):

```python
# --- _make_grounding_rng ---


def test_make_grounding_rng_unseeded_returns_fresh_random(mock_inference_config):
    import random as _random

    synth = _make_synthesizer(mock_inference_config)
    rng = synth._make_grounding_rng(seed=None, sample_index=0)
    assert isinstance(rng, _random.Random)


def test_make_grounding_rng_seeded_is_reproducible(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    rng_a = synth._make_grounding_rng(seed=42, sample_index=3)
    rng_b = synth._make_grounding_rng(seed=42, sample_index=3)
    # Same seed + same sample_index produces the same stream.
    assert [rng_a.random() for _ in range(5)] == [rng_b.random() for _ in range(5)]


def test_make_grounding_rng_seeded_varies_across_sample_indices(
    mock_inference_config,
):
    synth = _make_synthesizer(mock_inference_config)
    rng_0 = synth._make_grounding_rng(seed=42, sample_index=0)
    rng_1 = synth._make_grounding_rng(seed=42, sample_index=1)
    assert [rng_0.random() for _ in range(5)] != [rng_1.random() for _ in range(5)]
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k make_grounding_rng -v
```

Expected: `AttributeError: 'ConversationSynthesizer' object has no attribute '_make_grounding_rng'`.

- [ ] **Step 3: Implement helper**

In `src/oumi/core/synthesis/conversation_synthesizer.py`:

Add `import random` at the top of the file (alongside the other stdlib imports — check whether `random` is already imported; if not, add it).

Add the method inside `ConversationSynthesizer`, near the other `_make_*` helpers or grouped logically (e.g., just above `_run_single_tool_call`):

```python
    def _make_grounding_rng(
        self, seed: int | None, sample_index: int
    ) -> random.Random:
        """Build the RNG used for sampling grounding facts for one sample.

        Unseeded (``seed=None``) uses the default ``random.Random()`` with
        entropy from the OS, matching the non-reproducible behavior used by
        ``DatasetPlanner`` for sampled attributes. Seeded mode makes each
        sample's facts deterministic from ``(seed + sample_index)``.
        """
        if seed is None:
            return random.Random()
        return random.Random(seed + sample_index)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k make_grounding_rng -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oumi/core/synthesis/conversation_synthesizer.py tests/unit/core/synthesis/test_conversation_synthesizer.py
git commit -m "Add _make_grounding_rng helper on ConversationSynthesizer"
```

---

## Task 6: `_attach_grounding_facts` method

**Files:**
- Modify: `src/oumi/core/synthesis/conversation_synthesizer.py`
- Test: `tests/unit/core/synthesis/test_conversation_synthesizer.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/synthesis/test_conversation_synthesizer.py`:

```python
# --- _attach_grounding_facts ---


def _grounded_det_env(
    env_id: str = "env1",
    tool_id: str = "lookup",
    n_entries: int = 10,
    sample_size: int = 3,
    seed: int | None = None,
) -> DeterministicEnvironment:
    from oumi.environments import GroundingConfig

    outputs = [
        DeterministicToolOutput(
            input={"id": str(i)},
            output={"title": f"title-{i}"},
        )
        for i in range(n_entries)
    ]
    return DeterministicEnvironment(
        id=env_id,
        name=env_id,
        description=f"env {env_id}",
        grounding=GroundingConfig(sample_size=sample_size, seed=seed),
        tools=[
            Tool(
                id=tool_id,
                name=tool_id,
                description="Look up an id.",
                deterministic_outputs=outputs,
            )
        ],
    )


def _grounded_env_config(**env_kwargs) -> EnvironmentConfig:
    return EnvironmentConfig(environments=[_grounded_det_env(**env_kwargs)])


def test_attach_grounding_facts_noop_without_env_config(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    samples = [{"a": 1}, {"b": 2}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
    )

    synth._attach_grounding_facts(samples, attr)

    assert "grounding_facts" not in samples[0]
    assert "grounding_facts" not in samples[1]


def test_attach_grounding_facts_noop_when_no_env_has_grounding(
    mock_inference_config,
):
    env_config = _tool_env_config()  # no grounding on the env
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    samples = [{}, {}]
    attr = _tool_multiturn_attr()

    synth._attach_grounding_facts(samples, attr)

    assert "grounding_facts" not in samples[0]
    assert "grounding_facts" not in samples[1]


def test_attach_grounding_facts_populates_samples(mock_inference_config):
    env_config = _grounded_env_config(n_entries=10, sample_size=3, seed=42)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    samples = [{}, {}, {}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    synth._attach_grounding_facts(samples, attr)

    for sample in samples:
        assert "grounding_facts" in sample
        assert len(sample["grounding_facts"]) == 3
        for fact in sample["grounding_facts"]:
            assert isinstance(fact, DeterministicToolOutput)


def test_attach_grounding_facts_seeded_is_reproducible(mock_inference_config):
    env_config_a = _grounded_env_config(n_entries=20, sample_size=4, seed=7)
    env_config_b = _grounded_env_config(n_entries=20, sample_size=4, seed=7)
    synth_a = _make_synthesizer(
        mock_inference_config, environment_config=env_config_a
    )
    synth_b = _make_synthesizer(
        mock_inference_config, environment_config=env_config_b
    )
    samples_a = [{}, {}, {}]
    samples_b = [{}, {}, {}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    synth_a._attach_grounding_facts(samples_a, attr)
    synth_b._attach_grounding_facts(samples_b, attr)

    for a, b in zip(samples_a, samples_b):
        assert [f.input["id"] for f in a["grounding_facts"]] == [
            f.input["id"] for f in b["grounding_facts"]
        ]


def test_attach_grounding_facts_seeded_different_samples_differ(
    mock_inference_config,
):
    env_config = _grounded_env_config(n_entries=50, sample_size=3, seed=7)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    samples = [{}, {}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    synth._attach_grounding_facts(samples, attr)

    ids_0 = sorted(f.input["id"] for f in samples[0]["grounding_facts"])
    ids_1 = sorted(f.input["id"] for f in samples[1]["grounding_facts"])
    assert ids_0 != ids_1


def test_attach_grounding_facts_respects_available_environments_scoping(
    mock_inference_config,
):
    env_a = _grounded_det_env(
        env_id="env_a", tool_id="tool_a", n_entries=5, sample_size=2, seed=1
    )
    env_b = _grounded_det_env(
        env_id="env_b", tool_id="tool_b", n_entries=5, sample_size=2, seed=2
    )
    env_config = EnvironmentConfig(environments=[env_a, env_b])
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    samples = [{}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        available_environments=["env_a"],
        available_tools=["tool_a"],
    )

    synth._attach_grounding_facts(samples, attr)

    # Only env_a contributes facts (sample_size=2).
    assert len(samples[0]["grounding_facts"]) == 2


def test_attach_grounding_facts_concatenates_across_multiple_envs(
    mock_inference_config,
):
    env_a = _grounded_det_env(
        env_id="env_a", tool_id="tool_a", n_entries=5, sample_size=2, seed=1
    )
    env_b = _grounded_det_env(
        env_id="env_b", tool_id="tool_b", n_entries=5, sample_size=3, seed=2
    )
    env_config = EnvironmentConfig(environments=[env_a, env_b])
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    samples = [{}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        # available_environments=None → all envs in config are in scope
        available_tools=["tool_a", "tool_b"],
    )

    synth._attach_grounding_facts(samples, attr)

    assert len(samples[0]["grounding_facts"]) == 5  # 2 + 3


def test_attach_grounding_facts_truncation_emits_logger_warning(
    mock_inference_config, caplog
):
    import logging

    env_config = _grounded_env_config(n_entries=2, sample_size=5, seed=1)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    samples = [{}, {}]
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._attach_grounding_facts(samples, attr)

    truncation_records = [
        rec for rec in caplog.records if "sample_size" in rec.getMessage()
    ]
    # Exactly one warning per env per synthesize invocation, even with 2 samples.
    assert len(truncation_records) == 1
    assert "env1" in truncation_records[0].getMessage()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k attach_grounding_facts -v
```

Expected: `AttributeError` — `_attach_grounding_facts` doesn't exist.

- [ ] **Step 3: Implement `_attach_grounding_facts`**

In `src/oumi/core/synthesis/conversation_synthesizer.py`:

Add the import at the top (if not already there):

```python
from oumi.environments import (
    DeterministicToolOutput,
    Tool,
    ToolArgumentError,
    ToolError,
)
```

(Keep the existing `from oumi.environments import Tool, ToolArgumentError, ToolError` line replaced by the expanded version above; only add `DeterministicToolOutput`.)

Add the method inside `ConversationSynthesizer`, placed just after `_make_grounding_rng`:

```python
    def _attach_grounding_facts(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> None:
        """Attach per-sample grounding facts drawn from grounded envs in scope.

        Writes ``sample["grounding_facts"]`` as a flat list concatenated
        across all envs in scope that declare a ``GroundingConfig``. No-op
        when ``environment_config`` is absent or no env in scope declares
        grounding. Emits one ``logger.warning`` per env when truncation
        occurs (sample_size > pool_size).
        """
        if self._environment_config is None:
            return

        scoped_env_ids = (
            set(multiturn_attribute.available_environments)
            if multiturn_attribute.available_environments
            else {env.id for env in self._environment_config.environments}
        )
        grounding_envs = [
            env
            for env in self._environment_config.environments
            if env.id in scoped_env_ids and env.grounding is not None
        ]
        if not grounding_envs:
            return

        warned_envs: set[str] = set()
        for sample_index, sample in enumerate(samples):
            facts: list[DeterministicToolOutput] = []
            for env in grounding_envs:
                assert env.grounding is not None  # narrowed above
                rng = self._make_grounding_rng(env.grounding.seed, sample_index)
                sampled = env.sample_grounding(
                    n=env.grounding.sample_size, rng=rng
                )
                if (
                    len(sampled) < env.grounding.sample_size
                    and env.id not in warned_envs
                ):
                    logger.warning(
                        "Grounding sample_size=%d exceeds pool size for "
                        "environment '%s'; truncating to %d facts.",
                        env.grounding.sample_size,
                        env.id,
                        len(sampled),
                    )
                    warned_envs.add(env.id)
                facts.extend(sampled)
            sample["grounding_facts"] = facts
```

Note: the `assert env.grounding is not None` is the filtering invariant from `grounding_envs`; kept to appease type checkers.

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k attach_grounding_facts -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oumi/core/synthesis/conversation_synthesizer.py tests/unit/core/synthesis/test_conversation_synthesizer.py
git commit -m "Add _attach_grounding_facts to populate per-sample facts"
```

---

## Task 7: Call `_attach_grounding_facts` from `synthesize()` and render planner block

**Files:**
- Modify: `src/oumi/core/synthesis/conversation_synthesizer.py`
- Test: `tests/unit/core/synthesis/test_conversation_synthesizer.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/synthesis/test_conversation_synthesizer.py`:

```python
# --- Planner prompt grounding injection ---


def test_create_planner_prompt_injects_grounding_block_when_facts_present(
    mock_inference_config,
):
    env_config = _grounded_env_config(n_entries=10, sample_size=2, seed=1)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
        "grounding_facts": [
            DeterministicToolOutput(
                input={"id": "42"}, output={"title": "Dune"}
            ),
            DeterministicToolOutput(
                input={"id": "7"}, output={"title": "LotR"}
            ),
        ],
    }

    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan in these specific entities" in planner_user_msg
    assert '- id="42", title="Dune"' in planner_user_msg
    assert '- id="7", title="LotR"' in planner_user_msg
    assert (
        "Your turn plans must only reference these entities"
        in planner_user_msg
    )


def test_create_planner_prompt_no_grounding_block_when_facts_absent(
    mock_inference_config,
):
    env_config = _tool_env_config()  # no grounding configured
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    attr = _tool_multiturn_attr()
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
    }

    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan" not in planner_user_msg


def test_create_planner_prompt_empty_grounding_facts_omits_block(
    mock_inference_config,
):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    attr = _tool_multiturn_attr()
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
        "grounding_facts": [],
    }

    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan" not in planner_user_msg


def test_synthesize_invokes_attach_grounding_facts(mock_inference_config):
    """End-to-end: synthesize() calls _attach_grounding_facts before planning."""
    env_config = _grounded_env_config(n_entries=10, sample_size=2, seed=5)
    # Script the inference engine: 1 planner response, 4 conversation turns.
    plan_json = '[{"turn": 1, "instruction": "a"}, {"turn": 2, "instruction": "b"}]'
    engine = _scripted_inference_engine(
        [
            [plan_json],  # planner call
            ["user turn 1"],  # user turn 1
            ["assistant final turn 2"],  # assistant turn 2 (non-tool)
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config,
        environment_config=env_config,
        inference_engine=engine,
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    samples = [{}]
    result = synth.synthesize(samples, attr)

    # The sample dict passed in should have grounding_facts attached.
    assert "grounding_facts" in samples[0]
    assert len(samples[0]["grounding_facts"]) == 2
    # Basic regression: result shape is preserved.
    assert len(result) == 1
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k "planner_prompt_injects_grounding or planner_prompt_no_grounding or planner_prompt_empty_grounding or synthesize_invokes_attach" -v
```

Expected: failures — the grounding block isn't rendered, `_attach_grounding_facts` isn't called from `synthesize()`.

- [ ] **Step 3: Wire `_attach_grounding_facts` into `synthesize()`**

In `src/oumi/core/synthesis/conversation_synthesizer.py`, locate `synthesize()`. Find the existing lines:

```python
        self._validate_roles(multiturn_attributes)
        self._validate_tool_configuration(multiturn_attributes)
```

Add the grounding attachment after them:

```python
        self._validate_roles(multiturn_attributes)
        self._validate_tool_configuration(multiturn_attributes)

        logger.info(
            f"Synthesizing {len(samples)} conversations for "
            f"attribute '{multiturn_attributes.id}'"
        )
        available_tools = self._resolve_available_tools(multiturn_attributes)
        if available_tools:
            logger.debug(
                "Resolved tools for '%s': %s",
                multiturn_attributes.id,
                [tool.id for tool in available_tools],
            )

        self._attach_grounding_facts(samples, multiturn_attributes)
```

*(Note: the exact insertion point is just before the existing `samples = self._plan_samples(...)` line. Preserve existing logging/logging call structure; only add the new `_attach_grounding_facts` call.)*

- [ ] **Step 4: Render grounding block in `_create_planner_prompt`**

In `src/oumi/core/synthesis/conversation_synthesizer.py`, locate `_create_planner_prompt`. Find the block that builds `base_prompt`; it ends with:

```python
        if role_context:
            base_prompt += f"\nRole context:\n{role_context}\n"

        if multiturn_attribute.conversation_planner:
            formatted_planner = self._formatter.format(
                sample,
                multiturn_attribute.conversation_planner,
                missing_values_allowed=False,
            )
            base_prompt += f"\nAdditional instructions: {formatted_planner}\n"

        base_prompt += "\nOutput ONLY the JSON array. No markdown. No other text."
```

Insert the grounding block *between* the `Role context` block and the `conversation_planner` block:

```python
        if role_context:
            base_prompt += f"\nRole context:\n{role_context}\n"

        grounding_facts = sample.get("grounding_facts") or []
        if grounding_facts:
            # All facts share the same dict-shaped rendering; pick the first
            # grounded env's describer (identical for v1 since every env uses
            # the default). Future envs with custom describers should be
            # revisited here.
            describer_env = next(
                (
                    env
                    for env in (self._environment_config.environments if self._environment_config else [])
                    if env.grounding is not None
                ),
                None,
            )
            if describer_env is not None:
                block = describer_env.describe_grounding(grounding_facts)
            else:
                block = describe_grounding_default(grounding_facts)
            base_prompt += (
                "\nGround this plan in these specific entities:\n"
                f"{block}\n"
                "Your turn plans must only reference these entities.\n"
            )

        if multiturn_attribute.conversation_planner:
            ...
```

Also update the imports in `conversation_synthesizer.py`:

```python
from oumi.environments import (
    DeterministicToolOutput,
    Tool,
    ToolArgumentError,
    ToolError,
)
from oumi.environments.base_tool import describe_grounding_default
```

- [ ] **Step 5: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k "planner_prompt_injects_grounding or planner_prompt_no_grounding or planner_prompt_empty_grounding or synthesize_invokes_attach" -v
```

Expected: 4 passed.

- [ ] **Step 6: Run the two pre-existing files' full sweep (deselecting the known pre-existing failures)**

```bash
uv run pytest \
  tests/unit/core/configs/params/test_tool_params.py \
  tests/unit/core/synthesis/test_conversation_synthesizer.py \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_format_persona_injects_tools_for_assistant_only \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_build_role_context_includes_tools_for_assistant \
  -q
```

Expected: previous count + all new tests passing. No regressions.

- [ ] **Step 7: Commit**

```bash
git add src/oumi/core/synthesis/conversation_synthesizer.py tests/unit/core/synthesis/test_conversation_synthesizer.py
git commit -m "Wire grounding into synthesize() and planner prompt"
```

---

## Task 8: Warning when `{grounding_facts}` appears in user/assistant persona template

**Files:**
- Modify: `src/oumi/core/synthesis/conversation_synthesizer.py`
- Test: `tests/unit/core/synthesis/test_conversation_synthesizer.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/core/synthesis/test_conversation_synthesizer.py`:

```python
# --- {grounding_facts} placeholder misuse warning ---


def test_validate_tool_configuration_warns_on_grounding_placeholder_in_user(
    mock_inference_config, caplog
):
    import logging

    env_config = _grounded_env_config(n_entries=5, sample_size=2, seed=1)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user interested in {grounding_facts}.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._validate_tool_configuration(attr)

    warnings = [
        rec
        for rec in caplog.records
        if "grounding is planner-only" in rec.getMessage()
        or "grounding_facts" in rec.getMessage()
    ]
    assert len(warnings) >= 1
    assert "user" in warnings[0].getMessage().lower()


def test_validate_tool_configuration_warns_on_grounding_placeholder_in_assistant(
    mock_inference_config, caplog
):
    import logging

    env_config = _grounded_env_config(n_entries=5, sample_size=2, seed=1)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You know these entities: {grounding_facts}.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._validate_tool_configuration(attr)

    warnings = [
        rec
        for rec in caplog.records
        if "grounding_facts" in rec.getMessage()
    ]
    assert len(warnings) >= 1
    assert "assistant" in warnings[0].getMessage().lower()


def test_validate_tool_configuration_no_warning_when_placeholder_absent(
    mock_inference_config, caplog
):
    import logging

    env_config = _grounded_env_config(n_entries=5, sample_size=2, seed=1)
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._validate_tool_configuration(attr)

    grounding_warnings = [
        rec
        for rec in caplog.records
        if "grounding_facts" in rec.getMessage()
    ]
    assert grounding_warnings == []
```

- [ ] **Step 2: Run tests, expect failure**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k validate_tool_configuration_warns_on_grounding -v
```

Expected: no warnings emitted because the check doesn't exist yet.

- [ ] **Step 3: Implement warning in `_validate_tool_configuration`**

In `src/oumi/core/synthesis/conversation_synthesizer.py`, locate `_validate_tool_configuration`. Its existing body raises `ValueError` on tool/env mismatch. Extend it with the placeholder check at the end of the method:

```python
    def _validate_tool_configuration(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> None:
        """Validate that tool/environment declarations have a backing
        environment_config, and warn about grounding placeholder misuse."""
        declares_tools = bool(multiturn_attribute.available_tools) or bool(
            multiturn_attribute.available_environments
        )
        if declares_tools and self._environment_config is None:
            raise ValueError(
                f"MultiTurnAttribute '{multiturn_attribute.id}' declares "
                f"available_tools/available_environments but no environment_config "
                f"was provided to ConversationSynthesizer."
            )

        # grounding_facts is planner-only. Warn if a config author placed the
        # placeholder in a user or assistant persona template.
        for role, persona in multiturn_attribute.role_instruction_messages.items():
            if not isinstance(persona, str):
                continue
            if "{grounding_facts}" in persona and role in (
                Role.USER,
                Role.ASSISTANT,
            ):
                logger.warning(
                    "MultiTurnAttribute '%s' references {grounding_facts} in "
                    "the %s persona template. grounding is planner-only; "
                    "placing {grounding_facts} in user/assistant templates "
                    "defeats its purpose and may leak env state to roles that "
                    "should not see it.",
                    multiturn_attribute.id,
                    role.value,
                )
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k validate_tool_configuration_warns_on_grounding -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/oumi/core/synthesis/conversation_synthesizer.py tests/unit/core/synthesis/test_conversation_synthesizer.py
git commit -m "Warn on {grounding_facts} placeholder in user/assistant personas"
```

---

## Task 9: Regression + end-to-end integration test

**Files:**
- Test: `tests/unit/core/synthesis/test_conversation_synthesizer.py`
- (No production code changes.)

- [ ] **Step 1: Add failing integration tests**

Append to `tests/unit/core/synthesis/test_conversation_synthesizer.py`:

```python
# --- Regression + integration: planner prompt byte-equivalence when ungrounded ---


def test_create_planner_prompt_byte_identical_when_no_grounding(
    mock_inference_config,
):
    """Regression: when no env in scope has grounding, planner prompt is
    unchanged from the pre-grounding baseline."""
    env_config = _tool_env_config()  # no grounding
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config
    )
    attr = _tool_multiturn_attr()
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
    }

    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content

    # The three grounding marker substrings must be absent.
    assert "Ground this plan" not in planner_user_msg
    assert "turn plans must only reference" not in planner_user_msg
    assert "{grounding_facts}" not in planner_user_msg  # not interpolated


def test_end_to_end_grounded_conversation_uses_sampled_entity_ids(
    mock_inference_config,
):
    """Full synthesize() flow: grounded planner prompt drives the plan
    through a scripted inference engine, confirming the facts make it all
    the way into the planner call."""
    env_config = _grounded_env_config(n_entries=10, sample_size=3, seed=13)

    captured_planner_prompts: list[str] = []

    def infer_side_effect(conversations, **kwargs):
        for conv in conversations:
            last = conv.messages[-1].content
            if isinstance(last, str) and "Plan" in last:
                captured_planner_prompts.append(last)
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content=(
                            '[{"turn": 1, "instruction": "a"}, '
                            '{"turn": 2, "instruction": "b"}]'
                        ),
                    )
                ]
                if any(
                    "Plan" in (m.content if isinstance(m.content, str) else "")
                    for m in conv.messages
                )
                else [Message(role=Role.ASSISTANT, content="turn content")]
            )
            for conv in conversations
        ]

    engine = Mock()
    engine.infer.side_effect = infer_side_effect

    synth = _make_synthesizer(
        mock_inference_config,
        environment_config=env_config,
        inference_engine=engine,
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )
    samples = [{}]
    synth.synthesize(samples, attr)

    # The planner was invoked with a grounding block.
    assert captured_planner_prompts, "planner prompt was never captured"
    planner_prompt = captured_planner_prompts[0]
    assert "Ground this plan in these specific entities" in planner_prompt
    # Every ID mentioned in the block must be one of the 10 configured inputs.
    configured_ids = {str(i) for i in range(10)}
    facts = samples[0]["grounding_facts"]
    for fact in facts:
        fact_id = fact.input["id"]
        assert fact_id in configured_ids
        assert f'id="{fact_id}"' in planner_prompt
```

- [ ] **Step 2: Run tests, expect them to pass immediately**

```bash
uv run pytest tests/unit/core/synthesis/test_conversation_synthesizer.py -k "byte_identical_when_no_grounding or end_to_end_grounded_conversation" -v
```

Expected: 2 passed. (These exercise already-landed code; they're *verification* tests, not TDD drivers.) If either fails, go back and fix the implementation rather than the test.

- [ ] **Step 3: Run the full plan-affected sweep**

```bash
uv run pytest \
  tests/unit/core/configs/params/test_tool_params.py \
  tests/unit/core/synthesis/test_conversation_synthesizer.py \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_format_persona_injects_tools_for_assistant_only \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_build_role_context_includes_tools_for_assistant \
  -q
```

Expected: all tests pass (109 pre-plan + ~35 new = ~144 passed, 2 deselected).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/core/synthesis/test_conversation_synthesizer.py
git commit -m "Add regression and end-to-end tests for grounding"
```

---

## Task 10: Final verification — broader test sweep

**Files:** none.

- [ ] **Step 1: Run the broader synthesis + configs test universe**

```bash
uv run pytest tests/unit/core/synthesis/ tests/unit/core/configs/ \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_format_persona_injects_tools_for_assistant_only \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_build_role_context_includes_tools_for_assistant \
  -q
```

Expected: all tests pass. 2 deselected (pre-existing unrelated failures). No new failures introduced by this plan.

- [ ] **Step 2: Run the environments sweep (imports, smoke tests)**

```bash
uv run pytest tests/unit/ -q -k "environment or grounding" \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_format_persona_injects_tools_for_assistant_only \
  --deselect tests/unit/core/synthesis/test_conversation_synthesizer.py::test_build_role_context_includes_tools_for_assistant
```

Expected: all pass, including the newly added `grounding_*` tests.

- [ ] **Step 3: Summarize changes**

Run `git log --oneline` to sanity-check that each task produced exactly one commit:

```bash
git log --oneline -12
```

Expected: 9 commits from this plan (Tasks 1–9; Task 10 is verification-only).

---

## Self-review

**Spec coverage check:**

| Spec section | Covered by task(s) |
|--------------|---------------------|
| §2 Interfaces: `GroundingConfig` | Task 1 |
| §2 Interfaces: `BaseEnvironment.grounding` field, default `sample_grounding`, default `describe_grounding` | Tasks 2, 3 |
| §2 Interfaces: `DeterministicEnvironment.sample_grounding` override | Task 4 |
| §2 Interfaces: `GroundingFact` = `DeterministicToolOutput` | Implicit across Tasks 1–4 |
| §3 Data flow: `_attach_grounding_facts` | Task 6 |
| §3 Data flow: `_make_grounding_rng` | Task 5 |
| §3 Data flow: called from `synthesize()` | Task 7 |
| §3 Data flow: rendered in `_create_planner_prompt` | Task 7 |
| §4 Config surface: Python + YAML (YAML coercion in `BaseEnvironment.create`) | Task 3 |
| §4 Config surface: `GroundingConfig` exported from `oumi.environments` | Task 1 |
| §5 Error handling: `sample_size < 1` ValueError | Task 1 |
| §5 Error handling: truncation warning | Task 6 |
| §5 Error handling: `{grounding_facts}` misuse warning | Task 8 |
| §5 Error handling: no-op when no grounding | Tasks 6, 7, 9 |
| §6 Testing: all unit and integration test buckets | Tasks 1–9 |
| §7 Scope boundary | Covered by explicit omissions; no tasks for out-of-scope items |

No gaps identified.

**Placeholder scan:** no "TBD"/"TODO"/"appropriate"/"similar to Task N" references. All code blocks contain concrete implementations.

**Type consistency check:**
- `GroundingConfig(sample_size: int, seed: int | None)` — consistent across Tasks 1, 3, 5, 6.
- `sample_grounding(n: int, *, rng: random.Random) -> list[DeterministicToolOutput]` — consistent across Tasks 2, 3, 4, 6.
- `describe_grounding(facts: list[DeterministicToolOutput]) -> str` — consistent across Tasks 2, 3, 7.
- `describe_grounding_default(facts: list[DeterministicToolOutput]) -> str` (module-level helper) — consistent across Tasks 2, 3, 7.
- `_make_grounding_rng(seed: int | None, sample_index: int) -> random.Random` — consistent across Tasks 5, 6.
- `_attach_grounding_facts(samples: list[dict], multiturn_attribute: MultiTurnAttribute) -> None` — consistent across Tasks 6, 7.

No inconsistencies.
