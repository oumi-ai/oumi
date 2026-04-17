# Grounding for Deterministic Environments — v1 Design

**Date:** 2026-04-17
**Status:** Approved, pending implementation plan
**Scope:** `DeterministicEnvironment` only. Synthetic env grounding is explicitly deferred.

---

## Problem

Conversation synthesis produces bad training data when user personas invent entity
IDs that don't exist in the environment. Example failure mode:

1. Planner emits abstract turn plans ("user asks about a book").
2. User persona enacts turn 1 by inventing `book_id=7777`.
3. Assistant calls `lookup({id: 7777})` against a `DeterministicEnvironment` that only
   has ids `1..50`.
4. `ToolLookupError` fires, the assistant apologizes, and the rest of the
   conversation derails into "that doesn't exist."

The resulting conversation teaches the model to produce error-laden exchanges
instead of successful tool-use flows. The root cause is that no role in the
synthesis pipeline is grounded in the environment's actual state.

## Grounding as a three-way contract

Different roles need different views of the environment, and mixing them up
produces bad data:

| Role | Needs to know | Does NOT need to know |
|------|---------------|------------------------|
| **Planner** | Which entities exist → so turn plans reference real IDs | How tools work internally |
| **User persona** | Nothing directly — inherits grounding via turn plans | Tool schemas, internal state |
| **Assistant persona** | Tool schemas only → must *discover* state via tool calls | The state itself — that's the whole point of tool use |

Critical invariant: **the assistant must stay un-grounded.** Injecting state into
its system prompt teaches it to answer from memory rather than call tools,
defeating the purpose of tool-use training data.

Corollary: only the planner is meta-aware. The user persona is a downstream actor
that enacts turn plans; if the planner is grounded, turn plans embed specific
entities and the user persona enacts them directly. Feeding the user a separate
entity list is redundant and risks drift from the planned entity.

## Design decisions (resolved during brainstorming)

1. **Storage rail** — approach C (hybrid): grounding facts live on
   `sample["grounding_facts"]` via the existing sample-dict rail (for
   inspection/debug parity with sampled attributes), but the *only consumer* is
   the planner prompt. No automatic injection into user/assistant personas.

2. **Diversity mechanism** — independent random sampling per conversation (mode A),
   with an optional `seed` knob on `GroundingConfig` for reproducibility. Matches
   the existing `SampledAttribute` sampling semantics.

3. **Rendering format** — bulleted markdown, flattened `input | output` dicts
   (`id=42, title="Dune"`). Same rendering for every consumer in v1; audience-
   aware formatting is a v2 concern.

4. **Config placement** — `grounding: GroundingConfig | None` on `BaseEnvironment`,
   not on `MultiTurnAttribute`. The env owns its pool, so it owns sampling
   defaults. `MultiTurnAttribute` is untouched.

5. **Opt-in by default** — omit `grounding` on the env → no grounding injected.
   Existing configs keep working byte-identically.

6. **No `ungrounded_fraction` in v1** — all grounded samples are grounded.
   Error-recovery training data via mixed grounding is deferred to v2.

## Architecture

```
DatasetPlanner.plan() produces sample dicts          (existing)
        │
        ▼
ConversationSynthesizer.synthesize(samples, multiturn_attr)
        │
        ├─ _validate_roles                           (existing)
        ├─ _validate_tool_configuration              (existing)
        │
        ├─ _attach_grounding_facts(samples, multiturn_attr)    (NEW)
        │     for each sample (by index i):
        │       facts = []
        │       for env in resolved_envs(multiturn_attr):
        │         if env.grounding is not None:
        │           rng = _make_rng(env.grounding.seed, i)
        │           facts.extend(env.sample_grounding(
        │               env.grounding.sample_size, rng=rng
        │           ))
        │       sample["grounding_facts"] = facts
        │
        ├─ _plan_samples(...)                        (existing; planner prompt
        │                                             now renders grounding block
        │                                             when facts are present)
        │
        └─ _synthesize_all_samples(...)              (unchanged)
```

Grounding is attached once, upstream of planning. The same facts are surfaced to
the planner once per conversation; they flow into turn plans via the planner's
output; they flow into the user persona indirectly via turn instructions; they
never reach the assistant persona.

## Interfaces

### `GroundingConfig`

New dataclass in `src/oumi/environments/grounding.py` (or colocated in
`base_tool.py` alongside `ToolSchema`).

```python
@dataclass
class GroundingConfig(BaseParams):
    """Per-environment grounding configuration.

    When set on an environment, the ConversationSynthesizer samples facts from
    that environment's state and injects them into the planner prompt so turn
    plans reference real entities.
    """

    sample_size: int = 3
    """Number of grounding facts sampled per conversation."""

    seed: int | None = None
    """If set, per-sample RNG is seeded from (seed + sample_index) for
    reproducibility. If None, uses global random state."""

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ValueError(
                f"{type(self).__name__}.sample_size must be >= 1, "
                f"got {self.sample_size}."
            )
```

### `BaseEnvironment` extensions

```python
@dataclass
class BaseEnvironment(BaseParams, ABC):
    id: str
    name: str
    description: str
    tools: list[Tool] = field(default_factory=list)
    grounding: GroundingConfig | None = None   # NEW

    def sample_grounding(
        self, n: int, *, rng: random.Random
    ) -> list[GroundingFact]:
        """Sample n grounding facts from this environment's state.

        Default: empty — subclasses override. Envs that don't implement
        grounding simply return [] and never contribute facts.
        """
        return []

    def describe_grounding(self, facts: list[GroundingFact]) -> str:
        """Render grounding facts as a bulleted markdown block.

        Default implementation flattens each fact's input/output dicts into
        a single `key=value, key=value` line. Suitable for any fact whose
        shape is dict-based (i.e., all `DeterministicToolOutput` facts).
        """
        # default impl in base class
        ...
```

**`GroundingFact` type** — v1 uses `DeterministicToolOutput` directly. It already
has the `input` and `output` dict fields we need. When synthetic-env grounding
lands, we may promote this to a Protocol or abstract dataclass, but v1 doesn't
need the abstraction.

### `DeterministicEnvironment` overrides

```python
class DeterministicEnvironment(BaseEnvironment):
    ...

    def sample_grounding(
        self, n: int, *, rng: random.Random
    ) -> list[DeterministicToolOutput]:
        pool = [
            output
            for tool in self.tools
            for output in tool.deterministic_outputs
        ]
        return rng.sample(pool, min(n, len(pool)))
```

No `describe_grounding` override needed — the base default handles
`DeterministicToolOutput` natively (flattening `input | output`).

### `ConversationSynthesizer` changes

New helper method:

```python
def _attach_grounding_facts(
    self,
    samples: list[dict],
    multiturn_attribute: MultiTurnAttribute,
) -> None:
    """Attach per-sample grounding facts drawn from envs with grounding config."""
    if self._environment_config is None:
        return

    # Reuse the same env-scoping semantics as _resolve_available_tools:
    # available_environments=None means "all envs in the config."
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

    for i, sample in enumerate(samples):
        facts: list[DeterministicToolOutput] = []
        for env in grounding_envs:
            rng = self._make_grounding_rng(env.grounding.seed, i)
            facts.extend(
                env.sample_grounding(n=env.grounding.sample_size, rng=rng)
            )
        sample["grounding_facts"] = facts

def _make_grounding_rng(
    self, seed: int | None, sample_index: int
) -> random.Random:
    if seed is None:
        return random.Random()   # unseeded, picks up global entropy
    return random.Random(seed + sample_index)
```

Called once from `synthesize()`, immediately after the existing
`_validate_tool_configuration`.

Planner prompt update in `_create_planner_prompt`: if the sample has a
non-empty `grounding_facts` list, format a block and insert it just before the
`Role context:` section of `base_prompt`:

```
Ground this plan in these specific entities:
- id=42, title="Dune", author="Frank Herbert"
- id=7, title="The Lord of the Rings", author="Tolkien"
- id=115, title="Pride and Prejudice", author="Austen"
Your turn plans must only reference these entities.
```

The exact wording lives in a constant so it's easy to tune later.

## Config surface

### Python

```python
env = DeterministicEnvironment(
    id="book_catalog",
    name="Books",
    description="Book lookup tools.",
    grounding=GroundingConfig(sample_size=3, seed=42),
    tools=[
        Tool(
            id="lookup_book",
            name="Lookup",
            description="Look up a book by id.",
            deterministic_outputs=[
                DeterministicToolOutput(
                    input={"id": "42"},
                    output={"title": "Dune", "author": "Frank Herbert"},
                ),
                # ... more entries
            ],
        ),
    ],
)
```

### YAML

```yaml
environments:
  - id: book_catalog
    type: deterministic
    name: Books
    description: Book lookup tools.
    grounding:
      sample_size: 3
      seed: 42
    tools:
      - id: lookup_book
        name: Lookup
        description: Look up a book by id.
        deterministic_outputs:
          - input: { id: "42" }
            output: { title: "Dune", author: "Frank Herbert" }
```

`GroundingConfig` is coerced from a dict the same way existing config objects are
(via `BaseEnvironment.create` and the dataclass `__post_init__`).

## Error handling

| Situation | Behavior |
|-----------|----------|
| `sample_size > pool_size` | Truncate to pool size. Emit `logger.warning(...)` on the first such truncation per `synthesize()` invocation per env. |
| `sample_size < 1` | Config-time `ValueError` in `GroundingConfig.__post_init__`. |
| Env in `available_environments` has `grounding=None` | That env contributes 0 facts silently. |
| All envs in scope lack grounding | No grounding block injected. Planner prompt identical to today. |
| `{grounding_facts}` placeholder used in user/assistant persona template | `logger.warning(...)` emitted at `synthesize()` time: "grounding is planner-only; referencing `{grounding_facts}` in user/assistant templates defeats its purpose." Not an error — escape hatch for advanced users. |
| Env with `grounding` configured but empty tool output pool | `sample_grounding` returns `[]`; planner receives no grounding block (equivalent to grounding being disabled for that conversation). |

## Testing strategy

### Unit

- `GroundingConfig.__post_init__` rejects `sample_size < 1`.
- `GroundingConfig` accepts integer and None for `seed`.
- `DeterministicEnvironment.sample_grounding`:
  - returns exactly `min(n, pool_size)` facts
  - no replacement within a single call
  - seeded `rng` gives deterministic output across runs
  - unseeded `rng` gives varying output (statistical check: 10 calls should produce
    ≥2 distinct sets)
  - truncates silently when `n > pool_size` and logs once
- `BaseEnvironment.describe_grounding` default:
  - single fact: `"- id=42, title=\"Dune\""`
  - multi-fact: bulleted list, preserves sample order
  - empty list: returns empty string
  - conflicting key between input and output: output value wins (flatten semantics)
- `_attach_grounding_facts`:
  - populates `sample["grounding_facts"]` for each sample
  - concatenates across multiple envs in scope
  - leaves sample dict untouched when no env has grounding
  - respects `multiturn_attribute.available_environments` scoping
  - per-sample RNG: `seed=42` gives different facts for sample index 0 vs 1;
    same seed + same index gives identical facts across two runs

### Integration

- End-to-end: `DeterministicEnvironment` with 10 pre-configured outputs,
  `sample_size=3`, run through `synthesize()` with a scripted inference engine
  mock. Assert:
  - planner prompt contains exactly 3 bulleted entity lines
  - the IDs in the bulleted lines are a subset of the env's configured inputs
- Regression: env with `grounding=None` produces byte-identical planner prompt
  to the pre-change baseline.

### Not tested in v1

- Real-LLM conversation generation (integration territory).
- Synthetic env grounding (out of scope).
- `ungrounded_fraction` behavior (deferred).
- Sampling strategies other than flat random.

## Scope boundary

### In v1

- `GroundingConfig` dataclass with `sample_size` + `seed`.
- `BaseEnvironment.sample_grounding` / `describe_grounding` with no-op defaults.
- `DeterministicEnvironment.sample_grounding` override.
- `BaseEnvironment.grounding` field, opt-in (default `None`).
- `ConversationSynthesizer._attach_grounding_facts` step.
- `_create_planner_prompt` renders grounding block when facts present.
- Soft-warning for misplaced `{grounding_facts}` placeholder.
- Public export of `GroundingConfig` from `oumi.environments`.
- ~150 LOC production + tests.

### Out of v1 (explicit)

- Synthetic environment grounding (`SyntheticEnvironment.sample_grounding`
  remains the default no-op).
- Sampling strategies other than flat random (`stratified`, `neighborhood`,
  `relational`).
- `ungrounded_fraction` / mixed grounded/ungrounded corpora.
- Auto-injection into user persona (design decision: planner is the only
  meta-aware role).
- Audience-aware `describe_grounding` rendering (planner vs user).
- Huge-state env variants (`SqliteDeterministicEnvironment`, etc.).
- Per-conversation-type grounding overrides on `MultiTurnAttribute`.

## Open questions, deferred

- **Coverage guarantees over a corpus.** v1 uses independent random sampling;
  every fact appears with expected frequency `~N_samples * k / pool_size`.
  Strict "every fact at least once" is a v2 sampling strategy, not v1.
- **Progressive grounding per turn.** Currently all grounding is attached
  upfront. For very long conversations with many references, per-turn
  refinement may be needed; deferred.
- **Formal `GroundingFact` type.** v1 uses `DeterministicToolOutput` concretely.
  Introducing an abstract `GroundingFact` Protocol is a v2 concern when
  synthetic envs need a different shape.
