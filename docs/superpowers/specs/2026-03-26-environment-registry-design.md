# Environment Registry: Generate Once, Copy N

**Date:** 2026-03-26
**Status:** Proposed

## Problem

Environment initialization is the primary source of synthesis failures. The current approach generates schema and initial state independently for every sample via two sequential LLM calls that each produce large JSON blobs. For 50 samples this means ~100 LLM calls just for init, each of which can fail JSON parsing or schema validation. Retries compound the cost.

The root causes:

1. **Redundant generation** — every sample generates its own schema and state from scratch, even though all samples for a given environment config need the same structure.
2. **Monolithic state blobs** — the LLM must produce the entire state (7+ collections with cross-references) in a single call. Larger JSON = more truncation, malformed brackets, and schema mismatches.
3. **Schema generation is unnecessary** — tool definitions already describe the data shape (parameter schemas, output schemas, entity names). Asking the LLM to invent the schema introduces variance and failure.

## Solution

An `EnvironmentRegistry` that builds each environment **once** through a layered pipeline, then produces N independent copies for samples. The pipeline replaces monolithic blob generation with small, focused, per-collection LLM calls ordered by dependency.

**Before:** `2 * N_samples` large LLM calls for init (e.g., 100 for 50 samples).
**After:** `~5-7` small LLM calls total, regardless of sample count.

## Architecture

### EnvironmentRegistry

New class in `src/oumi/core/synthesis/environment_registry.py`. Owns the full build pipeline.

**Public API:**

```python
class EnvironmentRegistry:
    def build(
        self,
        config: ToolEnvironmentAttribute,
        tools: list[ToolAttribute],
        inference_engine: InferenceEngine,
        inference_config: InferenceConfig,
        scenario_context: str | None = None,
    ) -> None:
        """Build a fully populated environment through the layered pipeline.

        Stores the result internally, keyed by config.id.
        """

    def create_copies(self, env_id: str, n: int) -> list[GeneratedToolEnvironment]:
        """Return n independent deepcopies of the built environment."""
```

**Build pipeline phases:**

```
Phase 1: Schema Resolution
    config provides state_schema? → use it
    else → derive from tool definitions (deterministic)
    fallback → single LLM call (old path, only once)

Phase 2: Dependency Analysis
    parse schema collections → build dependency graph → topological sort into waves
    e.g. [[tenants, units], [leases], [payments, maintenance_requests]]

Phase 3: Per-Collection Population (waves)
    for each wave (sequential):
        for each collection in wave (parallel via infer() batch):
            LLM call: generate 3-8 records for this collection
            context includes: sub-schema + previously generated collections
            validate: JSON parsing + sub-schema + referential integrity
            retry up to 2 times on failure (only this collection)

Phase 4: Assembly + Validation
    combine collections into full state dict
    validate against complete schema
    store in registry
```

### Schema Derivation from Tools

When the config omits `state_schema`, the registry derives it deterministically from tool definitions.

**Algorithm:**

1. Collect all tools bound to this environment.
2. For each tool, extract entity hints from parameter and output schemas:
   - Fields ending in `_id` whose prefix matches across tools → entity collection names. `tenant_id` → strip `_id` → `tenant` → pluralize → `tenants`.
   - Output schema fields → record shape for that collection (e.g., `GetTenant` returns `{name, email, phone}` → `tenants` records have those fields).
   - Parameter enums → field constraints (e.g., `status: {enum: [occupied, vacant]}`).
3. Merge across tools — if `GetTenant` and `CreateLease` both reference `tenant_id`, they contribute to the same `tenants` collection.
4. Build JSON Schema: each collection is `"type": "object", "additionalProperties": {...}` keyed by string ID, with merged field definitions.

**Fallback:** If derivation produces 0 collections, fall back to a single LLM schema generation call. This is a safety net, not the normal path.

### Dependency Analysis and Collection Ordering

**Algorithm:**

1. Parse the schema's top-level keys as collection names.
2. For each collection, scan field definitions for foreign-key references: fields ending in `_id` whose prefix matches another collection name.
3. Build dependency graph: `leases -> [tenants, units]`, `payments -> [leases]`.
4. Topological sort into waves — groups of collections with no unresolved deps:
   - Wave 0: `tenants`, `units` (no deps, generate in parallel)
   - Wave 1: `leases` (depends on wave 0)
   - Wave 2: `payments`, `maintenance_requests` (depend on wave 0/1)

**Cycle handling:** Break cycles by picking the collection with fewer inbound references, populating it first, then patching references after the dependent collection is generated. Cycles are unlikely in practice.

### Per-Collection Population

One small LLM call per collection, producing ~10-15 lines of JSON.

**Prompt structure:**

- **System:** environment name, description, system_prompt from config.
- **User message:** collection sub-schema, record count (parsed from config description or default 3-8), previously generated collections as context (for waves 1+), ID format instructions.

Example prompt for a wave-1 collection:

```
Existing state:
tenants: {"T-001": {"name": "Alice Chen", ...}, "T-002": ...}
units: {"U-101": {"unit_number": "101", ...}, ...}

Generate 3-5 leases. Each lease must reference an existing
tenant_id and unit_id from above. Use string IDs like "L-001".
Output ONLY a JSON object keyed by ID. No markdown fences. Start with {.
```

**Validation per collection:**
- Parse JSON.
- Validate against collection sub-schema.
- Validate referential integrity: every `*_id` field must reference an existing record in the corresponding collection.
- On failure, retry up to 2 times with error context in the prompt.

**Assembly:** After all waves complete, combine into full state dict and validate against complete schema.

## Changes to Existing Code

### GeneratedToolEnvironment

**Remove** (no longer used by synthesizer):
- `build_schema_prompt()`
- `apply_schema()`
- `build_initial_state_prompt()`
- `apply_initial_state()`
- `_last_parsed_state` field

**Keep unchanged:**
- `__init__`, `state`, `set_state`, `set_schema` — used by registry to configure copies
- `build_result_prompt`, `apply_result` — runtime tool result generation
- `build_state_update_prompt`, `apply_state_update` — runtime state updates
- `summarize_state` — used during conversation turns

### ConversationSynthesizer

**Replace `_init_environments`** (~140 lines of batched schema + state generation with retry loops) with:

```python
registry = EnvironmentRegistry()
for env_id, bound_tools in env_tools.items():
    config = self._env_configs[env_id]
    if config.initial_state is not None and config.state_schema is not None:
        # Config provides everything — no LLM calls needed
        registry.register_static(config)
    else:
        registry.build(config, bound_tools, self._inference_engine,
                       self._inference_config, scenario_context)

sample_envs = []
for _ in samples:
    envs = {env_id: registry.create_copy(env_id) for env_id in env_tools}
    sample_envs.append(envs)
```

The rest of the synthesizer (turn generation, tool execution, state updates during conversation) is untouched.

### Test Changes

- Remove tests for `build_schema_prompt`, `apply_schema`, `build_initial_state_prompt`, `apply_initial_state`, `_last_parsed_state` from `test_environment.py`.
- New test file: `tests/unit/core/synthesis/test_environment_registry.py` covering:
  - Schema derivation from tool definitions
  - Dependency graph construction and topological sort
  - Per-collection prompt building
  - Per-collection validation (JSON, sub-schema, referential integrity)
  - Assembly and full-schema validation
  - `create_copies` returns independent deepcopies
  - Static registration (config provides both schema and state)
  - Fallback paths (0 collections detected, collection generation failure)

## Error Handling

**Per-collection retry:** If a collection fails JSON parsing, schema validation, or referential integrity, retry up to 2 times with error context. Only the failed collection is retried.

**Partial state fallback:** If a leaf collection exhausts retries (e.g., `payments`), the registry produces a valid state with the remaining collections. Logs which collections failed. The synthesizer can adjust tool availability if needed.

**Schema derivation fallback:** If tool-based derivation produces 0 collections, fall back to a single LLM schema generation call.

**Full failure:** If the entire pipeline fails (no collections populated), the environment is marked as failed and all samples using it are killed.

## New Files

- `src/oumi/core/synthesis/environment_registry.py` — registry class, schema derivation, dependency analysis, per-collection population prompts

## Files Modified

- `src/oumi/core/synthesis/environment.py` — remove init-phase methods
- `src/oumi/core/synthesis/conversation_synthesizer.py` — replace `_init_environments` with registry calls
- `tests/unit/core/synthesis/test_environment.py` — remove init-phase tests
