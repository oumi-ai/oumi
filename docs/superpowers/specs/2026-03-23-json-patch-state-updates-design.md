# JSON Patch State Updates for Environment Tool Synthesis

**Date:** 2026-03-23
**Status:** Approved
**Scope:** `oumi/utils/json_patch.py`, `oumi/core/synthesis/environment.py`

## Problem

`GeneratedToolEnvironment.apply_state_update()` requires the LLM to output the
**entire** environment state as a single JSON document after each write operation.
For database environments with 3-5 tables and 5-15 rows each, the state can
exceed 4,000 tokens. The LLM frequently fails to reproduce it accurately —
truncating output, drifting structure, or violating the auto-generated schema.

When the state update fails, `_apply_state_updates_batched` silently keeps the
previous state. Because the tool result is generated *before* the state update,
the result already reports success (e.g., `"rows_affected": 1`). Subsequent
read queries then return stale data, creating contradictions that cascade into:

- Duplicate tool calls (assistant retries hoping for different results)
- Hallucinated success claims (assistant trusts the "success" result over
  verification queries)
- Degraded training data quality

## Solution

Replace full-state replacement with **RFC 6902 JSON Patch**. Instead of asking
the LLM to output the entire state, ask it to output a small array of patch
operations describing only what changed. Apply the patch programmatically.

Two complementary changes:

1. **JSON Patch for state updates** — ask the LLM to output a small patch array
   instead of the full state.
2. **Dict-keyed state structure** — store collections as `{"id": {row}}` dicts
   instead of arrays, so JSON Pointer paths are semantic (key-based) rather than
   positional (index-based).

**Example:** An `UPDATE appointments SET status = 'Completed' WHERE appointment_id = 2`
currently requires ~4,000 tokens of full state output. With dict-keyed state +
JSON Patch:

```json
[
  {"op": "replace", "path": "/appointments/2/status", "value": "Completed"},
  {"op": "replace", "path": "/appointments/2/updated_at", "value": "2024-11-10T14:30:00"}
]
```

~50 tokens. The path `/appointments/2` refers to the dict key `"2"`, not array
index 2. No counting required — the LLM just uses the identifier it already
knows from the tool call arguments.

## Design

### 1. Dict-keyed state structure

Update `build_schema_prompt` to instruct the schema-generation LLM to use
**dictionaries keyed by primary identifiers** instead of arrays for collections.

Current (array-based):
```json
{
  "appointments": [
    {"appointment_id": 1, "status": "Completed"},
    {"appointment_id": 2, "status": "Scheduled"}
  ]
}
```

New (dict-keyed):
```json
{
  "appointments": {
    "1": {"appointment_id": 1, "status": "Completed"},
    "2": {"appointment_id": 2, "status": "Scheduled"}
  }
}
```

The instruction is added to the user message in `build_schema_prompt`:

> "IMPORTANT: For collections of records (e.g., tables, lists of entities),
> use dictionaries keyed by the record's primary identifier — NOT arrays.
> For example, use `{"users": {"1": {...}, "2": {...}}}` instead of
> `{"users": [{...}, {...}]}`."

The `build_initial_state_prompt` already follows the schema, so it will
naturally produce dict-keyed state. No changes needed there.

This ensures JSON Pointer paths in patches are always semantic
(`/appointments/2/status`) rather than positional (`/appointments/1/status`),
eliminating the array-index-counting failure mode.

### 2. New utility: `src/oumi/utils/json_patch.py`

A standalone utility module, not coupled to synthesis. Can be used anywhere in
the codebase.

**New dependency:** `jsonpatch` (pure Python, BSD license). Added to core
dependencies in `pyproject.toml`.

#### Public API

```python
class JsonPatchError(Exception):
    """Raised when a patch is malformed or cannot be applied."""

class JsonPatchValidationError(Exception):
    """Raised when the patched document fails schema validation."""

def apply_json_patch(
    document: dict[str, Any],
    patch: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply an RFC 6902 JSON Patch to a document.

    Args:
        document: The JSON document to patch (not mutated).
        patch: List of RFC 6902 patch operations.
        schema: Optional JSON Schema to validate the result against.

    Returns:
        A new dict with the patch applied.

    Raises:
        JsonPatchError: If the patch is malformed or application fails.
        JsonPatchValidationError: If the result fails schema validation.
    """

def parse_patch_response(text: str) -> list[dict[str, Any]] | None:
    """Extract a JSON Patch array from LLM-generated text.

    Handles markdown code fences and uses the existing extract_json utility.

    Returns:
        A list of patch operation dicts, or None if parsing fails.
    """
```

#### Behavior

- **Atomic:** If any operation in the patch fails, the original document is
  returned unchanged and `JsonPatchError` is raised.
- **Non-mutating:** The input `document` is never modified. A deep copy is
  patched and returned.
- **Schema validation:** When a schema is provided, the patched result is
  validated with `jsonschema.validate()`. Failure raises
  `JsonPatchValidationError` (not `JsonPatchError`) so callers can distinguish
  between patch application failures and schema mismatches.
- **Parsing:** `parse_patch_response` delegates to `extract_json(text, expected_type=list)`.
  Returns `None` on failure (caller handles retry).

### 4. Changes to `src/oumi/core/synthesis/environment.py`

#### `build_schema_prompt`

Add dict-keyed collection instruction to the user message (see section 1).

#### `build_state_update_prompt`

Changes from "output full state" to a **few-shot prompted JSON Patch request**.
The prompt becomes a 4-message conversation (system, user example, assistant
example, user actual) instead of the current 2-message format.

**System message:** Same environment system prompt + state schema (unchanged).

**Few-shot example (user):**
```
Current state:
{"users": {"1": {"name": "Alice", "role": "admin"}, "2": {"name": "Bob", "role": "viewer"}}}

Tool 'UpdateRole' was called with: {"user_id": "2", "new_role": "editor"}
Tool returned: {"status": "success", "rows_affected": 1}

Output a JSON Patch (RFC 6902) array describing ONLY the changes.
Use "replace" to update, "add" to insert, "remove" to delete.
Paths reference dict keys, not array indices (e.g., /users/2/role).
Output ONLY the JSON array.
```

**Few-shot example (assistant):**
```json
[{"op": "replace", "path": "/users/2/role", "value": "editor"}]
```

**Actual request (user):** Same format as the example — current state, tool
call, tool result, and the instruction block. A dynamic example path is derived
from `self._state` to reinforce the dict-key path format for the specific state
structure.

**Retry prompt:** When `retry=True`, appended message says: "Your previous
output was not a valid JSON Patch array. Output ONLY a JSON array of patch
operations."

#### `apply_state_update`

Changes from full-state replacement to patch application:

```python
def apply_state_update(self, response: Conversation) -> bool:
    text = self._extract_text(response)
    patch = parse_patch_response(text)
    if patch is None:
        logger.warning(f"Failed to parse JSON Patch: {text[:200]}")
        return False
    try:
        self._state = apply_json_patch(
            self._state, patch, schema=self._state_schema
        )
        return True
    except JsonPatchError as e:
        logger.warning(f"JSON Patch application failed: {e}")
        return False
    except JsonPatchValidationError as e:
        logger.warning(f"Patched state failed schema validation: {e}")
        return False
```

#### `_validate_state`

This method remains unchanged and is still used by `apply_initial_state`. It is
no longer called by `apply_state_update` (schema validation moves into
`apply_json_patch`), but must not be removed.

#### No other changes

- `build_result_prompt`, `apply_result`, `build_schema_prompt`,
  `apply_schema`, `build_initial_state_prompt`, `apply_initial_state`,
  `summarize_state` — all unchanged.
- `_apply_state_updates_batched` in `conversation_synthesizer.py` — unchanged.
  It calls `env.apply_state_update()` which returns `bool`. The retry logic
  works as-is.

### 5. Tests

#### `tests/unit/utils/test_json_patch.py` (new)

- Valid patch: replace, add, remove operations
- Atomic failure: one bad op in a multi-op patch leaves document unchanged
- Schema validation: valid patch but invalid result raises `JsonPatchValidationError`
- Parse from LLM text: markdown fences, bare JSON, malformed input
- Empty patch: no-op, returns copy of original

#### `tests/unit/core/synthesis/test_environment.py` (update)

- Existing `apply_state_update` tests updated to use patch format
- New test: patch that produces schema-invalid state returns False
- New test: malformed patch text returns False
- New test: `build_schema_prompt` includes dict-keyed collection instruction
- New test: `build_state_update_prompt` returns 4-message few-shot conversation

## What This Does NOT Change

- **Config format:** No changes to YAML configs.
- **`conversation_synthesizer.py`:** Calls the same `build_state_update_prompt` /
  `apply_state_update` interface.
- **`tool_executor.py`:** No changes.
- **Other environment methods:** Schema generation, initial state, result
  generation — all unchanged.
- **Retry logic:** `_apply_state_updates_batched` works identically since
  `apply_state_update` still returns `bool`.

## Risks

- **LLM may produce wrong JSON Pointer paths.** Mitigated by: dict-keyed state
  eliminates index-counting errors, few-shot examples demonstrate the path
  format, and the retry mechanism handles transient failures.
- **`jsonpatch` library dependency.** Low risk: pure Python, BSD license,
  well-maintained, no transitive dependencies.
- **Existing configs/datasets.** Regenerating datasets with this change will
  produce dict-keyed state. No migration needed — schemas and initial states
  are auto-generated per run.
