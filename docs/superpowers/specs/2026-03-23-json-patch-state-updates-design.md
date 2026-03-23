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

**Example:** An `UPDATE appointments SET status = 'Completed' WHERE appointment_id = 2`
currently requires ~4,000 tokens of full state output. With JSON Patch:

```json
[
  {"op": "replace", "path": "/appointments/1/status", "value": "Completed"},
  {"op": "replace", "path": "/appointments/1/updated_at", "value": "2024-11-10T14:30:00"}
]
```

~50 tokens. Dramatically reduces failure rate.

**Note on JSON Pointer paths:** RFC 6902 uses JSON Pointers (RFC 6901) for
paths. For arrays, paths use 0-based indices (e.g., `/appointments/1` is the
second element). The LLM must determine the correct array index from the current
state shown in the prompt. The prompt includes the full current state so the LLM
can count positions.

## Design

### 1. New utility: `src/oumi/utils/json_patch.py`

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

### 2. Changes to `src/oumi/core/synthesis/environment.py`

#### `build_state_update_prompt`

The user message changes from:

> "Update the state to reflect this tool call. Output ONLY valid JSON
> conforming to the state schema."

To a prompt that:

1. Explains JSON Patch format with a **concrete example derived from the current
   state** (e.g., using a real key from `self._state` to show the path format)
2. Explicitly notes that array paths use 0-based indices
3. Asks for ONLY the JSON Patch array

Example prompt text:

> "Output a JSON Patch (RFC 6902) array describing ONLY the changes to the
> state. Each operation has: op (add/remove/replace), path (JSON Pointer using
> 0-based array indices), and value (for add/replace).
>
> Example for this state: `[{"op": "replace", "path": "/{first_key}/0/{first_field}", "value": "new"}]`
>
> Output ONLY the JSON array."

Where `{first_key}` and `{first_field}` are dynamically derived from
`self._state` to ground the example in the actual data structure.

The system message and the inclusion of current state, tool call details, and
tool result remain unchanged.

When `retry=True`, the appended message is updated to reference JSON Patch
format: "Ensure the output is a valid JSON Patch (RFC 6902) array."

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

### 3. Tests

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

- **LLM may produce wrong JSON Pointer paths** (e.g., wrong array index).
  Mitigated by: the prompt includes the full current state, and retries handle
  transient failures. Future enhancement: path auto-resolution for common
  patterns.
- **`jsonpatch` library dependency.** Low risk: pure Python, BSD license,
  well-maintained, no transitive dependencies.
