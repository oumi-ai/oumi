# JSON Patch State Updates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace full-state-replacement with RFC 6902 JSON Patch for environment state updates, with dict-keyed state and few-shot prompting.

**Architecture:** New `oumi/utils/json_patch.py` utility wraps the `jsonpatch` library with error handling and LLM response parsing. `environment.py` changes `build_state_update_prompt` to a few-shot JSON Patch prompt and `apply_state_update` to apply patches instead of replacing state. `build_schema_prompt` adds a dict-keyed collection instruction.

**Tech Stack:** `jsonpatch` (already installed), `jsonschema` (already used)

**Spec:** `docs/superpowers/specs/2026-03-23-json-patch-state-updates-design.md`

---

### Task 1: Add `jsonpatch` to `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:40-87` (dependencies list)

- [ ] **Step 1: Add jsonpatch dependency**

Add `"jsonpatch>=1.33,<2.0",` to the `dependencies` list in `pyproject.toml`, alphabetically after `"jsonlines",` (line 51).

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: add jsonpatch dependency for RFC 6902 patch support"
```

---

### Task 2: Create `oumi/utils/json_patch.py` with tests (TDD)

**Files:**
- Create: `src/oumi/utils/json_patch.py`
- Create: `tests/unit/utils/test_json_patch.py`

- [ ] **Step 1: Write failing tests**

Create `tests/unit/utils/test_json_patch.py`:

```python
# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for oumi.utils.json_patch."""

import pytest

from oumi.utils.json_patch import (
    JsonPatchError,
    JsonPatchValidationError,
    apply_json_patch,
    parse_patch_response,
)


class TestApplyJsonPatch:
    def test_replace_op(self):
        doc = {"users": {"1": {"name": "Alice", "role": "admin"}}}
        patch = [{"op": "replace", "path": "/users/1/role", "value": "viewer"}]
        result = apply_json_patch(doc, patch)
        assert result["users"]["1"]["role"] == "viewer"

    def test_add_op(self):
        doc = {"users": {"1": {"name": "Alice"}}}
        patch = [{"op": "add", "path": "/users/2", "value": {"name": "Bob"}}]
        result = apply_json_patch(doc, patch)
        assert result["users"]["2"] == {"name": "Bob"}
        assert result["users"]["1"] == {"name": "Alice"}

    def test_remove_op(self):
        doc = {"users": {"1": {"name": "Alice"}, "2": {"name": "Bob"}}}
        patch = [{"op": "remove", "path": "/users/2"}]
        result = apply_json_patch(doc, patch)
        assert "2" not in result["users"]
        assert "1" in result["users"]

    def test_multi_op_patch(self):
        doc = {"items": {"a": {"count": 1}, "b": {"count": 2}}}
        patch = [
            {"op": "replace", "path": "/items/a/count", "value": 10},
            {"op": "replace", "path": "/items/b/count", "value": 20},
        ]
        result = apply_json_patch(doc, patch)
        assert result["items"]["a"]["count"] == 10
        assert result["items"]["b"]["count"] == 20

    def test_does_not_mutate_original(self):
        doc = {"x": {"1": {"v": "old"}}}
        patch = [{"op": "replace", "path": "/x/1/v", "value": "new"}]
        apply_json_patch(doc, patch)
        assert doc["x"]["1"]["v"] == "old"

    def test_empty_patch_returns_copy(self):
        doc = {"a": 1}
        result = apply_json_patch(doc, [])
        assert result == doc
        assert result is not doc

    def test_invalid_path_raises_error(self):
        doc = {"users": {"1": {"name": "Alice"}}}
        patch = [{"op": "replace", "path": "/nonexistent/field", "value": "x"}]
        with pytest.raises(JsonPatchError):
            apply_json_patch(doc, patch)

    def test_malformed_op_raises_error(self):
        doc = {"a": 1}
        patch = [{"bad_key": "replace"}]
        with pytest.raises(JsonPatchError):
            apply_json_patch(doc, patch)

    def test_atomic_failure(self):
        """If second op fails, first op should not be applied."""
        doc = {"a": {"1": {"v": "orig"}}, "b": {"1": {"v": "orig"}}}
        patch = [
            {"op": "replace", "path": "/a/1/v", "value": "changed"},
            {"op": "replace", "path": "/nonexistent/path", "value": "bad"},
        ]
        with pytest.raises(JsonPatchError):
            apply_json_patch(doc, patch)
        # Original must be unchanged
        assert doc["a"]["1"]["v"] == "orig"

    def test_schema_validation_pass(self):
        doc = {"count": 0}
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        patch = [{"op": "replace", "path": "/count", "value": 5}]
        result = apply_json_patch(doc, patch, schema=schema)
        assert result["count"] == 5

    def test_schema_validation_fail(self):
        doc = {"count": 0}
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        patch = [{"op": "replace", "path": "/count", "value": "not_an_int"}]
        with pytest.raises(JsonPatchValidationError):
            apply_json_patch(doc, patch, schema=schema)


class TestParsePatchResponse:
    def test_bare_json_array(self):
        text = '[{"op": "replace", "path": "/a", "value": 1}]'
        result = parse_patch_response(text)
        assert result == [{"op": "replace", "path": "/a", "value": 1}]

    def test_markdown_fenced(self):
        text = '```json\n[{"op": "add", "path": "/b", "value": 2}]\n```'
        result = parse_patch_response(text)
        assert result == [{"op": "add", "path": "/b", "value": 2}]

    def test_surrounding_prose(self):
        text = (
            'Here is the patch:\n[{"op": "remove", "path": "/c"}]\n'
            "This removes key c."
        )
        result = parse_patch_response(text)
        assert result == [{"op": "remove", "path": "/c"}]

    def test_empty_array(self):
        text = "[]"
        result = parse_patch_response(text)
        assert result == []

    def test_invalid_text(self):
        result = parse_patch_response("this is not json at all")
        assert result is None

    def test_dict_instead_of_list(self):
        text = '{"op": "replace", "path": "/a", "value": 1}'
        result = parse_patch_response(text)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/utils/test_json_patch.py -v`
Expected: ImportError — `oumi.utils.json_patch` does not exist yet.

- [ ] **Step 3: Write the implementation**

Create `src/oumi/utils/json_patch.py`:

```python
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

"""RFC 6902 JSON Patch utilities."""

import copy
from typing import Any

import jsonpatch
import jsonschema

from oumi.utils.str_utils import extract_json


class JsonPatchError(Exception):
    """Raised when a JSON Patch is malformed or cannot be applied."""


class JsonPatchValidationError(Exception):
    """Raised when the patched document fails JSON Schema validation."""


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
    doc_copy = copy.deepcopy(document)
    try:
        jp = jsonpatch.JsonPatch(patch)
        result = jp.apply(doc_copy)
    except (
        jsonpatch.JsonPatchException,
        jsonpatch.JsonPointerException,
        TypeError,
        KeyError,
        IndexError,
    ) as e:
        raise JsonPatchError(f"Failed to apply patch: {e}") from e

    if schema is not None:
        try:
            jsonschema.validate(instance=result, schema=schema)
        except jsonschema.ValidationError as e:
            raise JsonPatchValidationError(
                f"Patched document failed schema validation: {e.message}"
            ) from e

    return result


def parse_patch_response(text: str) -> list[dict[str, Any]] | None:
    """Extract a JSON Patch array from LLM-generated text.

    Handles markdown code fences and surrounding prose via extract_json.

    Args:
        text: Raw LLM response text.

    Returns:
        A list of patch operation dicts, or None if parsing fails.
    """
    result = extract_json(text, expected_type=list)
    if isinstance(result, list):
        return result
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/utils/test_json_patch.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oumi/utils/json_patch.py tests/unit/utils/test_json_patch.py
git commit -m "feat: add json_patch utility with apply, parse, and error types"
```

---

### Task 3: Update `build_schema_prompt` for dict-keyed collections

**Files:**
- Modify: `src/oumi/core/synthesis/environment.py:155-201` (`build_schema_prompt`)
- Modify: `tests/unit/core/synthesis/test_environment.py` (add test)

- [ ] **Step 1: Write failing test**

Add to `test_environment.py` in the `TestBatchedInitMethods` class:

```python
def test_build_schema_prompt_includes_dict_keyed_instruction(self):
    """Schema prompt instructs LLM to use dict-keyed collections."""
    config = _make_env_config(state_schema=None, initial_state=None)
    env = GeneratedToolEnvironment(config=config)
    tool = _make_env_tool()

    conv = env.build_schema_prompt([tool])

    user_text = conv.messages[1].content
    assert "dictionaries keyed by" in user_text.lower() or "dict" in user_text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/core/synthesis/test_environment.py::TestBatchedInitMethods::test_build_schema_prompt_includes_dict_keyed_instruction -v`
Expected: FAIL — current prompt doesn't include dict-keyed instruction.

- [ ] **Step 3: Add dict-keyed instruction to `build_schema_prompt`**

In `src/oumi/core/synthesis/environment.py`, modify the user message in `build_schema_prompt` (around line 187). Change the existing string:

```python
        user_parts = [
            "The following tools operate on this environment:\n\n"
            + "\n\n".join(tool_descriptions)
            + "\n\nDesign a JSON Schema for the state. "
            "Every field that any tool reads or writes must be represented. "
            "Output ONLY a valid JSON Schema object.",
        ]
```

To:

```python
        user_parts = [
            "The following tools operate on this environment:\n\n"
            + "\n\n".join(tool_descriptions)
            + "\n\nDesign a JSON Schema for the state. "
            "Every field that any tool reads or writes must be represented.\n\n"
            "IMPORTANT: For collections of records (e.g., tables, lists of "
            "entities), use dictionaries keyed by the record's primary "
            "identifier — NOT arrays. For example, use "
            '{"users": {"1": {...}, "2": {...}}} instead of '
            '{"users": [{...}, {...}]}. This ensures stable key-based '
            "lookups.\n\n"
            "Output ONLY a valid JSON Schema object.",
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/core/synthesis/test_environment.py::TestBatchedInitMethods::test_build_schema_prompt_includes_dict_keyed_instruction -v`
Expected: PASS.

- [ ] **Step 5: Run all existing environment tests**

Run: `pytest tests/unit/core/synthesis/test_environment.py -v`
Expected: All PASS (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/oumi/core/synthesis/environment.py tests/unit/core/synthesis/test_environment.py
git commit -m "feat: instruct schema LLM to use dict-keyed collections"
```

---

### Task 4: Update `build_state_update_prompt` with few-shot JSON Patch format

**Files:**
- Modify: `src/oumi/core/synthesis/environment.py:90-124` (`build_state_update_prompt`)
- Modify: `tests/unit/core/synthesis/test_environment.py` (add tests)

- [ ] **Step 1: Write failing tests**

Add a new test class to `test_environment.py`:

```python
class TestStateUpdatePrompt:
    def test_prompt_is_four_messages(self):
        """State update prompt has system, user example, assistant example, user actual."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )

        assert len(conv.messages) == 4
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[1].role == Role.USER
        assert conv.messages[2].role == Role.ASSISTANT
        assert conv.messages[3].role == Role.USER

    def test_few_shot_example_contains_patch_format(self):
        """The assistant example message contains a JSON Patch array."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )

        assistant_text = conv.messages[2].content
        assert '"op"' in assistant_text
        assert '"path"' in assistant_text
        assert '"replace"' in assistant_text

    def test_actual_request_contains_current_state(self):
        """The actual user message includes current state and tool details."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
        )

        user_text = conv.messages[3].content
        assert "a.txt" in user_text
        assert "WriteFile" in user_text
        assert "JSON Patch" in user_text

    def test_actual_request_contains_dynamic_example_path(self):
        """The actual user message includes an example path from current state."""
        config = _make_env_config(
            initial_state={"files": {"readme.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "readme.txt", "content": "world"},
            result='{"status": "success"}',
        )

        user_text = conv.messages[3].content
        # Should contain a path example using actual state keys
        assert "/files/" in user_text

    def test_retry_prompt_references_json_patch(self):
        """Retry=True appends a message referencing JSON Patch format."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)
        tool = _make_env_tool(read_only=False, id="write_file", name="WriteFile")

        conv = env.build_state_update_prompt(
            tool,
            arguments={"path": "a.txt", "content": "world"},
            result='{"status": "success"}',
            retry=True,
        )

        last_text = conv.messages[-1].content
        assert "JSON Patch" in last_text or "json patch" in last_text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/core/synthesis/test_environment.py::TestStateUpdatePrompt -v`
Expected: FAIL — current prompt returns 2 messages, not 4.

- [ ] **Step 3: Rewrite `build_state_update_prompt`**

Replace the method in `src/oumi/core/synthesis/environment.py`:

```python
    def build_state_update_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        result: str,
        retry: bool = False,
    ) -> Conversation:
        """Build a few-shot prompt for generating a JSON Patch state update."""
        system_parts = [
            self._config.system_prompt,
        ]
        if self._state_schema:
            system_parts.append(
                f"\nState schema:\n{json.dumps(self._state_schema, indent=2)}"
            )

        # Few-shot example
        example_user = (
            'Current state:\n{"users": {"1": {"name": "Alice", "role": "admin"}, '
            '"2": {"name": "Bob", "role": "viewer"}}}\n\n'
            "Tool 'UpdateRole' was called with:\n"
            '{"user_id": "2", "new_role": "editor"}\n\n'
            'Tool returned:\n{"status": "success", "rows_affected": 1}\n\n'
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. /users/2/role), "
            "and value (for add/replace). Output ONLY the JSON array."
        )
        example_assistant = (
            '[{"op": "replace", "path": "/users/2/role", "value": "editor"}]'
        )

        # Build a dynamic example path from current state
        example_path = self._build_example_path()

        # Actual request
        user_parts = [
            f"Current state:\n{json.dumps(self._state, indent=2)}\n\n"
            f"Tool '{tool.name}' was called with:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            f"Tool returned:\n{result}\n\n"
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. "
            f"{example_path}), "
            "and value (for add/replace). Output ONLY the JSON array.",
        ]
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous output was not a valid JSON Patch "
                "array. Output ONLY a JSON array of RFC 6902 patch operations."
            )

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content=example_user),
            Message(role=Role.ASSISTANT, content=example_assistant),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def _build_example_path(self) -> str:
        """Build a dynamic JSON Pointer example from current state keys.

        Walks the first key at each level up to depth 2 to produce something
        like "/users/1/name". Falls back to "/key/value" if state is empty.
        """
        parts: list[str] = []
        current: Any = self._state
        for _ in range(3):
            if isinstance(current, dict) and current:
                key = next(iter(current))
                parts.append(str(key))
                current = current[key]
            else:
                break
        if not parts:
            return "/key/value"
        return "/" + "/".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/core/synthesis/test_environment.py::TestStateUpdatePrompt -v`
Expected: All PASS.

- [ ] **Step 5: Run all environment tests**

Run: `pytest tests/unit/core/synthesis/test_environment.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/oumi/core/synthesis/environment.py tests/unit/core/synthesis/test_environment.py
git commit -m "feat: few-shot JSON Patch prompt for build_state_update_prompt"
```

---

### Task 5: Update `apply_state_update` to use JSON Patch

**Files:**
- Modify: `src/oumi/core/synthesis/environment.py:130-142` (`apply_state_update`)
- Modify: `tests/unit/core/synthesis/test_environment.py` (add tests)

- [ ] **Step 1: Write failing tests**

Add a new test class to `test_environment.py`:

```python
class TestApplyStateUpdate:
    def _make_response(self, text: str) -> Conversation:
        return Conversation(
            messages=[Message(role=Role.ASSISTANT, content=text)]
        )

    def test_valid_patch_updates_state(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "old"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "replace", "path": "/files/a.txt", "value": "new"}]'
        )
        result = env.apply_state_update(response)

        assert result is True
        assert env.state["files"]["a.txt"] == "new"

    def test_add_op_updates_state(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "add", "path": "/files/b.txt", "value": "world"}]'
        )
        result = env.apply_state_update(response)

        assert result is True
        assert env.state["files"]["b.txt"] == "world"
        assert env.state["files"]["a.txt"] == "hello"

    def test_invalid_patch_returns_false(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '[{"op": "replace", "path": "/nonexistent/path", "value": "x"}]'
        )
        result = env.apply_state_update(response)

        assert result is False
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_malformed_json_returns_false(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response("not json at all")
        result = env.apply_state_update(response)

        assert result is False
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_schema_validation_failure_returns_false(self):
        """Patch succeeds but result violates schema → returns False."""
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["files"],
            },
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        # Replace string value with integer — violates additionalProperties: string
        response = self._make_response(
            '[{"op": "replace", "path": "/files/a.txt", "value": 12345}]'
        )
        result = env.apply_state_update(response)

        assert result is False
        assert env.state["files"]["a.txt"] == "hello"

    def test_empty_patch_is_valid(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response("[]")
        result = env.apply_state_update(response)

        assert result is True
        assert env.state == {"files": {"a.txt": "hello"}}

    def test_markdown_fenced_patch(self):
        config = _make_env_config(
            initial_state={"files": {"a.txt": "old"}},
        )
        env = GeneratedToolEnvironment(config=config)

        response = self._make_response(
            '```json\n[{"op": "replace", "path": "/files/a.txt", "value": "new"}]\n```'
        )
        result = env.apply_state_update(response)

        assert result is True
        assert env.state["files"]["a.txt"] == "new"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/core/synthesis/test_environment.py::TestApplyStateUpdate -v`
Expected: FAIL — current `apply_state_update` expects full state, not patches.

- [ ] **Step 3: Rewrite `apply_state_update`**

In `src/oumi/core/synthesis/environment.py`, replace the `apply_state_update` method:

```python
    def apply_state_update(self, response: Conversation) -> bool:
        """Parse, validate, and apply a JSON Patch from an inference response.

        Returns True if the state was successfully updated, False otherwise.
        """
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

Also update the imports at the top of `environment.py`:

```python
from oumi.utils.json_patch import (
    JsonPatchError,
    JsonPatchValidationError,
    apply_json_patch,
    parse_patch_response,
)
```

And remove the now-unused `extract_json` import (it's still used by `apply_schema` and `apply_initial_state` — check before removing; if still used, keep it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/core/synthesis/test_environment.py::TestApplyStateUpdate -v`
Expected: All PASS.

- [ ] **Step 5: Run ALL environment tests to check for regressions**

Run: `pytest tests/unit/core/synthesis/test_environment.py -v`
Expected: All PASS. Note: `_validate_state` is still used by `apply_initial_state` (line 269), so no code should be removed.

- [ ] **Step 6: Commit**

```bash
git add src/oumi/core/synthesis/environment.py tests/unit/core/synthesis/test_environment.py
git commit -m "feat: apply_state_update now uses JSON Patch instead of full state replacement"
```

---

### Task 6: Final integration test — run full test suite

**Files:** None (verification only)

- [ ] **Step 1: Run all synthesis tests**

Run: `pytest tests/unit/core/synthesis/ -v`
Expected: All PASS.

- [ ] **Step 2: Run the json_patch utility tests**

Run: `pytest tests/unit/utils/test_json_patch.py -v`
Expected: All PASS.

- [ ] **Step 3: Verify no import errors**

Run: `python -c "from oumi.core.synthesis.environment import GeneratedToolEnvironment; print('OK')"`
Expected: `OK`

Run: `python -c "from oumi.utils.json_patch import apply_json_patch, parse_patch_response; print('OK')"`
Expected: `OK`
