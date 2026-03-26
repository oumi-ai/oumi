# Environment Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-sample environment generation with a registry that builds once via layered per-collection population, then copies N times.

**Architecture:** New `EnvironmentRegistry` class owns a 4-phase build pipeline: schema resolution (config-provided or derived from tools), dependency analysis (topological sort of collections), per-collection LLM population (small focused calls ordered by wave), and assembly with validation. The synthesizer creates the registry, calls `build()` once per environment, then `create_copies()` for all samples.

**Tech Stack:** Python, jsonschema, existing oumi inference engine (`infer()` batch API), existing `GeneratedToolEnvironment` for runtime.

---

### Task 1: Schema Derivation — Pure Functions

Build the deterministic schema derivation logic as standalone pure functions. No LLM calls, no classes — just `tools in, schema out`.

**Files:**
- Create: `src/oumi/core/synthesis/environment_registry.py`
- Test: `tests/unit/core/synthesis/test_environment_registry.py`

- [ ] **Step 1: Write failing tests for `_pluralize`**

```python
# tests/unit/core/synthesis/test_environment_registry.py

"""Tests for EnvironmentRegistry."""

from oumi.core.synthesis.environment_registry import _pluralize


class TestPluralize:
    def test_regular_noun(self):
        assert _pluralize("tenant") == "tenants"

    def test_already_plural(self):
        assert _pluralize("tenants") == "tenants"

    def test_noun_ending_in_s(self):
        assert _pluralize("status") == "statuses"

    def test_noun_ending_in_y(self):
        assert _pluralize("category") == "categories"

    def test_noun_ending_in_e(self):
        assert _pluralize("file") == "files"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestPluralize -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `_pluralize`**

```python
# src/oumi/core/synthesis/environment_registry.py

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

"""Environment registry: build once, copy N times."""


def _pluralize(word: str) -> str:
    """Naive English pluralization for entity names.

    Handles common suffixes. Not meant to be exhaustive — just good enough
    for collection names like tenant→tenants, category→categories.
    """
    if word.endswith("s"):
        if word.endswith("ss") or word.endswith("us"):
            return word + "es"
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestPluralize -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for `derive_schema_from_tools`**

```python
# Append to tests/unit/core/synthesis/test_environment_registry.py

from typing import Any

from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.environment_registry import derive_schema_from_tools


def _make_tool(**overrides: Any) -> ToolAttribute:
    defaults: dict[str, Any] = dict(
        id="get_tenant",
        name="GetTenant",
        description="Look up a tenant",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
        parameters={
            "type": "object",
            "properties": {"tenant_id": {"type": "string"}},
            "required": ["tenant_id"],
        },
    )
    defaults.update(overrides)
    return ToolAttribute(**defaults)


class TestDeriveSchemaFromTools:
    def test_single_entity_from_output_schema(self):
        """A tool returning tenant fields creates a tenants collection."""
        tool = _make_tool(
            output_schema={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            }
        )
        schema = derive_schema_from_tools([tool])

        assert schema["type"] == "object"
        assert "tenants" in schema["properties"]
        tenants_schema = schema["properties"]["tenants"]
        assert tenants_schema["type"] == "object"
        record_props = tenants_schema["additionalProperties"]["properties"]
        assert "name" in record_props
        assert "email" in record_props

    def test_multiple_entities_from_multiple_tools(self):
        """Tools referencing different _id fields create separate collections."""
        tenant_tool = _make_tool(
            id="get_tenant",
            name="GetTenant",
            parameters={
                "type": "object",
                "properties": {"tenant_id": {"type": "string"}},
                "required": ["tenant_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
            },
        )
        unit_tool = _make_tool(
            id="get_unit",
            name="GetUnit",
            parameters={
                "type": "object",
                "properties": {"unit_id": {"type": "string"}},
                "required": ["unit_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "unit_number": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        )

        schema = derive_schema_from_tools([tenant_tool, unit_tool])

        assert "tenants" in schema["properties"]
        assert "units" in schema["properties"]

    def test_merges_fields_across_tools(self):
        """Two tools referencing the same entity merge their fields."""
        get_tool = _make_tool(
            id="get_tenant",
            name="GetTenant",
            output_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
        )
        create_tool = _make_tool(
            id="create_tenant",
            name="CreateTenant",
            parameters={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                },
                "required": ["tenant_id", "name"],
            },
            output_schema=None,
        )

        schema = derive_schema_from_tools([get_tool, create_tool])

        record_props = schema["properties"]["tenants"]["additionalProperties"]["properties"]
        assert "name" in record_props
        assert "email" in record_props
        assert "phone" in record_props

    def test_preserves_enum_constraints(self):
        """Enum values from parameters are preserved in the schema."""
        tool = _make_tool(
            id="update_unit",
            name="UpdateUnit",
            parameters={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["occupied", "vacant"]},
                },
                "required": ["unit_id"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        )

        schema = derive_schema_from_tools([tool])

        record_props = schema["properties"]["units"]["additionalProperties"]["properties"]
        assert record_props["status"].get("enum") == ["occupied", "vacant"]

    def test_empty_tools_returns_empty_schema(self):
        """No tools produces an empty schema (triggers LLM fallback)."""
        schema = derive_schema_from_tools([])
        assert schema["type"] == "object"
        assert len(schema.get("properties", {})) == 0

    def test_id_fields_excluded_from_record_properties(self):
        """The entity's own _id field is not duplicated inside the record."""
        tool = _make_tool(
            output_schema={
                "type": "object",
                "properties": {
                    "tenant_id": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        )

        schema = derive_schema_from_tools([tool])

        record_props = schema["properties"]["tenants"]["additionalProperties"]["properties"]
        assert "tenant_id" not in record_props
        assert "name" in record_props
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestDeriveSchemaFromTools -v`
Expected: FAIL — `derive_schema_from_tools` not found.

- [ ] **Step 7: Implement `derive_schema_from_tools`**

```python
# Append to src/oumi/core/synthesis/environment_registry.py

import re
from typing import Any

from oumi.core.configs.params.tool_params import ToolAttribute

_ID_SUFFIX_PATTERN = re.compile(r"^(.+)_id$")


def derive_schema_from_tools(tools: list[ToolAttribute]) -> dict[str, Any]:
    """Derive a JSON Schema for environment state from tool definitions.

    Inspects parameter and output schemas to identify entity collections,
    their fields, and type constraints. Returns a JSON Schema with each
    collection as an additionalProperties object keyed by string ID.

    Args:
        tools: Tools bound to this environment.

    Returns:
        A JSON Schema dict. Empty properties dict if no entities detected.
    """
    # collection_name -> field_name -> field_schema
    collections: dict[str, dict[str, dict[str, Any]]] = {}

    for tool in tools:
        # Find which entity this tool primarily operates on
        entity_name = _detect_primary_entity(tool)
        if not entity_name:
            continue

        collection_name = _pluralize(entity_name)

        if collection_name not in collections:
            collections[collection_name] = {}

        # Extract fields from output schema
        if tool.output_schema and tool.output_schema.get("type") == "object":
            output_props = tool.output_schema.get("properties", {})
            for field_name, field_schema in output_props.items():
                if field_name == f"{entity_name}_id":
                    continue  # Don't include the entity's own ID
                if field_name not in collections[collection_name]:
                    collections[collection_name][field_name] = dict(field_schema)

        # Extract fields from parameters (may have richer type info / enums)
        if tool.parameters and tool.parameters.get("type") == "object":
            param_props = tool.parameters.get("properties", {})
            for field_name, field_schema in param_props.items():
                if field_name == f"{entity_name}_id":
                    continue
                # If this field references another entity, keep it
                # (it's a foreign key like lease.tenant_id)
                existing = collections[collection_name].get(field_name, {})
                merged = dict(existing)
                # Prefer enum from parameters (more specific)
                if "enum" in field_schema:
                    merged["enum"] = field_schema["enum"]
                if "type" in field_schema and "type" not in merged:
                    merged["type"] = field_schema["type"]
                if merged:
                    collections[collection_name][field_name] = merged

    # Build the JSON Schema
    properties: dict[str, Any] = {}
    for collection_name, fields in collections.items():
        properties[collection_name] = {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": fields,
            },
        }

    return {"type": "object", "properties": properties}


def _detect_primary_entity(tool: ToolAttribute) -> str | None:
    """Detect the primary entity a tool operates on.

    Looks at parameter names for patterns like `tenant_id` in required
    params first, then all params. Returns the entity name (singular)
    or None if no entity detected.
    """
    if not tool.parameters or tool.parameters.get("type") != "object":
        return None

    props = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))

    # Check required params first for primary entity
    for field_name in required:
        match = _ID_SUFFIX_PATTERN.match(field_name)
        if match:
            return match.group(1)

    # Fall back to any param with _id suffix
    for field_name in props:
        match = _ID_SUFFIX_PATTERN.match(field_name)
        if match:
            return match.group(1)

    return None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/oumi/core/synthesis/environment_registry.py tests/unit/core/synthesis/test_environment_registry.py
git commit -m "feat: add schema derivation from tool definitions"
```

---

### Task 2: Dependency Analysis — Topological Sort into Waves

Build the dependency graph from a schema and sort collections into generation waves.

**Files:**
- Modify: `src/oumi/core/synthesis/environment_registry.py`
- Test: `tests/unit/core/synthesis/test_environment_registry.py`

- [ ] **Step 1: Write failing tests for `build_dependency_graph` and `sort_into_waves`**

```python
# Append to tests/unit/core/synthesis/test_environment_registry.py

from oumi.core.synthesis.environment_registry import (
    build_dependency_graph,
    sort_into_waves,
)


class TestBuildDependencyGraph:
    def test_no_foreign_keys(self):
        """Collections with no _id references have empty dependency sets."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                    },
                },
                "units": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph == {"tenants": set(), "units": set()}

    def test_foreign_key_creates_dependency(self):
        """A lease with tenant_id depends on tenants."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "leases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "start_date": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["leases"] == {"tenants"}
        assert graph["tenants"] == set()

    def test_multiple_foreign_keys(self):
        """A lease referencing tenant_id and unit_id depends on both."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                },
                "units": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"number": {"type": "string"}},
                    },
                },
                "leases": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "unit_id": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["leases"] == {"tenants", "units"}

    def test_self_referencing_id_ignored(self):
        """A tenant_id inside the tenants collection is not a dependency."""
        schema = {
            "type": "object",
            "properties": {
                "tenants": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "tenant_id": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                },
            },
        }
        graph = build_dependency_graph(schema)
        assert graph["tenants"] == set()


class TestSortIntoWaves:
    def test_independent_collections_in_one_wave(self):
        graph = {"tenants": set(), "units": set()}
        waves = sort_into_waves(graph)
        assert len(waves) == 1
        assert set(waves[0]) == {"tenants", "units"}

    def test_dependent_collection_in_later_wave(self):
        graph = {
            "tenants": set(),
            "units": set(),
            "leases": {"tenants", "units"},
        }
        waves = sort_into_waves(graph)
        assert len(waves) == 2
        assert set(waves[0]) == {"tenants", "units"}
        assert waves[1] == ["leases"]

    def test_three_wave_chain(self):
        graph = {
            "tenants": set(),
            "units": set(),
            "leases": {"tenants", "units"},
            "payments": {"leases"},
        }
        waves = sort_into_waves(graph)
        assert len(waves) == 3
        assert set(waves[0]) == {"tenants", "units"}
        assert waves[1] == ["leases"]
        assert waves[2] == ["payments"]

    def test_empty_graph(self):
        waves = sort_into_waves({})
        assert waves == []

    def test_cycle_broken(self):
        """Cycles are broken — all collections still appear in output."""
        graph = {
            "tenants": {"leases"},
            "leases": {"tenants"},
        }
        waves = sort_into_waves(graph)
        all_collections = [c for wave in waves for c in wave]
        assert set(all_collections) == {"tenants", "leases"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestBuildDependencyGraph tests/unit/core/synthesis/test_environment_registry.py::TestSortIntoWaves -v`
Expected: FAIL — functions not found.

- [ ] **Step 3: Implement `build_dependency_graph` and `sort_into_waves`**

```python
# Append to src/oumi/core/synthesis/environment_registry.py

from oumi.utils.logging import logger


def build_dependency_graph(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Build a dependency graph from a state schema.

    For each collection, scans its record properties for foreign-key fields
    (fields ending in `_id` whose prefix matches another collection).

    Args:
        schema: The full state JSON Schema.

    Returns:
        Dict mapping collection name to set of collection names it depends on.
    """
    properties = schema.get("properties", {})
    collection_names = set(properties.keys())
    graph: dict[str, set[str]] = {name: set() for name in collection_names}

    for collection_name, collection_schema in properties.items():
        additional = collection_schema.get("additionalProperties", {})
        record_props = additional.get("properties", {})

        for field_name in record_props:
            match = _ID_SUFFIX_PATTERN.match(field_name)
            if not match:
                continue
            entity = match.group(1)
            target_collection = _pluralize(entity)
            # Don't add self-references as dependencies
            if target_collection == collection_name:
                continue
            if target_collection in collection_names:
                graph[collection_name].add(target_collection)

    return graph


def sort_into_waves(graph: dict[str, set[str]]) -> list[list[str]]:
    """Topological sort of collections into parallel waves.

    Each wave contains collections whose dependencies are all satisfied
    by previous waves. Collections within a wave can be generated in parallel.

    Cycles are broken by forcing the collection with fewer inbound
    references into an earlier wave.

    Args:
        graph: Dependency graph from build_dependency_graph.

    Returns:
        List of waves. Each wave is a sorted list of collection names.
    """
    if not graph:
        return []

    remaining = {name: set(deps) for name, deps in graph.items()}
    waves: list[list[str]] = []

    while remaining:
        # Find collections with no unresolved dependencies
        ready = sorted(
            name for name, deps in remaining.items() if not deps
        )

        if not ready:
            # Cycle detected — break it by picking the collection with
            # the fewest inbound references
            inbound_counts = {name: 0 for name in remaining}
            for deps in remaining.values():
                for dep in deps:
                    if dep in inbound_counts:
                        inbound_counts[dep] += 1
            ready = [min(inbound_counts, key=lambda n: inbound_counts[n])]
            logger.warning(
                f"Cycle detected in dependency graph. Breaking by "
                f"forcing '{ready[0]}' into current wave."
            )

        waves.append(ready)

        resolved = set(ready)
        for name in ready:
            del remaining[name]
        for deps in remaining.values():
            deps -= resolved

    return waves
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/oumi/core/synthesis/environment_registry.py tests/unit/core/synthesis/test_environment_registry.py
git commit -m "feat: add dependency graph and topological wave sort"
```

---

### Task 3: Per-Collection Population Prompts

Build the prompt construction and validation logic for generating one collection at a time.

**Files:**
- Modify: `src/oumi/core/synthesis/environment_registry.py`
- Test: `tests/unit/core/synthesis/test_environment_registry.py`

- [ ] **Step 1: Write failing tests for `build_collection_prompt`**

```python
# Append to tests/unit/core/synthesis/test_environment_registry.py

import json

from oumi.core.configs.params.tool_params import ToolEnvironmentAttribute
from oumi.core.synthesis.environment_registry import build_collection_prompt


def _make_env_config(**overrides: Any) -> ToolEnvironmentAttribute:
    defaults: dict[str, Any] = dict(
        id="db",
        name="PropertyDB",
        description="A property management database.",
        system_prompt="You are a property management database.",
    )
    defaults.update(overrides)
    return ToolEnvironmentAttribute(**defaults)


class TestBuildCollectionPrompt:
    def test_wave_0_prompt_has_no_existing_state(self):
        """First-wave collections get no existing state context."""
        config = _make_env_config()
        sub_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
        }
        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
        )

        assert len(conv.messages) == 2
        assert conv.messages[0].role == Role.SYSTEM
        user_text = conv.messages[1].content
        assert "tenants" in user_text
        assert "name" in user_text
        assert "Existing state" not in user_text

    def test_wave_1_prompt_includes_existing_state(self):
        """Later-wave collections see previously generated data."""
        config = _make_env_config()
        sub_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "start_date": {"type": "string"},
            },
        }
        existing = {"tenants": {"T-001": {"name": "Alice"}}}

        conv = build_collection_prompt(
            config=config,
            collection_name="leases",
            sub_schema=sub_schema,
            existing_state=existing,
        )

        user_text = conv.messages[1].content
        assert "T-001" in user_text
        assert "Alice" in user_text
        assert "tenant_id" in user_text

    def test_prompt_includes_id_format_hint(self):
        """Prompt tells the LLM what ID format to use."""
        config = _make_env_config()
        sub_schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
        )

        user_text = conv.messages[1].content
        assert "string IDs" in user_text.lower() or "keyed by ID" in user_text

    def test_prompt_requests_json_only(self):
        """Prompt instructs LLM to output only JSON."""
        config = _make_env_config()
        sub_schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
        )

        user_text = conv.messages[1].content
        assert "No markdown" in user_text or "no markdown" in user_text.lower()
        assert "Start with {" in user_text or "{" in user_text

    def test_scenario_context_included(self):
        """Scenario context appears in the prompt when provided."""
        config = _make_env_config()
        sub_schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        conv = build_collection_prompt(
            config=config,
            collection_name="tenants",
            sub_schema=sub_schema,
            existing_state={},
            scenario_context="Residential apartment complex in Seattle",
        )

        user_text = conv.messages[1].content
        assert "Seattle" in user_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestBuildCollectionPrompt -v`
Expected: FAIL — `build_collection_prompt` not found.

- [ ] **Step 3: Implement `build_collection_prompt`**

```python
# Append to src/oumi/core/synthesis/environment_registry.py

import json

from oumi.core.configs.params.tool_params import ToolEnvironmentAttribute
from oumi.core.types.conversation import Conversation, Message, Role

_DEFAULT_RECORD_COUNT = "3-8"


def build_collection_prompt(
    config: ToolEnvironmentAttribute,
    collection_name: str,
    sub_schema: dict[str, Any],
    existing_state: dict[str, Any],
    scenario_context: str | None = None,
    record_count: str = _DEFAULT_RECORD_COUNT,
    retry_error: str | None = None,
) -> Conversation:
    """Build a prompt for generating one collection's records.

    Args:
        config: The environment config (name, description, system_prompt).
        collection_name: Name of the collection to populate (e.g., "tenants").
        sub_schema: JSON Schema for a single record in this collection.
        existing_state: Previously generated collections (for foreign key context).
        scenario_context: Optional scenario description for realism.
        record_count: How many records to generate (e.g., "3-5").
        retry_error: If retrying, the error message from the previous attempt.

    Returns:
        A Conversation with system and user messages.
    """
    system_msg = (
        f"You are populating data for an environment.\n\n"
        f"Environment: {config.name}\n"
        f"Description: {config.description}\n\n"
        f"{config.system_prompt}"
    )

    user_parts: list[str] = []

    if scenario_context:
        user_parts.append(f"Scenario: {scenario_context}\n")

    if existing_state:
        state_lines = []
        for coll_name, coll_data in existing_state.items():
            state_lines.append(f"{coll_name}: {json.dumps(coll_data, indent=2)}")
        user_parts.append(
            f"Existing state:\n" + "\n\n".join(state_lines) + "\n"
        )

    schema_str = json.dumps(sub_schema, indent=2)
    user_parts.append(
        f"Generate {record_count} records for the '{collection_name}' collection.\n\n"
        f"Each record must match this schema:\n{schema_str}\n"
    )

    # Foreign key instructions
    fk_fields = [
        f for f in sub_schema.get("properties", {})
        if _ID_SUFFIX_PATTERN.match(f)
    ]
    if fk_fields and existing_state:
        refs = []
        for fk in fk_fields:
            match = _ID_SUFFIX_PATTERN.match(fk)
            if match:
                target = _pluralize(match.group(1))
                if target in existing_state:
                    refs.append(f"- {fk} must reference an existing ID from '{target}'")
        if refs:
            user_parts.append("Referential integrity:\n" + "\n".join(refs) + "\n")

    user_parts.append(
        "Output a JSON object keyed by string IDs (e.g., \"1\", \"2\" or "
        f"domain-appropriate IDs). No markdown fences. Start with {{."
    )

    if retry_error:
        user_parts.append(
            f"\nIMPORTANT: Your previous output failed validation: {retry_error}\n"
            "Fix the issue and output only valid JSON."
        )

    messages = [
        Message(role=Role.SYSTEM, content=system_msg),
        Message(role=Role.USER, content="\n".join(user_parts)),
    ]
    return Conversation(messages=messages)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Write failing tests for `validate_collection`**

```python
# Append to tests/unit/core/synthesis/test_environment_registry.py

from oumi.core.synthesis.environment_registry import validate_collection


class TestValidateCollection:
    def test_valid_collection(self):
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"1": {"name": "Alice"}, "2": {"name": "Bob"}}

        ok, error = validate_collection("tenants", data, sub_schema, {})

        assert ok is True
        assert error is None

    def test_invalid_json_type(self):
        """Non-dict data fails."""
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        ok, error = validate_collection("tenants", [1, 2], sub_schema, {})

        assert ok is False
        assert "dict" in error.lower() or "object" in error.lower()

    def test_record_schema_violation(self):
        """A record violating the sub-schema fails."""
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        data = {"1": {"name": 12345}}

        ok, error = validate_collection("tenants", data, sub_schema, {})

        assert ok is False

    def test_referential_integrity_pass(self):
        """Foreign keys pointing to existing records pass."""
        sub_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "start_date": {"type": "string"},
            },
        }
        data = {"L-001": {"tenant_id": "T-001", "start_date": "2024-01-01"}}
        existing = {"tenants": {"T-001": {"name": "Alice"}}}

        ok, error = validate_collection("leases", data, sub_schema, existing)

        assert ok is True

    def test_referential_integrity_fail(self):
        """Foreign keys pointing to nonexistent records fail."""
        sub_schema = {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "start_date": {"type": "string"},
            },
        }
        data = {"L-001": {"tenant_id": "T-999", "start_date": "2024-01-01"}}
        existing = {"tenants": {"T-001": {"name": "Alice"}}}

        ok, error = validate_collection("leases", data, sub_schema, existing)

        assert ok is False
        assert "T-999" in error

    def test_empty_collection_valid(self):
        """An empty dict is valid (generates 0 records)."""
        sub_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

        ok, error = validate_collection("tenants", {}, sub_schema, {})

        assert ok is True
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestValidateCollection -v`
Expected: FAIL — `validate_collection` not found.

- [ ] **Step 7: Implement `validate_collection`**

```python
# Append to src/oumi/core/synthesis/environment_registry.py

import jsonschema


def validate_collection(
    collection_name: str,
    data: Any,
    sub_schema: dict[str, Any],
    existing_state: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate a generated collection's data.

    Checks:
    1. Data is a dict (keyed by ID).
    2. Each record validates against the sub-schema.
    3. Foreign key fields reference existing records in other collections.

    Args:
        collection_name: Name of this collection.
        data: The generated data (should be dict[str, dict]).
        sub_schema: JSON Schema for a single record.
        existing_state: Previously generated collections for FK checks.

    Returns:
        (True, None) if valid, (False, error_message) if not.
    """
    if not isinstance(data, dict):
        return False, f"Expected a dict keyed by ID, got {type(data).__name__}."

    # Validate each record against the sub-schema
    for record_id, record in data.items():
        try:
            jsonschema.validate(instance=record, schema=sub_schema)
        except jsonschema.ValidationError as e:
            return False, f"Record '{record_id}' failed schema validation: {e.message}"

    # Check referential integrity
    for record_id, record in data.items():
        if not isinstance(record, dict):
            continue
        for field_name, value in record.items():
            match = _ID_SUFFIX_PATTERN.match(field_name)
            if not match or not isinstance(value, str):
                continue
            entity = match.group(1)
            target_collection = _pluralize(entity)
            if target_collection == collection_name:
                continue  # Self-references are fine
            if target_collection not in existing_state:
                continue  # No data to check against
            if value not in existing_state[target_collection]:
                return (
                    False,
                    f"Record '{record_id}' references {field_name}='{value}' "
                    f"but '{value}' does not exist in '{target_collection}'.",
                )

    return True, None
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add src/oumi/core/synthesis/environment_registry.py tests/unit/core/synthesis/test_environment_registry.py
git commit -m "feat: add per-collection prompt building and validation"
```

---

### Task 4: EnvironmentRegistry Class — Build Pipeline and Copy

Wire the phases together into the `EnvironmentRegistry` class with `build()`, `register_static()`, and `create_copies()`.

**Files:**
- Modify: `src/oumi/core/synthesis/environment_registry.py`
- Test: `tests/unit/core/synthesis/test_environment_registry.py`

- [ ] **Step 1: Write failing tests for the registry**

```python
# Append to tests/unit/core/synthesis/test_environment_registry.py

from unittest.mock import MagicMock

from oumi.core.synthesis.environment import GeneratedToolEnvironment
from oumi.core.synthesis.environment_registry import EnvironmentRegistry


class TestEnvironmentRegistryStatic:
    def test_register_static_and_create_copies(self):
        """Static registration stores env and copies are independent."""
        config = _make_env_config(
            state_schema={"type": "object", "properties": {"tenants": {"type": "object"}}},
            initial_state={"tenants": {"1": {"name": "Alice"}}},
        )
        registry = EnvironmentRegistry()
        registry.register_static(config)

        copies = registry.create_copies("db", 3)

        assert len(copies) == 3
        for env in copies:
            assert isinstance(env, GeneratedToolEnvironment)
            assert env.state == {"tenants": {"1": {"name": "Alice"}}}

        # Copies are independent
        copies[0].state["tenants"]["1"]["name"] = "Mutated"
        assert copies[1].state["tenants"]["1"]["name"] == "Alice"

    def test_create_copies_unknown_env_raises(self):
        """Requesting copies of an unregistered env raises KeyError."""
        registry = EnvironmentRegistry()
        try:
            registry.create_copies("nonexistent", 1)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass


class TestEnvironmentRegistryBuild:
    def _make_mock_engine(self, responses: list[str]):
        """Create a mock inference engine returning canned responses."""
        engine = MagicMock()
        call_count = 0

        def mock_infer(input, inference_config=None):
            nonlocal call_count
            results = []
            for _ in input:
                text = responses[call_count % len(responses)]
                call_count += 1
                results.append(
                    Conversation(messages=[Message(role=Role.ASSISTANT, content=text)])
                )
            return results

        engine.infer = mock_infer
        return engine

    def test_build_with_config_provided_schema(self):
        """When config has state_schema, no schema derivation happens."""
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            },
        )
        tenant_data = json.dumps({"T-001": {"name": "Alice"}, "T-002": {"name": "Bob"}})
        engine = self._make_mock_engine([tenant_data])
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        tools = [_make_tool()]
        registry.build(config, tools, engine, inference_config)

        copies = registry.create_copies("db", 2)
        assert len(copies) == 2
        assert "T-001" in copies[0].state["tenants"]

    def test_build_derives_schema_when_not_provided(self):
        """When config has no state_schema, schema is derived from tools."""
        config = _make_env_config(state_schema=None, initial_state=None)
        tenant_data = json.dumps({"T-001": {"name": "Alice"}})
        engine = self._make_mock_engine([tenant_data])
        inference_config = MagicMock()

        tools = [
            _make_tool(
                output_schema={
                    "type": "object",
                    "properties": {
                        "tenant_id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
            )
        ]

        registry = EnvironmentRegistry()
        registry.build(config, tools, engine, inference_config)

        copies = registry.create_copies("db", 1)
        assert copies[0].state.get("tenants") is not None

    def test_build_multi_wave(self):
        """Collections with dependencies are generated in wave order."""
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                    "leases": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "tenant_id": {"type": "string"},
                                "start_date": {"type": "string"},
                            },
                        },
                    },
                },
            },
        )
        tenant_data = json.dumps({"T-001": {"name": "Alice"}})
        lease_data = json.dumps({"L-001": {"tenant_id": "T-001", "start_date": "2024-01-01"}})
        engine = self._make_mock_engine([tenant_data, lease_data])
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        registry.build(config, [_make_tool()], engine, inference_config)

        copies = registry.create_copies("db", 1)
        state = copies[0].state
        assert "T-001" in state["tenants"]
        assert "L-001" in state["leases"]
        assert state["leases"]["L-001"]["tenant_id"] == "T-001"

    def test_build_partial_failure_keeps_successful_collections(self):
        """If a collection fails, others are still populated."""
        config = _make_env_config(
            state_schema={
                "type": "object",
                "properties": {
                    "tenants": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                    "units": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {"number": {"type": "string"}},
                        },
                    },
                },
            },
        )
        tenant_data = json.dumps({"T-001": {"name": "Alice"}})
        # Units always returns invalid JSON
        engine = self._make_mock_engine([tenant_data, "not valid json at all"])
        inference_config = MagicMock()

        registry = EnvironmentRegistry()
        registry.build(config, [_make_tool()], engine, inference_config)

        copies = registry.create_copies("db", 1)
        state = copies[0].state
        assert "T-001" in state.get("tenants", {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py::TestEnvironmentRegistryStatic tests/unit/core/synthesis/test_environment_registry.py::TestEnvironmentRegistryBuild -v`
Expected: FAIL — `EnvironmentRegistry` not found.

- [ ] **Step 3: Implement `EnvironmentRegistry`**

```python
# Append to src/oumi/core/synthesis/environment_registry.py

import copy

from oumi.core.configs.params.tool_params import ToolEnvironmentAttribute
from oumi.core.synthesis.environment import GeneratedToolEnvironment
from oumi.utils.str_utils import extract_json

_MAX_COLLECTION_RETRIES = 2


class EnvironmentRegistry:
    """Builds environments once through a layered pipeline, then copies N times.

    Phases:
    1. Schema resolution: config-provided or derived from tools.
    2. Dependency analysis: topological sort into generation waves.
    3. Per-collection population: small LLM calls per collection, wave-ordered.
    4. Assembly + validation: combine and validate full state.
    """

    def __init__(self):
        self._built: dict[str, GeneratedToolEnvironment] = {}

    def register_static(self, config: ToolEnvironmentAttribute) -> None:
        """Register an environment with config-provided schema and state.

        No LLM calls needed — the config has everything.
        """
        env = GeneratedToolEnvironment(config=config)
        self._built[config.id] = env

    def build(
        self,
        config: ToolEnvironmentAttribute,
        tools: list[ToolAttribute],
        inference_engine: Any,
        inference_config: Any,
        scenario_context: str | None = None,
    ) -> None:
        """Build a fully populated environment through the layered pipeline.

        Args:
            config: Environment configuration.
            tools: Tools bound to this environment.
            inference_engine: Engine for LLM inference calls.
            inference_config: Config for inference calls.
            scenario_context: Optional scenario for realistic data generation.
        """
        # Phase 1: Schema resolution
        schema = self._resolve_schema(config, tools)

        # Phase 2: Dependency analysis
        graph = build_dependency_graph(schema)
        waves = sort_into_waves(graph)

        # Phase 3: Per-collection population
        state: dict[str, Any] = {}
        properties = schema.get("properties", {})

        for wave in waves:
            prompts = []
            wave_collections = []

            for collection_name in wave:
                collection_schema = properties.get(collection_name, {})
                sub_schema = collection_schema.get("additionalProperties", {})
                prompt = build_collection_prompt(
                    config=config,
                    collection_name=collection_name,
                    sub_schema=sub_schema,
                    existing_state=state,
                    scenario_context=scenario_context,
                )
                prompts.append(prompt)
                wave_collections.append((collection_name, sub_schema))

            if not prompts:
                continue

            responses = inference_engine.infer(
                prompts, inference_config=inference_config
            )

            # Process responses and retry failures
            for idx, ((collection_name, sub_schema), response) in enumerate(
                zip(wave_collections, responses)
            ):
                success = self._process_collection_response(
                    collection_name=collection_name,
                    sub_schema=sub_schema,
                    response=response,
                    state=state,
                    config=config,
                    existing_state=state,
                    scenario_context=scenario_context,
                    inference_engine=inference_engine,
                    inference_config=inference_config,
                )
                if not success:
                    logger.warning(
                        f"Collection '{collection_name}' failed after "
                        f"{_MAX_COLLECTION_RETRIES} retries. Skipping."
                    )

        # Phase 4: Assembly — state is already assembled, store it
        env = GeneratedToolEnvironment(config=config)
        env.set_schema(schema)
        env.set_state(state, validate=False)
        self._built[config.id] = env

    def create_copies(
        self, env_id: str, n: int
    ) -> list[GeneratedToolEnvironment]:
        """Return n independent deepcopies of the built environment.

        Args:
            env_id: The environment config ID.
            n: Number of copies to produce.

        Returns:
            List of independent GeneratedToolEnvironment instances.

        Raises:
            KeyError: If env_id has not been built or registered.
        """
        if env_id not in self._built:
            raise KeyError(
                f"Environment '{env_id}' not found. "
                f"Call build() or register_static() first."
            )
        source = self._built[env_id]
        return [copy.deepcopy(source) for _ in range(n)]

    def _resolve_schema(
        self,
        config: ToolEnvironmentAttribute,
        tools: list[ToolAttribute],
    ) -> dict[str, Any]:
        """Resolve the state schema: config-provided or tool-derived."""
        if config.state_schema:
            return copy.deepcopy(config.state_schema)

        schema = derive_schema_from_tools(tools)
        if not schema.get("properties"):
            logger.warning(
                f"Schema derivation for '{config.id}' produced no collections. "
                f"Using permissive schema."
            )
            return {"type": "object"}
        return schema

    def _process_collection_response(
        self,
        collection_name: str,
        sub_schema: dict[str, Any],
        response: Any,
        state: dict[str, Any],
        config: ToolEnvironmentAttribute,
        existing_state: dict[str, Any],
        scenario_context: str | None,
        inference_engine: Any,
        inference_config: Any,
    ) -> bool:
        """Process a collection generation response with retry on failure.

        Returns True if the collection was successfully populated.
        """
        text = _extract_response_text(response)
        parsed = extract_json(text, expected_type=dict)

        if isinstance(parsed, dict):
            ok, error = validate_collection(
                collection_name, parsed, sub_schema, existing_state
            )
            if ok:
                state[collection_name] = parsed
                return True
            retry_error = error
        else:
            retry_error = f"Could not parse JSON from response: {text[:200]}"

        # Retry loop
        for _ in range(_MAX_COLLECTION_RETRIES):
            retry_prompt = build_collection_prompt(
                config=config,
                collection_name=collection_name,
                sub_schema=sub_schema,
                existing_state=existing_state,
                scenario_context=scenario_context,
                retry_error=retry_error,
            )
            retry_responses = inference_engine.infer(
                [retry_prompt], inference_config=inference_config
            )
            text = _extract_response_text(retry_responses[0])
            parsed = extract_json(text, expected_type=dict)

            if isinstance(parsed, dict):
                ok, error = validate_collection(
                    collection_name, parsed, sub_schema, existing_state
                )
                if ok:
                    state[collection_name] = parsed
                    return True
                retry_error = error
            else:
                retry_error = f"Could not parse JSON from response: {text[:200]}"

        return False


def _extract_response_text(response: Conversation) -> str:
    """Extract text from the last message of an inference response."""
    if not response.messages:
        return ""
    content = response.messages[-1].content
    return content if isinstance(content, str) else ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/core/synthesis/test_environment_registry.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/oumi/core/synthesis/environment_registry.py tests/unit/core/synthesis/test_environment_registry.py
git commit -m "feat: add EnvironmentRegistry with build pipeline and copy"
```

---

### Task 5: Clean Up GeneratedToolEnvironment — Remove Init Methods

Remove the init-phase methods that the registry replaces, and their tests.

**Files:**
- Modify: `src/oumi/core/synthesis/environment.py`
- Modify: `tests/unit/core/synthesis/test_environment.py`

- [ ] **Step 1: Remove init-phase methods from `environment.py`**

Remove these methods and fields from `GeneratedToolEnvironment`:
- `_last_parsed_state` field (line 65)
- `build_schema_prompt()` (lines 267-344)
- `apply_schema()` (lines 346-362)
- `build_initial_state_prompt()` (lines 364-427)
- `apply_initial_state()` (lines 429-445)

Also remove the now-unused import `extract_json` from the imports section (line 32).

The class should retain:
- `__init__` (but remove `self._last_parsed_state = None`)
- `state`, `set_state`, `set_schema`
- `build_result_prompt`, `apply_result`
- `build_state_update_prompt`, `apply_state_update`
- `_validate_state`, `_build_example_path`, `_extract_text`
- `summarize_state`, `_summarize_value`

- [ ] **Step 2: Remove init-phase tests from `test_environment.py`**

Remove these test classes:
- `TestBatchedInitMethods` (lines 69-273) — tests `build_schema_prompt`, `apply_schema`, `build_initial_state_prompt`, `apply_initial_state`, `summarize_state`
- `TestApplyInitialStateFallback` (lines 540-567) — tests `_last_parsed_state`

**Keep** `TestSummarizeState` tests — extract the `test_summarize_state_*` methods from `TestBatchedInitMethods` into a new class:

```python
class TestSummarizeState:
    def test_summarize_state_basic(self):
        """Summary contains key names; no LLM call is made."""
        config = _make_env_config(
            initial_state={"files": {"a.txt": "hello", "b.txt": "world"}}
        )
        env = GeneratedToolEnvironment(config=config)

        summary = env.summarize_state()

        assert "files" in summary
        assert "a.txt" in summary
        assert "b.txt" in summary

    def test_summarize_state_nested_arrays(self):
        """Summary reports array lengths."""
        config = _make_env_config(
            state_schema=None,
            initial_state={"items": [{"name": "x"}, {"name": "y"}, {"name": "z"}]},
        )
        env = GeneratedToolEnvironment(config=config)

        summary = env.summarize_state()

        assert "items" in summary
        assert "3" in summary

    def test_summarize_state_empty(self):
        """Handles empty state gracefully."""
        config = _make_env_config(initial_state=None, state_schema=None)
        env = GeneratedToolEnvironment(config=config)
        env._state = {}

        summary = env.summarize_state()

        assert isinstance(summary, str)
```

**Keep** all other test classes unchanged:
- `TestStateUpdatePrompt`
- `TestApplyStateUpdate`
- `TestPublicSetters`
- `TestStateUpdatePromptThreeShot`

- [ ] **Step 3: Run all environment tests to verify nothing is broken**

Run: `python -m pytest tests/unit/core/synthesis/test_environment.py tests/unit/core/synthesis/test_environment_registry.py -v`
Expected: ALL PASS. The removed test classes should not appear, the kept tests should pass.

- [ ] **Step 4: Commit**

```bash
git add src/oumi/core/synthesis/environment.py tests/unit/core/synthesis/test_environment.py
git commit -m "refactor: remove init-phase methods from GeneratedToolEnvironment"
```

---

### Task 6: Integrate Registry into ConversationSynthesizer

Replace `_init_sample_environments` with registry-based initialization.

**Files:**
- Modify: `src/oumi/core/synthesis/conversation_synthesizer.py`

- [ ] **Step 1: Update imports in `conversation_synthesizer.py`**

Add the registry import and remove the now-unnecessary init-phase imports:

```python
# Replace this import block (lines 32-36):
from oumi.core.synthesis.environment import (
    _MAX_RESULT_RETRIES,
    _MAX_STATE_UPDATE_RETRIES,
    GeneratedToolEnvironment,
)

# With:
from oumi.core.synthesis.environment import (
    _MAX_RESULT_RETRIES,
    _MAX_STATE_UPDATE_RETRIES,
    GeneratedToolEnvironment,
)
from oumi.core.synthesis.environment_registry import EnvironmentRegistry
```

Note: `_MAX_STATE_UPDATE_RETRIES` may still be used elsewhere in the file for runtime state update retries. Check and keep only what's still referenced.

- [ ] **Step 2: Replace `_init_sample_environments` method**

Replace the entire `_init_sample_environments` method (lines 117-302) with:

```python
def _init_sample_environments(
    self,
    samples: list[dict],
    multiturn_attribute: MultiTurnAttribute,
) -> list[dict[str, GeneratedToolEnvironment] | None]:
    """Create and initialize per-sample environments via the registry.

    Builds each environment once, then copies for all samples.

    Args:
        samples: List of sample dicts, each containing resolved attribute values.
        multiturn_attribute: The multi-turn attribute defining which tools are used.

    Returns:
        A list aligned to samples. Each entry is a dict mapping env_id to a
        fresh GeneratedToolEnvironment, or None if no env-bound tools exist.
    """
    tools = self._get_tools_for_multiturn(multiturn_attribute)
    env_tools: dict[str, list[ToolAttribute]] = {}
    for tool in tools:
        if tool.environment:
            env_tools.setdefault(tool.environment, []).append(tool)

    if not env_tools:
        return [None] * len(samples)

    # Build scenario context from role instructions
    scenario_parts = []
    for role, instruction in multiturn_attribute.role_instruction_messages.items():
        scenario_parts.append(f"{role.value}: {instruction}")
    scenario_template = "\n".join(scenario_parts) if scenario_parts else None

    scenario_context = None
    if scenario_template and samples:
        scenario_context = self._formatter.format(
            samples[0], scenario_template, missing_values_allowed=True
        )

    # Build each environment once via registry
    registry = EnvironmentRegistry()
    for env_id, bound_tools in env_tools.items():
        config = self._env_configs.get(env_id)
        if not config:
            logger.warning(f"Environment config not found for '{env_id}'")
            continue

        if config.initial_state is not None and config.state_schema is not None:
            registry.register_static(config)
        else:
            registry.build(
                config,
                bound_tools,
                self._inference_engine,
                self._inference_config,
                scenario_context=scenario_context,
            )

    # Create N copies
    n = len(samples)
    sample_envs: list[dict[str, GeneratedToolEnvironment] | None] = []
    for _ in range(n):
        envs: dict[str, GeneratedToolEnvironment] = {}
        for env_id in env_tools:
            if env_id not in self._env_configs:
                continue
            try:
                copies = registry.create_copies(env_id, 1)
                envs[env_id] = copies[0]
            except KeyError:
                logger.warning(
                    f"Environment '{env_id}' not built. Skipping."
                )
        if envs:
            all_empty = all(not env.state for env in envs.values())
            if all_empty:
                logger.warning(
                    "Dropping sample: all environments have empty state."
                )
                sample_envs.append(None)
            else:
                sample_envs.append(envs)
        else:
            sample_envs.append(None)

    return sample_envs
```

- [ ] **Step 3: Run existing synthesizer tests to verify nothing breaks**

Run: `python -m pytest tests/unit/core/synthesis/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/oumi/core/synthesis/conversation_synthesizer.py
git commit -m "refactor: replace per-sample env init with EnvironmentRegistry"
```

---

### Task 7: End-to-End Verification

Verify the full pipeline works with the existing configs.

**Files:**
- No new files — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/unit/core/synthesis/ -v`
Expected: ALL PASS

- [ ] **Step 2: Verify imports are clean**

Run: `python -c "from oumi.core.synthesis.environment_registry import EnvironmentRegistry, derive_schema_from_tools, build_dependency_graph, sort_into_waves, build_collection_prompt, validate_collection; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 3: Verify environment.py has no dead code**

Run: `python -c "from oumi.core.synthesis.environment import GeneratedToolEnvironment; env_methods = [m for m in dir(GeneratedToolEnvironment) if not m.startswith('_')]; print(env_methods)"`
Expected: Should show `apply_result`, `apply_state_update`, `build_result_prompt`, `build_state_update_prompt`, `set_schema`, `set_state`, `state`, `summarize_state` — no `build_schema_prompt`, `apply_schema`, `build_initial_state_prompt`, `apply_initial_state`.

- [ ] **Step 4: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: address issues found during e2e verification"
```

Only commit if changes were made. Skip if everything passed clean.
