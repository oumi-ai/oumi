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

import json
import re
from typing import Any

import jsonschema

from oumi.core.configs.params.tool_params import ToolAttribute, ToolEnvironmentAttribute
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger

_ID_SUFFIX_PATTERN = re.compile(r"^(.+)_id$")


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


def _detect_primary_entity(tool: ToolAttribute) -> str | None:
    """Detect the primary entity a tool operates on."""
    if not tool.parameters or tool.parameters.get("type") != "object":
        return None
    props = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))

    for field_name in required:
        match = _ID_SUFFIX_PATTERN.match(field_name)
        if match:
            return match.group(1)

    for field_name in props:
        match = _ID_SUFFIX_PATTERN.match(field_name)
        if match:
            return match.group(1)
    return None


def derive_schema_from_tools(tools: list[ToolAttribute]) -> dict[str, Any]:
    """Derive a JSON Schema for environment state from tool definitions.

    Scans each tool's parameters and output_schema to infer entity collections.
    A field named ``<entity>_id`` is treated as the primary key and used to
    name the collection (pluralized). Other fields are merged into the record
    schema for that collection.

    Args:
        tools: List of tool definitions to inspect.

    Returns:
        A JSON Schema ``object`` whose ``properties`` map collection names to
        per-record schemas (each an ``object`` with ``additionalProperties``).
    """
    collections: dict[str, dict[str, dict[str, Any]]] = {}

    for tool in tools:
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
                    continue
                if field_name not in collections[collection_name]:
                    collections[collection_name][field_name] = dict(field_schema)

        # Extract fields from parameters
        if tool.parameters and tool.parameters.get("type") == "object":
            param_props = tool.parameters.get("properties", {})
            for field_name, field_schema in param_props.items():
                if field_name == f"{entity_name}_id":
                    continue
                # If this field references another entity, keep it
                # (it's a foreign key like lease.tenant_id)
                existing = collections[collection_name].get(field_name, {})
                merged = dict(existing)
                if "enum" in field_schema:
                    # Prefer enum from parameters (more specific)
                    merged["enum"] = field_schema["enum"]
                if "type" in field_schema and "type" not in merged:
                    merged["type"] = field_schema["type"]
                if merged:
                    collections[collection_name][field_name] = merged

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


def build_dependency_graph(
    schema: dict[str, Any],
) -> dict[str, set[str]]:
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


def sort_into_waves(
    graph: dict[str, set[str]],
) -> list[list[str]]:
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
        ready = sorted(name for name, deps in remaining.items() if not deps)

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
        existing_state: Previously generated collections (for FK context).
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
            state_lines.append(
                f"{coll_name}: {json.dumps(coll_data, indent=2)}"
            )
        user_parts.append(
            "Existing state:\n" + "\n\n".join(state_lines) + "\n"
        )

    schema_str = json.dumps(sub_schema, indent=2)
    user_parts.append(
        f"Generate {record_count} records for the "
        f"'{collection_name}' collection.\n\n"
        f"Each record must match this schema:\n{schema_str}\n"
    )

    # Foreign key instructions
    fk_fields = [
        f
        for f in sub_schema.get("properties", {})
        if _ID_SUFFIX_PATTERN.match(f)
    ]
    if fk_fields and existing_state:
        refs = []
        for fk in fk_fields:
            match = _ID_SUFFIX_PATTERN.match(fk)
            if match:
                target = _pluralize(match.group(1))
                if target in existing_state:
                    refs.append(
                        f"- {fk} must reference an existing "
                        f"ID from '{target}'"
                    )
        if refs:
            user_parts.append(
                "Referential integrity:\n"
                + "\n".join(refs) + "\n"
            )

    user_parts.append(
        "Output a JSON object keyed by string IDs (e.g., "
        '"1", "2" or domain-appropriate IDs). '
        "No markdown fences. Start with {."
    )

    if retry_error:
        user_parts.append(
            f"\nIMPORTANT: Your previous output failed "
            f"validation: {retry_error}\n"
            "Fix the issue and output only valid JSON."
        )

    messages = [
        Message(role=Role.SYSTEM, content=system_msg),
        Message(role=Role.USER, content="\n".join(user_parts)),
    ]
    return Conversation(messages=messages)


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
    3. Foreign key fields reference existing records.

    Args:
        collection_name: Name of this collection.
        data: The generated data (should be dict[str, dict]).
        sub_schema: JSON Schema for a single record.
        existing_state: Previously generated collections for FK checks.

    Returns:
        (True, None) if valid, (False, error_message) if not.
    """
    if not isinstance(data, dict):
        return (
            False,
            f"Expected a dict keyed by ID, "
            f"got {type(data).__name__}.",
        )

    # Validate each record against the sub-schema
    for record_id, record in data.items():
        try:
            jsonschema.validate(instance=record, schema=sub_schema)
        except jsonschema.ValidationError as e:
            return (
                False,
                f"Record '{record_id}' failed schema "
                f"validation: {e.message}",
            )

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
                    f"Record '{record_id}' references "
                    f"{field_name}='{value}' but '{value}' "
                    f"does not exist in '{target_collection}'.",
                )

    return True, None
