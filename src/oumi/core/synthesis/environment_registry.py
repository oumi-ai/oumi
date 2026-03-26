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

import re
from typing import Any

from oumi.core.configs.params.tool_params import ToolAttribute
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
