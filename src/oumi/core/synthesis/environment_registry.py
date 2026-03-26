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

_ID_SUFFIX_PATTERN = re.compile(r"^(.+)_id$")


def _pluralize(word: str) -> str:
    """Naive English pluralization for entity names."""
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
    """Derive a JSON Schema for environment state from tool definitions."""
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
                existing = collections[collection_name].get(field_name, {})
                merged = dict(existing)
                if "enum" in field_schema:
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
