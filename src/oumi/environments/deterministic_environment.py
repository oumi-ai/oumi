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

"""Deterministic environment with fixed lookup responses."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

import jsonschema
from pydantic import JsonValue
from pydantic import ValidationError as PydanticValidationError

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.utils import parse_env_kwargs
from oumi.utils.logging import logger

# JSON Schema keywords whose values contain subschemas.
_SUBSCHEMA = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_LIST = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP = frozenset({"dependentSchemas", "patternProperties", "properties"})


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local ``$ref``, preserving sibling overrides."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#"):
        return schema
    target: Any = root
    for part in ref[1:].split("/"):
        if not part:
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or part not in target:
            return schema
        target = target[part]
    if not isinstance(target, dict):
        return schema
    return {**target, **{k: v for k, v in schema.items() if k != "$ref"}}


def _fill_argument_defaults(
    arguments: dict[str, Any],
    schema: dict[str, Any],
    root: dict[str, Any] | None = None,
    seen: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return a copy of arguments with property defaults applied recursively."""
    if root is None:
        root = schema
    result = copy.deepcopy(arguments)
    properties = _resolve_ref(schema, root).get("properties", {})
    if not isinstance(properties, dict):
        return result

    for name, property_schema in properties.items():
        if not isinstance(property_schema, dict):
            continue
        ref = property_schema.get("$ref")
        property_schema = _resolve_ref(property_schema, root)
        cyclic = isinstance(ref, str) and ref in seen
        if name not in result and "default" in property_schema and not cyclic:
            result[name] = copy.deepcopy(property_schema["default"])
        if isinstance(result.get(name), dict):
            result[name] = _fill_argument_defaults(
                result[name],
                property_schema,
                root,
                seen | {ref} if isinstance(ref, str) else seen,
            )
    return result


def _unfillable_default_paths(
    node: Any,
    path: str,
    root: dict[str, Any],
    reachable: bool = True,
    seen: frozenset[tuple[str, bool]] = frozenset(),
) -> list[str]:
    """Find defaults outside ``properties`` chains."""
    if not isinstance(node, dict):
        return []

    ref = node.get("$ref")
    if isinstance(ref, str):
        if (ref, reachable) in seen:
            return []
        node, seen = _resolve_ref(node, root), seen | {(ref, reachable)}

    found: list[str] = []
    if not reachable and "default" in node:
        found.append(f"{path}.default")
    for key, value in node.items():
        if key in _SUBSCHEMA:
            found += _unfillable_default_paths(
                value, f"{path}.{key}", root, False, seen
            )
        elif key in _SUBSCHEMA_LIST and isinstance(value, list):
            for index, item in enumerate(value):
                found += _unfillable_default_paths(
                    item, f"{path}.{key}[{index}]", root, False, seen
                )
        elif key in _SUBSCHEMA_MAP and isinstance(value, dict):
            child_reachable = reachable and key == "properties"
            for name, subschema in value.items():
                found += _unfillable_default_paths(
                    subschema, f"{path}.{key}.{name}", root, child_reachable, seen
                )
    return found


@dataclass
class ToolLookupEntry(BaseParams):
    """One (input, output) pair in a deterministic env's lookup table.

    ``output`` may be any JSON value (scalar, list, object, or null).
    """

    input: dict[str, Any] = field(default_factory=dict)
    output: JsonValue = None

    def input_key(self) -> str:
        """Canonical JSON form of ``input`` for matching and dedup."""
        return json.dumps(self.input, sort_keys=True)

    def matches(self, arguments: dict[str, Any]) -> bool:
        """Check if the input matches the given arguments."""
        return self.input_key() == json.dumps(arguments, sort_keys=True)


@dataclass
class DeterministicEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for DeterministicEnvironment."""

    lookup_table: dict[str, list[ToolLookupEntry]] = field(default_factory=dict)
    """Per-tool list of (input, output) entries, keyed by tool id."""

    def __post_init__(self) -> None:
        """Coerce raw entry dicts into ``ToolLookupEntry`` instances."""
        self.lookup_table = {
            tool_id: [
                entry
                if isinstance(entry, ToolLookupEntry)
                else ToolLookupEntry(**entry)
                for entry in entries
            ]
            for tool_id, entries in self.lookup_table.items()
        }


@register_environment("deterministic")
class DeterministicEnvironment(BaseEnvironment):
    """Environment that resolves tools from a per-tool lookup table.

    The env's ``env_kwargs.lookup_table`` is the source of truth for tool
    behavior. Tools listed in ``params.tools`` declare contracts only;
    their data lives on the env.
    """

    tool_params_cls = ToolParams

    def __init__(
        self,
        params: EnvironmentParams,
        kwargs: DeterministicEnvironmentKwargs,
    ) -> None:
        """Initialize a DeterministicEnvironment."""
        self._params = params
        self._kwargs = kwargs
        self._tools_by_id: dict[str, ToolParams] = {
            tool.id: tool for tool in params.tools
        }
        self._validate_lookup_table()
        self._warn_grounding_key_collisions()

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Resolve a batch of deterministic tool calls to their outputs."""
        return [self._resolve_one(tool_id, args) for tool_id, args in calls]

    def _resolve_one(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools_by_id.get(tool_id)
        if tool is None:
            raise ValueError(
                f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
                f"Available tools: {sorted(self._tools_by_id)}"
            )
        arguments = _fill_argument_defaults(arguments, tool.parameters)
        entries = self._kwargs.lookup_table.get(tool_id, [])
        for entry in entries:
            if entry.matches(arguments):
                return ToolResult(output=entry.output)
        available = [entry.input for entry in entries]
        raise ToolLookupError(
            f"No deterministic output matches arguments "
            f"{json.dumps(arguments, sort_keys=True)} for tool '{tool_id}'. "
            f"Configured inputs: {json.dumps(available, sort_keys=True)}"
        )

    def sample_grounding(
        self,
        n: int,
        *,
        rng: random.Random,
        tool_ids: set[str] | None = None,
    ) -> list[GroundingFact]:
        """Sample grounding facts from per-tool projected pools.

        Walks every tool that has a per-tool entry in
        ``params.grounding.tools``. Each entry in that tool's lookup table
        is projected to its ``input`` fields (merged with ``output`` when
        the output is a dict), filtered through the configured ``fields``
        whitelist. Tools without a grounding entry contribute nothing.
        """
        grounding = self._params.grounding
        if grounding is None or not grounding.tools:
            return []
        pool: list[GroundingFact] = []
        for tool in self._params.tools:
            tool_grounding = grounding.tools.get(tool.id)
            if tool_grounding is None:
                continue
            if tool_ids is not None and tool.id not in tool_ids:
                continue
            whitelist = set(tool_grounding.fields)
            for entry in self._kwargs.lookup_table.get(tool.id, []):
                # Non-dict outputs (scalars/lists) have no named fields to
                # project, so they ground on their input fields only; dict
                # outputs merge both.
                row = dict(entry.input)
                if isinstance(entry.output, dict):
                    row.update(entry.output)
                projected = {
                    key: value for key, value in row.items() if key in whitelist
                }
                pool.append(GroundingFact(data=projected))
        return rng.sample(pool, min(n, len(pool)))

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DeterministicEnvironment:
        """Build a DeterministicEnvironment from its params object."""
        kwargs = parse_env_kwargs(
            DeterministicEnvironmentKwargs,
            params,
            env_label="DeterministicEnvironment",
        )
        return cls(params, kwargs)

    def _validate_lookup_table(self) -> None:
        """Validate and normalize the lookup table.

        Entries under stale keys (no matching tool) are neither normalized nor
        validated. Inputs must conform to the tool's ``parameters``; outputs
        must be JSON values, and conform to ``output_schema`` only when the
        tool declares one.
        """
        for tool_id in self._kwargs.lookup_table:
            if tool_id not in self._tools_by_id:
                logger.warning(
                    "Environment '%s': lookup_table.'%s' references unknown "
                    "tool. Entries will be ignored.",
                    self._params.id,
                    tool_id,
                )
        for tool in self._params.tools:
            entries = self._kwargs.lookup_table.get(tool.id, [])
            if not entries:
                raise ValueError(
                    f"Tool '{tool.id}' has no entries in lookup_table for "
                    f"environment '{self._params.id}'."
                )
            unfillable = _unfillable_default_paths(
                tool.parameters, "parameters", tool.parameters
            )
            if unfillable:
                raise ValueError(
                    f"Tool '{tool.id}' in environment '{self._params.id}' declares "
                    "schema defaults that are never applied, at "
                    f"{sorted(unfillable)}. Defaults are only filled along "
                    "'properties' chains; move them onto a property."
                )
            seen: set[str] = set()
            for entry in entries:
                entry.input = _fill_argument_defaults(entry.input, tool.parameters)
                try:
                    tool.validate_arguments(entry.input)
                except ToolArgumentError as e:
                    raise ValueError(
                        f"Tool '{tool.id}' has lookup_table entry with invalid "
                        f"input {entry.input}: {e}"
                    ) from e
                # jsonschema accepts non-JSON values a JsonValue output rejects
                # (e.g. a dict with int keys), so check against the consumer first.
                try:
                    ToolResult(output=entry.output)
                except PydanticValidationError as e:
                    raise ValueError(
                        f"Tool '{tool.id}' has lookup_table entry with non-JSON "
                        f"output {entry.output} for input {entry.input}: {e}"
                    ) from e
                if tool.output_schema is not None:
                    try:
                        jsonschema.validate(entry.output, tool.output_schema)
                    except jsonschema.ValidationError as e:
                        raise ValueError(
                            f"Tool '{tool.id}' has lookup_table entry with invalid "
                            f"output {entry.output} for input {entry.input}: {e}"
                        ) from e
                key = entry.input_key()
                if key in seen:
                    raise ValueError(
                        f"Tool '{tool.id}' has duplicate input entry: {entry.input}"
                    )
                seen.add(key)

    def _warn_grounding_key_collisions(self) -> None:
        """Warn once when a dict output shadows a whitelisted input field.

        Only whitelisted fields matter — a collision on any other key is
        dropped by the projection and never reaches a grounding fact.
        """
        grounding = self._params.grounding
        if grounding is None or not grounding.tools:
            return
        for tool_id, tool_grounding in grounding.tools.items():
            whitelist = set(tool_grounding.fields)
            shadowed: set[str] = set()
            for entry in self._kwargs.lookup_table.get(tool_id, []):
                if isinstance(entry.output, dict):
                    shadowed |= entry.input.keys() & entry.output.keys() & whitelist
            if shadowed:
                logger.warning(
                    "Environment '%s': tool '%s' grounding field(s) %s appear "
                    "in both input and output; the output value shadows the "
                    "input in grounding facts.",
                    self._params.id,
                    tool_id,
                    sorted(shadowed),
                )
