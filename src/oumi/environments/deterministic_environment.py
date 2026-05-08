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

import dataclasses
import json
import random
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.configs.params.tool_params import ToolLookupError, ToolParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.utils.logging import logger


@dataclass
class ToolLookupEntry(BaseParams):
    """One (input, output) pair in a deterministic env's lookup table."""

    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

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
        self._tool_ids = {tool.id for tool in params.tools}
        self._validate_lookup_table()

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Resolve a deterministic tool call to its output.

        Raises:
            ValueError: If ``tool_id`` is not declared in this env's tools list.
            ToolLookupError: If no entry in the env's lookup table matches
                the provided arguments.
        """
        if tool_id not in self._tool_ids:
            raise ValueError(
                f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
                f"Available tools: {sorted(self._tool_ids)}"
            )
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
        is projected via ``{**input, **output}`` filtered through the
        configured ``fields`` whitelist. Tools without a grounding entry
        contribute nothing.
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
                row = {**entry.input, **entry.output}
                projected = {
                    key: value for key, value in row.items() if key in whitelist
                }
                pool.append(GroundingFact(data=projected))
        return rng.sample(pool, min(n, len(pool)))

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DeterministicEnvironment:
        """Build a DeterministicEnvironment from its params object."""
        raw_kwargs = params.env_kwargs or {}
        known = {f.name for f in dataclasses.fields(DeterministicEnvironmentKwargs)}
        unknown = set(raw_kwargs) - known
        if unknown:
            raise ValueError(
                f"DeterministicEnvironment got unknown env_kwargs: "
                f"{sorted(unknown)}. Known: {sorted(known)}"
            )
        kwargs = DeterministicEnvironmentKwargs(**raw_kwargs)
        kwargs.finalize_and_validate()
        return cls(params, kwargs)

    def _validate_lookup_table(self) -> None:
        """Validate the env's lookup_table against its tool list.

        - Stale ``lookup_table`` keys (no matching tool): log a warning;
          entries are dormant.
        - Tools without entries: hard error.
        - Duplicate inputs within a tool's entries: hard error.
        """
        for tool_id in self._kwargs.lookup_table:
            if tool_id not in self._tool_ids:
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
            seen: set[str] = set()
            for entry in entries:
                key = entry.input_key()
                if key in seen:
                    raise ValueError(
                        f"Tool '{tool.id}' has duplicate input entry: {entry.input}"
                    )
                seen.add(key)
