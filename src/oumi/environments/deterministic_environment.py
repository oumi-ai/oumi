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

import json
import random
from dataclasses import dataclass
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.configs.params.tool_params import ToolLookupError, ToolResult
from oumi.core.registry import register_environment
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)


@dataclass
class DeterministicEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for DeterministicEnvironment.

    Deterministic environments take no additional configuration beyond the
    common `EnvironmentParams` fields, so this is intentionally empty. Any
    keys present in `params.env_kwargs` will trigger a validation error.
    """


@register_environment("deterministic")
class DeterministicEnvironment(BaseEnvironment):
    """Environment that resolves tools from fixed lookups.

    Tools must declare at least one `deterministic_outputs` entry; calling
    `step()` returns the matching output (or `None` if no entry matches).
    """

    tool_params_cls = DeterministicTool

    def __init__(
        self,
        params: EnvironmentParams,
        kwargs: DeterministicEnvironmentKwargs,
    ) -> None:
        """Initialize a DeterministicEnvironment with the given params and kwargs."""
        self._params = params
        self._kwargs = kwargs
        self._validate_tools(params.tools)

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Resolve a deterministic tool call to its output.

        Raises:
            ToolLookupError: If no configured ``deterministic_outputs`` entry
                matches the provided arguments. The error message lists the
                configured inputs so the calling LLM can self-correct.
        """
        tool = self._lookup_tool(tool_id)
        for entry in tool.deterministic_outputs:
            if entry.matches(arguments):
                return ToolResult(output=entry.output)
        available = [entry.input for entry in tool.deterministic_outputs]
        raise ToolLookupError(
            f"No deterministic output matches arguments "
            f"{json.dumps(arguments, sort_keys=True)} for tool '{tool_id}'. "
            f"Configured inputs: {json.dumps(available, sort_keys=True)}"
        )

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> DeterministicEnvironment:
        """Build a DeterministicEnvironment from its params object."""
        if params.env_kwargs:
            raise ValueError(
                f"DeterministicEnvironment does not accept env_kwargs, "
                f"got: {sorted(params.env_kwargs)}"
            )
        kwargs = DeterministicEnvironmentKwargs()
        kwargs.finalize_and_validate()
        for tool in params.tools:
            tool.deterministic_outputs = [
                entry
                if isinstance(entry, DeterministicToolOutput)
                else DeterministicToolOutput(**entry)
                for entry in tool.deterministic_outputs
            ]
        return cls(params, kwargs)

    @staticmethod
    def _validate_tools(tools: list[DeterministicTool]) -> None:
        for tool in tools:
            if not tool.deterministic_outputs:
                raise ValueError(
                    f"Deterministic tool '{tool.id}' must have at least one "
                    "deterministic_output entry."
                )
            seen: set[str] = set()
            for entry in tool.deterministic_outputs:
                key = json.dumps(entry.input, sort_keys=True)
                if key in seen:
                    raise ValueError(
                        f"Deterministic tool '{tool.id}' has duplicate "
                        f"deterministic input entry: {entry.input}"
                    )
                seen.add(key)

    def _lookup_tool(self, tool_id: str) -> DeterministicTool:
        for tool in self._params.tools:
            if tool.id == tool_id:
                return tool
        raise ValueError(
            f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
            f"Available tools: {[tool.id for tool in self._params.tools]}"
        )

    def sample_grounding(
        self,
        n: int,
        *,
        rng: random.Random,
        tool_ids: set[str] | None = None,
    ) -> list[GroundingFact]:
        """Sample grounding facts from per-tool projected pools.

        Walks every tool in this environment that declares a ``grounding`` block.
        For each row in that tool's ``deterministic_outputs``, projects
        ``{**input, **output}`` to the configured ``fields`` and emits a
        ``GroundingFact``. Fields whitelisted but absent from a row are silently
        dropped.

        Tools without a ``grounding`` block contribute nothing. Tools whose
        ``id`` is not in ``tool_ids`` are also skipped when a filter is supplied.
        Per-tool projections are concatenated, then sampled without replacement.
        """
        pool: list[GroundingFact] = []
        for tool in self._params.tools:
            grounding = getattr(tool, "grounding", None)
            if grounding is None:
                continue
            if tool_ids is not None and tool.id not in tool_ids:
                continue
            whitelist = set(grounding.fields)
            for entry in tool.deterministic_outputs:
                row = {**entry.input, **entry.output}
                projected = {key: value for key, value in row.items() if key in whitelist}
                pool.append(GroundingFact(data=projected))
        sampled = rng.sample(pool, min(n, len(pool)))
        return sampled
