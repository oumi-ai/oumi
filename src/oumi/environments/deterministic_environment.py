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
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import (
    DeterministicToolOutput,
    Tool,
    ToolLookupError,
    ToolResult,
)


@dataclass
class DeterministicEnvironment(BaseEnvironment):
    """Environment that resolves tools from fixed lookups."""

    ENVIRONMENT_TYPE: ClassVar[str] = "deterministic"
    type: str = field(init=False, default=ENVIRONMENT_TYPE)

    def _coerce_tools(self, tools: list[Any]) -> list[Tool]:
        """Coerce tools and deterministic outputs into typed objects."""
        coerced_tools: list[Tool] = []
        for raw_tool in tools:
            tool = Tool.create(raw_tool)
            tool.deterministic_outputs = [
                entry
                if isinstance(entry, DeterministicToolOutput)
                else DeterministicToolOutput(**entry)
                for entry in tool.deterministic_outputs
            ]
            coerced_tools.append(tool)
        return coerced_tools

    def __post_init__(self):
        """Validate that deterministic tools have deterministic output entry."""
        super().__post_init__()
        for tool in self.tools:
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

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Resolve a deterministic tool call to its output.

        Raises:
            ToolLookupError: If no configured ``deterministic_outputs`` entry
                matches the provided arguments. The error message lists the
                configured inputs so the calling LLM can self-correct.
        """
        tool = self._get_tool_or_raise(tool_id)
        for entry in tool.deterministic_outputs:
            if entry.matches(arguments):
                return ToolResult(output=entry.output)
        available = [entry.input for entry in tool.deterministic_outputs]
        raise ToolLookupError(
            f"No deterministic output matches arguments "
            f"{json.dumps(arguments, sort_keys=True)} for tool '{tool_id}'. "
            f"Configured inputs: {json.dumps(available, sort_keys=True)}"
        )

    def sample_grounding(
        self, n: int, *, rng: random.Random
    ) -> list[DeterministicToolOutput]:
        """Sample grounding facts from the pool of deterministic outputs.

        Pools every ``DeterministicToolOutput`` across every tool owned by
        this environment, then draws ``min(n, len(pool))`` entries without
        replacement using the supplied RNG. Silent truncation — the
        synthesizer is responsible for surfacing a warning when applicable.
        """
        pool: list[DeterministicToolOutput] = [
            entry
            for tool in self.tools
            for entry in tool.deterministic_outputs
        ]
        return rng.sample(pool, min(n, len(pool)))
