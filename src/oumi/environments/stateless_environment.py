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

"""Stateless environment with optional response caching."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.core.configs.params.base_params import BaseParams
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import BaseTool
from oumi.environments.types import ToolEnvironmentType


@dataclass
class GeneratedToolOutput(BaseParams):
    """Configuration for tool output in a stateless environment."""

    instruction: str

    def __post_init__(self):
        """Validate the instruction field is not empty."""
        if not self.instruction:
            raise ValueError("GeneratedToolOutput.instruction cannot be empty.")


@dataclass
class StatelessTool(BaseTool):
    """Tool bound to a stateless environment."""

    generated_output: GeneratedToolOutput | None = None

    def __post_init__(self):
        """Validate stateless tool fields."""
        super().__post_init__()
        if self.generated_output is None:
            raise ValueError(
                f"StatelessTool '{self.id}' must have a generated_output."
            )

    @classmethod
    def create(cls, raw: Mapping[str, Any] | BaseTool) -> StatelessTool:
        """Create a StatelessTool from raw config data."""
        if isinstance(raw, StatelessTool):
            return raw
        if isinstance(raw, BaseTool):
            raise TypeError(
                f"Cannot coerce {type(raw).__name__} to StatelessTool. "
                f"Use a mapping with 'generated_output'."
            )
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, "
                f"got {type(raw)}"
            )
        generated_output = raw.get("generated_output")
        if isinstance(generated_output, Mapping):
            generated_output = GeneratedToolOutput(**generated_output)
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=raw.get("parameters", {}),
            generated_output=generated_output,
        )


@dataclass
class StatelessEnvironment(BaseEnvironment):
    """Environment that simulates outputs and optionally caches by input.

    Each tool is a ``StatelessTool`` with a ``generated_output`` instruction
    for the LLM. When ``cache_by_input`` is True, the environment caches
    results keyed by (tool_id, arguments) so repeated calls with the same
    input return consistent results.
    """

    ENVIRONMENT_TYPE: ClassVar[ToolEnvironmentType] = ToolEnvironmentType.STATELESS
    type: ToolEnvironmentType = field(init=False, default=ToolEnvironmentType.STATELESS)
    system_prompt: str = ""
    cache_by_input: bool = True
    tools: list[StatelessTool] = field(default_factory=list)  # type: ignore[assignment]

    def __post_init__(self):
        """Validate and initialize the response cache."""
        super().__post_init__()
        self._cache: dict[str, str] = {}
        self._frozen_context: str | None = None

    def _coerce_tools(self, tools: list[Any]) -> list[StatelessTool]:
        """Coerce raw tool definitions into StatelessTool instances."""
        return [StatelessTool.create(t) for t in tools]

    def _validate_type_specific(self) -> None:
        """Validate stateless-specific fields."""
        if not self.system_prompt:
            raise ValueError("StatelessEnvironment.system_prompt cannot be empty.")

    @property
    def frozen_context(self) -> str | None:
        """Frozen context generated once at build time."""
        return self._frozen_context

    def set_frozen_context(self, context: str) -> None:
        """Set the frozen context (called once during environment build)."""
        self._frozen_context = context

    @staticmethod
    def _cache_key(tool_id: str, arguments: dict[str, Any]) -> str:
        """Build a stable cache key from tool id and arguments."""
        return f"{tool_id}::{json.dumps(arguments, sort_keys=True)}"

    def resolve_cached(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> str | None:
        """Look up a cached result for the given tool call."""
        if not self.cache_by_input:
            return None
        return self._cache.get(self._cache_key(tool_id, arguments))

    def cache_result(
        self, tool_id: str, arguments: dict[str, Any], result: str
    ) -> None:
        """Store a generated result in the cache. No-op if caching is disabled."""
        if self.cache_by_input:
            self._cache[self._cache_key(tool_id, arguments)] = result
