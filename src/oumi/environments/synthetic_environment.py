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

"""Synthetic environment backed by LLM-simulated tool execution."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import jsonschema

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.deterministic_tool import DeterministicTool


@dataclass
class SyntheticStateParams(BaseParams):
    """Optional state configuration for a synthetic environment."""

    state_schema: dict[str, Any] | None = None
    initial_state: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate state config consistency."""
        if self.state_schema is not None and self.initial_state is not None:
            jsonschema.validate(self.initial_state, self.state_schema)


@dataclass
class SyntheticEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for SyntheticEnvironment."""

    system_prompt: str = ""
    state_params: SyntheticStateParams | None = None
    cache_by_input: bool = True

    def __post_init__(self) -> None:
        """Coerce state_params dict into SyntheticStateParams if needed."""
        if isinstance(self.state_params, dict):
            self.state_params = SyntheticStateParams(**self.state_params)

    def __finalize_and_validate__(self) -> None:
        """Finalize and validate the kwargs."""
        if not self.system_prompt:
            raise ValueError(
                "SyntheticEnvironmentKwargs.system_prompt cannot be empty."
            )
        if self.state_params is not None and self.cache_by_input:
            raise ValueError(
                "SyntheticEnvironmentKwargs.cache_by_input must be False when "
                "state_params is provided."
            )


@register_environment("synthetic")
class SyntheticEnvironment(BaseEnvironment):
    """LLM-simulated environment with optional mutable state."""

    def __init__(
        self,
        params: EnvironmentParams,
        kwargs: SyntheticEnvironmentKwargs,
    ) -> None:
        """Initialize a SyntheticEnvironment with the given params and kwargs."""
        self._params = params
        self._kwargs = kwargs
        self._validate_tools(params.tools)
        self._cache: dict[str, ToolResult] = {}
        self._state: dict[str, Any] | None = (
            copy.deepcopy(kwargs.state_params.initial_state)
            if kwargs.state_params is not None
            and kwargs.state_params.initial_state is not None
            else None
        )

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> SyntheticEnvironment:
        """Build a SyntheticEnvironment from its params object."""
        kwargs = SyntheticEnvironmentKwargs(**(params.env_kwargs or {}))
        kwargs.finalize_and_validate()
        return cls(params, kwargs)

    @staticmethod
    def _validate_tools(tools: list[ToolParams]) -> None:
        for tool in tools:
            if isinstance(tool, DeterministicTool):
                raise ValueError(
                    f"Synthetic tool '{tool.id}' cannot define deterministic_outputs."
                )

    @property
    def current_state(self) -> dict[str, Any] | None:
        """Return the current in-memory state snapshot."""
        if self._state is None:
            return None
        return copy.deepcopy(self._state)

    @staticmethod
    def _cache_key(tool_id: str, arguments: dict[str, Any]) -> str:
        """Build a stable cache key from tool id and arguments."""
        return f"{tool_id}::{json.dumps(arguments, sort_keys=True)}"

    def _resolve_cached(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> ToolResult | None:
        """Look up a cached result for the given tool call."""
        if not self._kwargs.cache_by_input:
            return None
        result = self._cache.get(self._cache_key(tool_id, arguments))
        if result is None:
            return None
        return ToolResult(
            output=copy.deepcopy(result.output),
            updated_state=copy.deepcopy(result.updated_state),
        )

    def _cache_result(
        self, tool_id: str, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        """Store a generated result in the cache."""
        if not self._kwargs.cache_by_input:
            return
        self._cache[self._cache_key(tool_id, arguments)] = ToolResult(
            output=copy.deepcopy(result.output),
            updated_state=copy.deepcopy(result.updated_state),
        )

    def _lookup_tool(self, tool_id: str) -> ToolParams:
        for tool in self._params.tools:
            if tool.id == tool_id:
                return tool
        raise ValueError(
            f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
            f"Available tools: {[tool.id for tool in self._params.tools]}"
        )

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a synthetic tool call."""
        self._lookup_tool(tool_id)
        raise NotImplementedError("SyntheticEnvironment.step() is not implemented yet.")
