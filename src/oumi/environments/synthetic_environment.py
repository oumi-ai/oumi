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
from dataclasses import dataclass, field
from typing import Any, ClassVar

from oumi.core.configs.params.base_params import BaseParams
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import ToolResult, _validate_json_schema_value


@dataclass
class SyntheticStateParams(BaseParams):
    """Optional state configuration for a synthetic environment."""

    state_schema: dict[str, Any] | None = None
    initial_state: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate state config consistency."""
        if self.state_schema is not None and self.initial_state is not None:
            _validate_json_schema_value(self.initial_state, self.state_schema)


@dataclass
class SyntheticEnvironment(BaseEnvironment):
    """LLM-simulated environment with optional mutable state."""

    ENVIRONMENT_TYPE: ClassVar[str] = "synthetic"
    type: str = field(init=False, default=ENVIRONMENT_TYPE)
    system_prompt: str = ""
    state_params: SyntheticStateParams | dict[str, Any] | None = None
    cache_by_input: bool = True

    def __post_init__(self):
        """Validate synthetic-only fields after common environment validation."""
        if isinstance(self.state_params, dict):
            self.state_params = SyntheticStateParams(**self.state_params)
        super().__post_init__()
        self._cache: dict[str, ToolResult] = {}
        self._state: dict[str, Any] | None = (
            copy.deepcopy(self.state_params.initial_state)
            if self.state_params is not None
            and self.state_params.initial_state is not None
            else None
        )
        if not self.system_prompt:
            raise ValueError("SyntheticEnvironment.system_prompt cannot be empty.")
        if self.state_params is not None and self.cache_by_input:
            raise ValueError(
                "SyntheticEnvironment.cache_by_input must be False when "
                "state_params is provided."
            )
        for tool in self.tools:
            if tool.deterministic_outputs:
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
        if not self.cache_by_input:
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
        if not self.cache_by_input:
            return
        self._cache[self._cache_key(tool_id, arguments)] = ToolResult(
            output=copy.deepcopy(result.output),
            updated_state=copy.deepcopy(result.updated_state),
        )

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a synthetic tool call."""
        self._get_tool_or_raise(tool_id)
        raise NotImplementedError("SyntheticEnvironment.step() is not implemented yet.")
