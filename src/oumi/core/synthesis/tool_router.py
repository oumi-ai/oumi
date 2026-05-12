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

"""Routing layer between LLM tool calls and environment-owned tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oumi.builders.environments import build_environment
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.types.tool_call import ToolDefinition, ToolResult
from oumi.environments.base_environment import BaseEnvironment


@dataclass
class ToolRouter:
    """Routes LLM tool calls to environment-owned tools."""

    tool_specs: list[ToolDefinition]
    tools_by_id: dict[str, ToolParams]
    env_by_id: dict[str, BaseEnvironment]
    tool_to_env: dict[str, BaseEnvironment]

    @classmethod
    def from_environment_config(
        cls,
        env_config: EnvironmentConfig,
        on_env_built: Callable[[BaseEnvironment], None] | None = None,
    ) -> ToolRouter:
        """Build a router from an env config."""
        tool_env_map = env_config.tool_environment_map
        reachable_env_ids = set(tool_env_map.values())

        env_by_id: dict[str, BaseEnvironment] = {}
        for env_params in env_config.environments:
            if env_params.id not in reachable_env_ids:
                continue
            env = build_environment(env_params)
            if on_env_built is not None:
                on_env_built(env)
            env_by_id[env_params.id] = env

        tools_by_id = {tool.id: tool for tool in env_config.all_tools}
        tool_to_env = {
            tool_id: env_by_id[env_id] for tool_id, env_id in tool_env_map.items()
        }
        tool_specs = [tool.to_tool_definition() for tool in env_config.all_tools]

        return cls(
            tool_specs=tool_specs,
            tools_by_id=tools_by_id,
            env_by_id=env_by_id,
            tool_to_env=tool_to_env,
        )

    def parse_and_validate_arguments(
        self, tool_id: str, raw_arguments: str
    ) -> dict[str, Any]:
        """Parse wire JSON args and validate against the tool's parameters schema."""
        if tool_id not in self.tools_by_id:
            raise ToolLookupError(
                f"Unknown tool '{tool_id}'. Known: {sorted(self.tools_by_id)}"
            )
        tool = self.tools_by_id[tool_id]
        try:
            parsed = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as e:
            raise ToolArgumentError(
                f"Tool '{tool_id}' arguments are not valid JSON: {e}"
            ) from e
        if not isinstance(parsed, dict):
            raise ToolArgumentError(
                f"Tool '{tool_id}' arguments must be a JSON object, got "
                f"{type(parsed).__name__}."
            )
        tool.validate_arguments(parsed)
        return parsed

    def route_batch(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Dispatch a batch of (tool_id, args) pairs; preserves call order."""
        if not calls:
            return []

        groups: dict[int, list[tuple[int, str, dict[str, Any]]]] = {}
        for idx, (tool_id, args) in enumerate(calls):
            if tool_id not in self.tool_to_env:
                raise ToolLookupError(
                    f"Unknown tool '{tool_id}'. Known: {sorted(self.tool_to_env)}"
                )
            env = self.tool_to_env[tool_id]
            groups.setdefault(id(env), []).append((idx, tool_id, args))

        results: list[ToolResult | None] = [None] * len(calls)
        for group in groups.values():
            env = self.tool_to_env[group[0][1]]
            outputs = env.step([(tid, args) for _, tid, args in group])
            for (idx, _, _), out in zip(group, outputs):
                results[idx] = out

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]
