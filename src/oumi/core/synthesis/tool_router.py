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
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.types.tool_call import ToolDefinition, ToolResult
from oumi.environments.base_environment import BaseEnvironment


@dataclass
class ToolRouter:
    """Routes LLM tool calls to environment-owned tools.

    Use :meth:`for_sample` to obtain a per-sample clone whose envs have
    independent state; the synthesizer builds one clone per sample at
    batch entry so tool mutations don't leak across samples.
    """

    tool_specs: list[ToolDefinition]
    tools_by_id: dict[str, ToolParams]
    env_by_id: dict[str, BaseEnvironment]
    tool_to_env: dict[str, BaseEnvironment]
    env_params_by_id: dict[str, EnvironmentParams]
    tool_env_map: dict[str, str]
    on_env_built: Callable[[BaseEnvironment], None] | None

    @classmethod
    def from_environment_config(
        cls,
        env_config: EnvironmentConfig,
        on_env_built: Callable[[BaseEnvironment], None] | None = None,
    ) -> ToolRouter:
        """Build a router from an env config."""
        tool_env_map = env_config.tool_environment_map
        included_env_ids = set(tool_env_map.values()) | {
            env_params.id
            for env_params in env_config.environments
            if env_params.grounding is not None
        }

        env_params_by_id: dict[str, EnvironmentParams] = {}
        env_by_id: dict[str, BaseEnvironment] = {}
        for env_params in env_config.environments:
            if env_params.id not in included_env_ids:
                continue
            env_params_by_id[env_params.id] = env_params
            env = build_environment(env_params)
            if on_env_built is not None:
                on_env_built(env)
            env_by_id[env_params.id] = env
            # Parent only reads requires_isolation() off this template; for_sample()
            # rebuilds it per-sample, so free it now (e.g. a DB temp snapshot).
            if env.requires_isolation():
                env.close()

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
            env_params_by_id=env_params_by_id,
            tool_env_map=tool_env_map,
            on_env_built=on_env_built,
        )

    def for_sample(self) -> ToolRouter:
        """Return a router safe to use for one sample.

        Envs whose ``requires_isolation()`` returns ``True`` are rebuilt via
        ``build_environment`` so their mutable state stays independent across
        samples; ``on_env_built`` re-runs on those fresh instances. Envs that
        don't require isolation (e.g. ``DeterministicEnvironment`` and
        stateless ``SyntheticEnvironment``) are shared with the parent
        router to avoid the per-sample build + inference-engine attach cost.
        """
        env_by_id_new: dict[str, BaseEnvironment] = {}
        for env_id, parent_env in self.env_by_id.items():
            if not parent_env.requires_isolation():
                env_by_id_new[env_id] = parent_env
                continue
            fresh = build_environment(self.env_params_by_id[env_id])
            if self.on_env_built is not None:
                self.on_env_built(fresh)
            env_by_id_new[env_id] = fresh

        return ToolRouter(
            tool_specs=self.tool_specs,
            tools_by_id=self.tools_by_id,
            env_by_id=env_by_id_new,
            tool_to_env={
                tool_id: env_by_id_new[env_id]
                for tool_id, env_id in self.tool_env_map.items()
            },
            env_params_by_id=self.env_params_by_id,
            tool_env_map=self.tool_env_map,
            on_env_built=self.on_env_built,
        )

    def close(self) -> None:
        """Release per-sample envs this router owns.

        Only isolation-requiring envs are rebuilt (and thus owned) per router;
        shared envs belong to the parent router and are left open. Each close is
        guarded so one env's failure can't leak the rest; the first error re-raises.
        """
        first_error: BaseException | None = None
        for env in self.env_by_id.values():
            if not env.requires_isolation():
                continue
            try:
                env.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

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
