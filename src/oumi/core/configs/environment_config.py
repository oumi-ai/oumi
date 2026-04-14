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

"""Configuration for agentic environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from oumi.core.configs.base_config import BaseConfig

if TYPE_CHECKING:
    from oumi.environments.base_environment import BaseEnvironment
    from oumi.environments.base_tool import Tool


@dataclass
class EnvironmentConfig(BaseConfig):
    """Top-level config for environment-first tool definitions."""

    environments: list[Any] = field(default_factory=list)
    """Reusable environments and their owned tools."""

    def __post_init__(self):
        """Verifies/populates params."""
        self.environments = [
            self._coerce_environment(environment) for environment in self.environments
        ]

        env_ids: set[str] = set()
        tool_ids: set[str] = set()

        for environment in self.environments:
            if environment.id in env_ids:
                raise ValueError(
                    f"EnvironmentConfig.environments contains duplicate "
                    f"environment id '{environment.id}'."
                )
            env_ids.add(environment.id)

            for tool in environment.tools:
                if tool.id in tool_ids:
                    raise ValueError(
                        f"EnvironmentConfig.environments contains duplicate "
                        f"tool id '{tool.id}'."
                    )
                tool_ids.add(tool.id)

    @property
    def all_tools(self) -> list[Tool]:
        """Flatten all tools across environments."""
        return [tool for environment in self.environments for tool in environment.tools]

    @property
    def tool_environment_map(self) -> dict[str, str]:
        """Map each tool id to the environment that owns it."""
        return {
            tool.id: environment.id
            for environment in self.environments
            for tool in environment.tools
        }

    def get_environment(self, environment_id: str) -> BaseEnvironment | None:
        """Look up an environment by id."""
        for environment in self.environments:
            if environment.id == environment_id:
                return environment
        return None

    def get_tool(self, tool_id: str) -> Tool | None:
        """Look up a tool by id."""
        for tool in self.all_tools:
            if tool.id == tool_id:
                return tool
        return None

    def resolve_tools(
        self,
        environment_ids: list[str] | None = None,
        tool_ids: list[str] | None = None,
    ) -> list[Tool]:
        """Resolve tools from selected environments and optional tool ids.

        Raises:
            ValueError: If any environment_id or tool_id is not found.
        """
        all_env_ids = {env.id for env in self.environments}

        if environment_ids:
            unknown_envs = set(environment_ids) - all_env_ids
            if unknown_envs:
                raise ValueError(
                    f"Unknown environment id(s): {sorted(unknown_envs)}. "
                    f"Defined: {sorted(all_env_ids)}"
                )
            selected_environment_ids = environment_ids
        else:
            selected_environment_ids = list(all_env_ids)

        selected_environments = [
            environment
            for environment in self.environments
            if environment.id in set(selected_environment_ids)
        ]
        tools = [
            tool for environment in selected_environments for tool in environment.tools
        ]

        if tool_ids:
            available_tool_ids = {tool.id for tool in tools}
            unknown_tools = set(tool_ids) - available_tool_ids
            if unknown_tools:
                raise ValueError(
                    f"Unknown tool id(s): {sorted(unknown_tools)}. "
                    f"Available in selected environments: "
                    f"{sorted(available_tool_ids)}"
                )
            allowed_tool_ids = set(tool_ids)
            tools = [tool for tool in tools if tool.id in allowed_tool_ids]

        return tools

    def _coerce_environment(self, environment: Any) -> BaseEnvironment:
        """Coerce a raw dict or environment instance into a concrete environment."""
        from oumi.environments.base_environment import BaseEnvironment

        return BaseEnvironment.create(environment)
