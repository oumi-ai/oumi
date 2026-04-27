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

"""Configuration for a single agentic environment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.tool_params import ToolParams


@dataclass
class EnvironmentParams(BaseParams):
    """Pure-data description of an environment."""

    id: str = ""
    name: str = ""
    description: str = ""
    env_type: str = ""
    tools: list[ToolParams] = field(default_factory=list)
    env_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Coerce raw tool dicts into ToolParams instances."""
        self.tools = [
            tool if isinstance(tool, ToolParams) else ToolParams.create(tool)
            for tool in self.tools
        ]

    def __finalize_and_validate__(self) -> None:
        """Validate common fields and registry membership."""
        if not self.id:
            raise ValueError(f"{type(self).__name__}.id cannot be empty.")
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name cannot be empty.")
        if not self.description:
            raise ValueError(f"{type(self).__name__}.description cannot be empty.")
        if not self.env_type:
            raise ValueError(f"{type(self).__name__}.env_type cannot be empty.")
        if self.env_kwargs is not None and not isinstance(self.env_kwargs, Mapping):
            raise ValueError(
                f"{type(self).__name__}.env_kwargs must be a mapping or None, "
                f"got {type(self.env_kwargs).__name__}."
            )

        self._validate_unique_tool_ids()
        self._validate_env_type_registered()

    def _validate_unique_tool_ids(self) -> None:
        seen: set[str] = set()
        for tool in self.tools:
            if tool.id in seen:
                raise ValueError(
                    f"{type(self).__name__} '{self.id}' contains duplicate "
                    f"tool id '{tool.id}'."
                )
            seen.add(tool.id)

    def _validate_env_type_registered(self) -> None:
        from oumi.core.registry import REGISTRY, RegistryType

        if not REGISTRY.contains(self.env_type, RegistryType.ENVIRONMENT):
            known = sorted(REGISTRY.get_all(RegistryType.ENVIRONMENT))
            raise ValueError(
                f"Unknown env_type '{self.env_type}'. Known types: {known}"
            )
