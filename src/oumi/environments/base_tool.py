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

"""Base tool class shared by all environment types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class BaseTool(BaseParams):
    """Common fields for all tools exposed by an environment."""

    id: str
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, raw: Mapping[str, Any] | BaseTool) -> BaseTool:
        """Create a tool from raw config data.

        Returns a ``BaseTool`` with only the common fields. Environment
        subclasses call their own typed factory (e.g.
        ``DeterministicTool.create``) to get the full subclass.
        """
        if isinstance(raw, BaseTool):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"Tool definitions must be tool objects or mappings, got {type(raw)}"
            )
        return cls(
            id=raw["id"],
            name=raw["name"],
            description=raw["description"],
            parameters=raw.get("parameters", {}),
        )

    def __post_init__(self):
        """Validate common tool fields."""
        if not self.id:
            raise ValueError(f"{type(self).__name__}.id cannot be empty.")
        if not self.name:
            raise ValueError(f"{type(self).__name__}.name cannot be empty.")
        if not self.description:
            raise ValueError(f"{type(self).__name__}.description cannot be empty.")
