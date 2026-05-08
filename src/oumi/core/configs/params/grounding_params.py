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

"""Per-environment grounding configuration and fact types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class ToolGroundingConfig(BaseParams):
    """Per-tool field whitelist for grounding projection.

    Tools listed in ``GroundingConfig.tools`` are projected through
    ``fields`` when the env samples grounding facts. Tools not listed
    contribute nothing.
    """

    fields: list[str]

    def __post_init__(self) -> None:
        """Validate ``fields`` invariants."""
        if not self.fields:
            raise ValueError(f"{type(self).__name__}.fields must be non-empty.")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(
                f"{type(self).__name__}.fields contains duplicate entries: "
                f"{self.fields!r}."
            )


@dataclass
class GroundingConfig(BaseParams):
    """Per-environment grounding configuration."""

    sample_size: int = 3
    """Number of grounding facts sampled per conversation."""

    seed: int | None = None
    """Optional seed for reproducible grounding sampling."""

    tools: dict[str, ToolGroundingConfig] = field(default_factory=dict)
    """Per-tool field whitelists, keyed by tool id."""

    def __post_init__(self) -> None:
        """Validate ``sample_size`` and coerce ``tools`` entries."""
        if self.sample_size < 1:
            raise ValueError(
                f"{type(self).__name__}.sample_size must be >= 1, "
                f"got {self.sample_size}."
            )
        self.tools = {
            tool_id: cfg
            if isinstance(cfg, ToolGroundingConfig)
            else ToolGroundingConfig(**cfg)
            for tool_id, cfg in self.tools.items()
        }


@dataclass
class GroundingFact(BaseParams):
    """Env-agnostic representation of a single grounding fact.

    Environments produce these during sampling; the planner prompt renders
    each fact's ``data`` dict as one bullet line. Values are expected to be
    JSON-serializable scalars.
    """

    data: dict[str, Any] = field(default_factory=dict)
