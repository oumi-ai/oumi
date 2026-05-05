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
class GroundingConfig(BaseParams):
    """Per-environment grounding configuration.

    When set on an environment, the ConversationSynthesizer samples facts from
    that environment and injects them into the planner prompt so turn plans
    reference real entities rather than hallucinated IDs.
    """

    sample_size: int = 3
    """Number of grounding facts sampled per conversation."""

    seed: int | None = None
    """Optional seed for reproducible grounding sampling."""

    def __post_init__(self) -> None:
        """Validate ``sample_size`` is positive."""
        if self.sample_size < 1:
            raise ValueError(
                f"{type(self).__name__}.sample_size must be >= 1, "
                f"got {self.sample_size}."
            )


@dataclass
class GroundingFact(BaseParams):
    """Env-agnostic representation of a single grounding fact.

    Environments produce these during sampling; the planner prompt renders
    each fact's ``data`` dict as one bullet line. Values are expected to be
    JSON-serializable scalars.
    """

    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolGroundingConfig(BaseParams):
    """Per-tool grounding configuration.

    Tools that declare this block contribute facts to the environment's
    grounding pool, projected to ``fields``. Tools without it contribute
    nothing.
    """

    key: str
    """Name of the entity primary-key field. Must appear in ``fields``."""

    fields: list[str]
    """Whitelisted field names projected into each ``GroundingFact.data`` dict."""

    def __post_init__(self) -> None:
        """Validate ``key`` and ``fields`` invariants."""
        if not self.key:
            raise ValueError(f"{type(self).__name__}.key cannot be empty.")
        if not self.fields:
            raise ValueError(f"{type(self).__name__}.fields must be non-empty.")
        if self.key not in self.fields:
            raise ValueError(
                f"{type(self).__name__}.fields must include 'key' "
                f"({self.key!r}); got {self.fields!r}."
            )
        if len(set(self.fields)) != len(self.fields):
            raise ValueError(
                f"{type(self).__name__}.fields contains duplicate entries: "
                f"{self.fields!r}."
            )
