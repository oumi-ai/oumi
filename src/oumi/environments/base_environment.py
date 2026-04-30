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

"""Abstract base class for tool environments."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.configs.params.tool_params import ToolParams, ToolResult
from oumi.environments.utils import describe_grounding_default


class BaseEnvironment(ABC):
    """Abstract base class for tool environments."""

    tool_params_cls: type[ToolParams] = ToolParams

    @abstractmethod
    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call within this environment."""

    def sample_grounding(
        self,
        n: int,
        *,
        rng: random.Random,
        tool_ids: set[str] | None = None,
    ) -> list[GroundingFact]:
        """Sample grounding facts from this environment.

        Args:
            n: Maximum number of facts to return. Implementations may return fewer
                if the pool is smaller than ``n``.
            rng: RNG used for sampling. Callers are expected to seed it
                deterministically when reproducibility is required.
            tool_ids: When non-None, restrict sampling to facts produced by the
                named tools. ``None`` means all grounded tools in this environment.

        Returns:
            Up to ``n`` ``GroundingFact`` instances.
        """
        return []

    def describe_grounding(self, facts: list[GroundingFact]) -> str:
        """Render grounding facts as a bulleted markdown block.

        Default implementation renders each fact's ``data`` dict as a
        single bullet line. Subclasses may override for custom rendering.
        """
        return describe_grounding_default(facts)
