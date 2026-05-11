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
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.utils import describe_grounding_default


class BaseEnvironment(ABC):
    """Abstract base class for tool environments."""

    tool_params_cls: type[ToolParams] = ToolParams

    @abstractmethod
    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute a batch of tool calls; returns results in the same order."""

    def sample_grounding(
        self,
        n: int,
        *,
        rng: random.Random,
        tool_ids: set[str] | None = None,
    ) -> list[GroundingFact]:
        """Sample grounding facts from this environment. Default: ``[]``."""
        return []

    def describe_grounding(self, facts: list[GroundingFact]) -> str:
        """Render grounding facts as a bulleted markdown block."""
        return describe_grounding_default(facts)
