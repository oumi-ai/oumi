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

from oumi.core.configs.params.tool_params import ToolParams, ToolResult
from oumi.environments.deterministic_tool import DeterministicToolOutput
from oumi.environments.utils import describe_grounding_default


class BaseEnvironment(ABC):
    """Abstract base class for tool environments."""

    tool_params_cls: type[ToolParams] = ToolParams

    @abstractmethod
    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call within this environment."""

    def sample_grounding(
        self, n: int, *, rng: random.Random
    ) -> list[DeterministicToolOutput]:
        """Sample ``n`` grounding facts from this environment.

        Default: returns an empty list. Subclasses that support grounding
        (currently only ``DeterministicEnvironment``) override this.
        """
        return []

    def describe_grounding(self, facts: list[DeterministicToolOutput]) -> str:
        """Render grounding facts as a bulleted markdown block.

        Default implementation flattens each fact's input and output dicts
        (output wins on key collisions) into a single bullet line. Suitable
        for any dict-shaped fact. Subclasses may override for custom
        rendering.
        """
        return describe_grounding_default(facts)
