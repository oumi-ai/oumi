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

"""Environments for agentic tool interactions.

Environments are simulated worlds that agents interact with via tools.
Consumers include synthesis (training data generation), evaluation
(agent testing), and RL (reward-driven training).

Each environment type defines how tool calls are resolved:

- **StatefulEnvironment**: mutable JSON state across calls.
- **StatelessEnvironment**: LLM-generated outputs with optional caching.
- **DeterministicEnvironment**: fixed input-to-output lookup tables.
"""

from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.base_tool import BaseTool
from oumi.environments.deterministic_environment import (
    DeterministicEnvironment,
    DeterministicTool,
    DeterministicToolOutput,
)
from oumi.environments.stateful_environment import (
    StatefulEnvironment,
    StatefulTool,
)
from oumi.environments.stateless_environment import (
    GeneratedToolOutput,
    StatelessEnvironment,
    StatelessTool,
)
from oumi.environments.types import ToolEnvironmentType

__all__ = [
    "BaseEnvironment",
    "BaseTool",
    "DeterministicEnvironment",
    "DeterministicTool",
    "DeterministicToolOutput",
    "GeneratedToolOutput",
    "StatefulEnvironment",
    "StatefulTool",
    "StatelessEnvironment",
    "StatelessTool",
    "ToolEnvironmentType",
]
