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

Importing this package populates the environment registry by triggering each
concrete environment's `@register_environment(...)` decorator.
"""

from oumi.core.configs.params.grounding_params import (
    GroundingConfig,
    GroundingFact,
)
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolError,
    ToolLookupError,
    ToolParams,
    ToolResult,
)
from oumi.core.types.tool_call import JSONSchema
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.deterministic_environment import (
    DeterministicEnvironment,
    DeterministicEnvironmentKwargs,
)
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)
from oumi.environments.synthetic_environment import (
    SyntheticEnvironment,
    SyntheticEnvironmentKwargs,
    SyntheticStateParams,
)

__all__ = [
    "BaseEnvironment",
    "DeterministicEnvironment",
    "DeterministicEnvironmentKwargs",
    "DeterministicTool",
    "DeterministicToolOutput",
    "GroundingConfig",
    "GroundingFact",
    "JSONSchema",
    "SyntheticEnvironment",
    "SyntheticEnvironmentKwargs",
    "SyntheticStateParams",
    "ToolArgumentError",
    "ToolError",
    "ToolLookupError",
    "ToolParams",
    "ToolResult",
]
