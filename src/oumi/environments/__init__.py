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
    StateGroundingConfig,
    ToolGroundingConfig,
)
from oumi.core.configs.params.tool_params import (
    ToolArgumentError,
    ToolError,
    ToolLookupError,
    ToolParams,
)
from oumi.core.types.tool_call import JSONSchema, ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.database_executable_environment import (
    DatabaseExecutableEnvironment,
)
from oumi.environments.endpoint_environment import (
    EndpointAuthParams,
    EndpointAuthType,
    EndpointCallError,
    EndpointEnvironment,
    EndpointEnvironmentKwargs,
    EndpointProtocol,
    EndpointTransport,
)
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.lookup_environment import (
    DeterministicEnvironment,
    DeterministicEnvironmentKwargs,
    LookupEnvironment,
    LookupEnvironmentKwargs,
    ToolLookupEntry,
)
from oumi.environments.simulated_environment import (
    SimulatedEnvironment,
    SimulatedEnvironmentKwargs,
    SimulatedStateParams,
    SyntheticEnvironment,
    SyntheticEnvironmentKwargs,
    SyntheticStateParams,
)

__all__ = [
    "BaseEnvironment",
    "DatabaseExecutableEnvironment",
    # Deprecated aliases, kept so existing imports keep resolving.
    "DeterministicEnvironment",
    "DeterministicEnvironmentKwargs",
    "EndpointAuthParams",
    "EndpointAuthType",
    "EndpointCallError",
    "EndpointEnvironment",
    "EndpointEnvironmentKwargs",
    "EndpointProtocol",
    "EndpointTransport",
    "ExecutableEnvironment",
    "ExecutableTool",
    "GroundingConfig",
    "GroundingFact",
    "JSONSchema",
    "LookupEnvironment",
    "LookupEnvironmentKwargs",
    "SimulatedEnvironment",
    "SimulatedEnvironmentKwargs",
    "SimulatedStateParams",
    "StateGroundingConfig",
    "SyntheticEnvironment",
    "SyntheticEnvironmentKwargs",
    "SyntheticStateParams",
    "ToolArgumentError",
    "ToolError",
    "ToolGroundingConfig",
    "ToolLookupEntry",
    "ToolLookupError",
    "ToolParams",
    "ToolResult",
]
