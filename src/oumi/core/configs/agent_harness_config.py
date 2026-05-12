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

"""Top-level config for the ``oumi agent`` harness."""

from __future__ import annotations

from dataclasses import dataclass, field

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig


@dataclass
class AgentHarnessConfig(BaseConfig):
    """Top-level config consumed by ``oumi agent <config>.yaml``."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    """Model, generation, engine, and remote-API parameters for the agent LLM."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    """Environments wired into the session; each owns its tools."""

    system_prompt: str = ""
    """Optional system message prepended to the chat conversation."""

    def __finalize_and_validate__(self) -> None:
        """Validate that at least one environment is configured with tools."""
        if not self.environment.environments:
            raise ValueError(
                "AgentHarnessConfig.environment.environments must contain at "
                "least one environment."
            )
        if not self.environment.all_tools:
            raise ValueError(
                "AgentHarnessConfig.environment exposes zero tools across "
                f"{len(self.environment.environments)} environment(s)."
            )
