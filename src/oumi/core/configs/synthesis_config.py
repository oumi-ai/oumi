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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.configs.params.tool_params import ToolParams
from oumi.exceptions import OumiConfigError


class SynthesisStrategy(str, Enum):
    """The supported synthesis strategies."""

    GENERAL = "general"
    """A general synthesis strategy that can be used for any task."""


@dataclass
class SynthesisConfig(BaseConfig):
    """The configuration for the synthesis pipeline."""

    output_path: str | None = None
    """The path to the output file where the generated data will be saved.

    If not specified, the data will be returned as a list of dictionaries.
    """

    strategy: SynthesisStrategy = SynthesisStrategy.GENERAL
    """The synthesis strategy to use."""

    strategy_params: GeneralSynthesisParams = field(
        default_factory=GeneralSynthesisParams
    )
    """The synthesis strategy parameters to use."""

    environment_config: EnvironmentConfig | None = None
    """Reusable environment-first tool configuration."""

    environment_config_path: str | None = None
    """Optional path to an EnvironmentConfig YAML file."""

    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    """The inference configuration to use."""

    num_samples: int = 1
    """The number of synthetic samples to generate."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.strategy == SynthesisStrategy.GENERAL:
            pass
        else:
            raise OumiConfigError(f"Unsupported synthesis strategy: {self.strategy}")

        if self.inference_config.input_path is not None:
            raise OumiConfigError(
                "Input path is not supported for general synthesis strategy."
            )

        if self.inference_config.output_path is not None:
            raise OumiConfigError(
                "Output path is not supported for general synthesis strategy."
            )

        if self.output_path is not None:
            if self.output_path == "":
                raise OumiConfigError("Output path cannot be empty.")

            if not self.output_path.endswith(".jsonl"):
                raise OumiConfigError("Output path must end with .jsonl.")

        self.environment_config = self._resolve_environment_config()
        self._validate_available_tooling()

    def _resolve_environment_config(self) -> EnvironmentConfig | None:
        """Resolve top-level environment configuration."""
        if (
            self.environment_config is not None
            and self.environment_config_path is not None
        ):
            raise OumiConfigError(
                "SynthesisConfig.environment_config and "
                "SynthesisConfig.environment_config_path cannot both be set."
            )

        if self.environment_config is not None:
            return self.environment_config

        if self.environment_config_path is not None:
            if self.environment_config_path == "":
                raise OumiConfigError(
                    "SynthesisConfig.environment_config_path cannot be empty."
                )

            config_path = Path(self.environment_config_path)
            if not config_path.exists():
                raise OumiConfigError(
                    f"Environment config path does not exist: "
                    f"{self.environment_config_path}"
                )
            return EnvironmentConfig.from_yaml(config_path)

        return None

    def resolve_multiturn_environments(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> list[EnvironmentParams]:
        """Resolve the environments available to a multiturn attribute."""
        if self.environment_config is None:
            return []

        if not multiturn_attribute.available_environments:
            return list(self.environment_config.environments)

        resolved_environments: list[EnvironmentParams] = []
        for environment_id in multiturn_attribute.available_environments:
            environment = self.environment_config.get_environment(environment_id)
            if environment is None:
                raise OumiConfigError(
                    f"MultiTurnAttribute '{multiturn_attribute.id}' references unknown "
                    f"environment '{environment_id}'. Defined environment ids: "
                    f"{sorted(env.id for env in self.environment_config.environments)}"
                )
            resolved_environments.append(environment)
        return resolved_environments

    def resolve_multiturn_tools(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> list[ToolParams]:
        """Resolve the tools available to a multiturn attribute."""
        if self.environment_config is None:
            return []

        environments = self.resolve_multiturn_environments(multiturn_attribute)
        return self.environment_config.resolve_tools(
            environment_ids=[environment.id for environment in environments],
            tool_ids=multiturn_attribute.available_tools or None,
        )

    def _validate_available_tooling(self) -> None:
        """Validate multiturn environment/tool selections against the catalog."""
        if not self.strategy_params.multiturn_attributes:
            return

        all_referenced_tools = [
            tool_id
            for mt_attr in self.strategy_params.multiturn_attributes
            for tool_id in mt_attr.available_tools
        ]
        all_referenced_environments = [
            environment_id
            for mt_attr in self.strategy_params.multiturn_attributes
            for environment_id in mt_attr.available_environments
        ]
        if not all_referenced_tools and not all_referenced_environments:
            return

        if self.environment_config is None:
            raise OumiConfigError(
                "Environment or tool references require "
                "SynthesisConfig.environment_config, or "
                "SynthesisConfig.environment_config_path."
            )

        for mt_attr in self.strategy_params.multiturn_attributes:
            selected_environments = self.resolve_multiturn_environments(mt_attr)
            selected_environment_ids = {
                environment.id for environment in selected_environments
            }
            selected_tools = self.environment_config.resolve_tools(
                environment_ids=list(selected_environment_ids)
            )
            selected_tool_ids = {tool.id for tool in selected_tools}

            for tool_id in mt_attr.available_tools:
                if tool_id not in selected_tool_ids:
                    raise OumiConfigError(
                        f"MultiTurnAttribute '{mt_attr.id}' references unknown "
                        f"tool '{tool_id}' for environments "
                        f"{sorted(selected_environment_ids)}. Defined tool ids: "
                        f"{sorted(selected_tool_ids)}"
                    )
