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

from dataclasses import dataclass, field
from enum import Enum

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.synthesis_params import GeneralSynthesisParams


class SynthesisStrategy(str, Enum):
    """The supported synthesis strategies."""

    GENERAL = "general"
    """A general synthesis strategy that can be used for any task."""


@dataclass
class SynthesisConfig(BaseConfig):
    strategy: SynthesisStrategy = SynthesisStrategy.GENERAL
    """The synthesis strategy to use."""

    strategy_params: GeneralSynthesisParams = field(
        default_factory=GeneralSynthesisParams
    )
    """The synthesis strategy parameters to use."""

    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    """The inference configuration to use."""

    num_samples: int = 1
    """The number of synthetic samples to generate."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.strategy == SynthesisStrategy.GENERAL:
            pass
        else:
            raise ValueError(f"Unsupported synthesis strategy: {self.strategy}")

        if self.inference_config.input_path is not None:
            raise ValueError(
                "Input path is not supported for general synthesis strategy."
            )

        if self.inference_config.output_path is not None:
            raise ValueError(
                "Output path is not supported for general synthesis strategy."
            )
