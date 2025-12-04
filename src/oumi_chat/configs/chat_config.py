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
from typing import Optional

from oumi.core.configs import InferenceConfig
from oumi.core.configs.base_config import BaseConfig
from oumi_chat.configs.style_params import StyleParams


@dataclass
class ChatConfig(BaseConfig):
    """Configuration for Oumi Chat interactive sessions.

    This configuration combines the inference configuration with chat-specific
    styling parameters for the Rich console UI.

    Example YAML configuration:
        ```yaml
        inference:
          model:
            model_name: "meta-llama/Llama-3.2-3B-Instruct"
            model_max_length: 8192
          generation:
            max_new_tokens: 2048
            temperature: 0.7
          engine: VLLM

        style:
          use_emoji: true
          assistant_title_style: "bold cyan"
          user_prompt_style: "bold blue"
        ```
    """

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    """Configuration for the inference engine and model parameters."""

    style: StyleParams = field(default_factory=StyleParams)
    """Parameters for customizing Rich console styling."""

    @classmethod
    def from_inference_config(
        cls, inference_config: InferenceConfig, style: Optional[StyleParams] = None
    ) -> "ChatConfig":
        """Create a ChatConfig from an existing InferenceConfig.

        This is useful for converting a standalone InferenceConfig to a ChatConfig
        when using the chat functionality.

        Args:
            inference_config: The inference configuration to use.
            style: Optional style parameters. Defaults to StyleParams().

        Returns:
            A new ChatConfig instance.
        """
        return cls(
            inference=inference_config,
            style=style or StyleParams(),
        )
