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

"""Utility functions for configuration management in commands."""

from oumi.core.configs import InferenceConfig
from oumi_chat.configs import ChatConfig


def create_config_preserving_ui_settings(
    base_config: InferenceConfig, current_config: ChatConfig
) -> ChatConfig:
    """Create a new ChatConfig preserving UI settings from current context.

    Args:
        base_config: The base inference configuration (from YAML, branch state, etc.)
        current_config: The current ChatConfig to preserve style settings from.

    Returns:
        New ChatConfig with inference settings from base_config and
        style preserved from current_config.
    """
    # Create new inference config with remote_params from current
    new_inference = InferenceConfig(
        model=base_config.model,
        generation=base_config.generation,
        engine=base_config.engine,
        remote_params=current_config.inference.remote_params,
    )

    return ChatConfig(
        inference=new_inference,
        style=current_config.style,
    )


def load_config_from_yaml_preserving_settings(
    yaml_path: str, current_config: ChatConfig
) -> ChatConfig:
    """Load config from YAML while preserving UI settings.

    Args:
        yaml_path: Path to the YAML configuration file
        current_config: Current ChatConfig to preserve style settings from

    Returns:
        New ChatConfig with inference settings from YAML but style preserved.

    Raises:
        Exception: If YAML loading fails
    """
    base_config = InferenceConfig.from_yaml(yaml_path)
    return create_config_preserving_ui_settings(base_config, current_config)
