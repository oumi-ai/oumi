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


def create_config_preserving_ui_settings(
    base_config: InferenceConfig, current_config: InferenceConfig
) -> InferenceConfig:
    """Create a new config preserving UI and remote settings from current context.

    Args:
        base_config: The base configuration (from YAML, branch state, etc.)
        current_config: The current context configuration to preserve settings from

    Returns:
        New InferenceConfig with model/generation/engine from base_config but
        style and remote_params preserved from current_config.
    """
    return InferenceConfig(
        model=base_config.model,
        generation=base_config.generation,
        engine=base_config.engine,
        style=current_config.style,  # Preserve UI settings
        remote_params=current_config.remote_params,  # Preserve remote settings
    )


def load_config_from_yaml_preserving_settings(
    yaml_path: str, current_config: InferenceConfig
) -> InferenceConfig:
    """Load config from YAML while preserving UI and remote settings.

    Args:
        yaml_path: Path to the YAML configuration file
        current_config: Current context configuration to preserve settings from

    Returns:
        New InferenceConfig with settings from YAML but UI/remote settings preserved.

    Raises:
        Exception: If YAML loading fails
    """
    base_config = InferenceConfig.from_yaml(yaml_path)
    return create_config_preserving_ui_settings(base_config, current_config)
