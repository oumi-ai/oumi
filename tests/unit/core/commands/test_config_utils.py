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

"""Tests for configuration utilities."""

from unittest.mock import MagicMock

import pytest

from oumi.core.commands.config_utils import create_config_preserving_ui_settings
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)


class TestConfigUtils:
    """Test suite for configuration utilities."""

    @pytest.fixture
    def base_config(self):
        """Base configuration with model settings."""
        return InferenceConfig(
            model=ModelParams(
                model_name="new-model",
                model_max_length=8192,
                torch_dtype_str="bfloat16",
            ),
            generation=GenerationParams(max_new_tokens=2048, temperature=0.8),
            engine=InferenceEngineType.VLLM,
        )

    @pytest.fixture
    def current_config(self):
        """Current configuration with UI and remote settings."""
        config = InferenceConfig(
            model=ModelParams(model_name="old-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.NATIVE,
        )
        # Mock the UI and remote settings
        config.style = MagicMock()
        config.style.use_emoji = True
        config.style.theme_name = "dark"

        config.remote_params = MagicMock()
        config.remote_params.api_key = "test-key"
        config.remote_params.base_url = "https://api.example.com"

        return config

    def test_create_config_preserving_ui_settings(self, base_config, current_config):
        """Test that UI and remote settings are preserved while updating model
        settings."""
        result_config = create_config_preserving_ui_settings(
            base_config, current_config
        )

        # Model/generation/engine should come from base_config
        assert result_config.model.model_name == "new-model"
        assert result_config.model.model_max_length == 8192
        assert result_config.generation.max_new_tokens == 2048
        assert result_config.generation.temperature == 0.8
        assert result_config.engine == InferenceEngineType.VLLM

        # Style and remote_params should come from current_config
        assert result_config.style is current_config.style
        assert result_config.remote_params is current_config.remote_params
        assert result_config.style.use_emoji is True
        assert result_config.style.theme_name == "dark"
        assert result_config.remote_params.api_key == "test-key"
        assert result_config.remote_params.base_url == "https://api.example.com"

    def test_preserves_references_not_copies(self, base_config, current_config):
        """Test that style and remote_params are preserved as references, not copies."""
        result_config = create_config_preserving_ui_settings(
            base_config, current_config
        )

        # Should be the same object references
        assert result_config.style is current_config.style
        assert result_config.remote_params is current_config.remote_params

        # Changes to original should affect result
        current_config.style.use_emoji = False
        assert result_config.style.use_emoji is False

    def test_handles_none_settings(self, base_config):
        """Test handling when current config has None style or remote_params."""
        current_config = InferenceConfig(
            model=ModelParams(model_name="old-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.NATIVE,
        )
        current_config.style = None
        current_config.remote_params = None

        result_config = create_config_preserving_ui_settings(
            base_config, current_config
        )

        # Model settings from base
        assert result_config.model.model_name == "new-model"
        assert result_config.engine == InferenceEngineType.VLLM

        # None settings preserved
        assert result_config.style is None
        assert result_config.remote_params is None

