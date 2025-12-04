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

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
    RemoteParams,
)
from oumi_chat.commands.config_utils import create_config_preserving_ui_settings
from oumi_chat.configs import ChatConfig, StyleParams


class TestConfigUtils:
    """Test suite for configuration utilities."""

    @pytest.fixture
    def base_inference_config(self):
        """Base inference configuration with model settings."""
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
    def current_chat_config(self):
        """Current ChatConfig with style and remote settings."""
        inference = InferenceConfig(
            model=ModelParams(model_name="old-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.NATIVE,
            remote_params=RemoteParams(api_key="test-key"),
        )
        style = StyleParams(use_emoji=True, assistant_title_style="bold cyan")
        return ChatConfig(inference=inference, style=style)

    def test_create_config_preserving_ui_settings(
        self, base_inference_config, current_chat_config
    ):
        """Test that style and remote settings are preserved."""
        result_config = create_config_preserving_ui_settings(
            base_inference_config, current_chat_config
        )

        # Model/generation/engine should come from base_inference_config
        assert result_config.inference.model.model_name == "new-model"
        assert result_config.inference.model.model_max_length == 8192
        assert result_config.inference.generation.max_new_tokens == 2048
        assert result_config.inference.generation.temperature == 0.8
        assert result_config.inference.engine == InferenceEngineType.VLLM

        # Style should come from current_chat_config
        assert result_config.style is current_chat_config.style
        assert result_config.style.use_emoji is True
        assert result_config.style.assistant_title_style == "bold cyan"

        # remote_params should come from current_chat_config.inference
        assert (
            result_config.inference.remote_params
            is current_chat_config.inference.remote_params
        )
        assert result_config.inference.remote_params is not None
        assert result_config.inference.remote_params.api_key == "test-key"

    def test_preserves_references_not_copies(
        self, base_inference_config, current_chat_config
    ):
        """Test that style is preserved as reference, not copy."""
        result_config = create_config_preserving_ui_settings(
            base_inference_config, current_chat_config
        )

        # Should be the same object reference
        assert result_config.style is current_chat_config.style

        # Changes to original should affect result
        current_chat_config.style.use_emoji = False
        assert result_config.style.use_emoji is False

    def test_handles_none_remote_params(self, base_inference_config):
        """Test handling when current config has None remote_params."""
        inference = InferenceConfig(
            model=ModelParams(model_name="old-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.NATIVE,
            remote_params=None,
        )
        current_config = ChatConfig(inference=inference, style=StyleParams())

        result_config = create_config_preserving_ui_settings(
            base_inference_config, current_config
        )

        # Model settings from base
        assert result_config.inference.model.model_name == "new-model"
        assert result_config.inference.engine == InferenceEngineType.VLLM

        # None settings preserved
        assert result_config.inference.remote_params is None
