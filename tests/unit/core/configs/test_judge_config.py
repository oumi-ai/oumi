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

from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.judge_config_v2 import (
    JudgeConfig,
    JudgeOutputType,
    JudgeResponseFormat,
)
from oumi.core.registry import REGISTRY, RegistryType, register_prompt_template


class TestJudgeConfigPromptTemplates:
    """Test suite for JudgeConfig prompt template registry integration."""

    def test_registry_prompt_template_resolution(self):
        """Test that prompt templates are correctly resolved from registry."""
        yaml_config = """
            prompt_template: unit_test
            response_format: XML
            judgment_type: BOOL
            include_explanation: false

            model:
                model_name: "gpt-4.1-mini-2025-04-14"

            engine: OPENAI

            generation:
                max_new_tokens: 8192
                temperature: 0.0
        """

        config = JudgeConfig.from_str(yaml_config)

        # Should resolve to the already registered template
        expected_template = "Unit test prompt template for Oumi Judge."
        assert config.prompt_template == expected_template

        # Verify other config fields are correctly parsed
        assert config.response_format == JudgeResponseFormat.XML
        assert config.judgment_type == JudgeOutputType.BOOL
        assert config.include_explanation is False
        assert config.engine == InferenceEngineType.OPENAI
        assert config.model.model_name == "gpt-4.1-mini-2025-04-14"
        assert config.generation.max_new_tokens == 8192
        assert config.generation.temperature == 0.0

    def test_registry_prompt_template_resolution_dynamic_registry(self):
        """Test that prompt templates are correctly resolved from registry."""
        prompt_template_name = "unit_test_2"
        prompt_template_value = "Second unit test prompt template for Oumi Judge."

        # Verify template doesn't exist before registration
        assert not REGISTRY.contains(prompt_template_name, RegistryType.PROMPT_TEMPLATE)

        # Register template dynamically
        register_prompt_template(prompt_template_name)(prompt_template_value)

        # Verify template is now registered
        assert REGISTRY.contains(prompt_template_name, RegistryType.PROMPT_TEMPLATE)
        assert REGISTRY.get_prompt_template(prompt_template_name) == (
            prompt_template_value
        )

        yaml_config = """
            prompt_template: unit_test_2
            response_format: JSON
            judgment_type: ENUM
            include_explanation: true
            judgment_scores:
                "excellent": 1.0
                "good": 0.7
                "poor": 0.3

            model:
                model_name: "gpt-4.1-mini-2025-04-14"

            engine: OPENAI

            generation:
                max_new_tokens: 8192
                temperature: 0.0
        """

        config = JudgeConfig.from_str(yaml_config)

        # Should resolve to the template we registered in this test
        assert config.prompt_template == prompt_template_value

        # Verify different config values are parsed correctly
        assert config.response_format == JudgeResponseFormat.JSON
        assert config.judgment_type == JudgeOutputType.ENUM
        assert config.include_explanation is True
        assert config.judgment_scores == {"excellent": 1.0, "good": 0.7, "poor": 0.3}

    def test_custom_string_template_unchanged(self):
        """Test that custom (unregistered) string templates are not modified."""
        custom_template = "This should not be changed."
        yaml_config = f"""
            prompt_template: "{custom_template}"
            response_format: XML
            judgment_type: BOOL
            include_explanation: false

            model:
                model_name: "gpt-4.1-mini-2025-04-14"

            engine: OPENAI

            generation:
                max_new_tokens: 8192
                temperature: 0.0
        """

        config = JudgeConfig.from_str(yaml_config)

        # Should remain unchanged.
        assert config.prompt_template == custom_template

        # Verify different config values are parsed correctly
        assert config.response_format == JudgeResponseFormat.XML
        assert config.judgment_type == JudgeOutputType.BOOL
        assert config.include_explanation is False
        assert config.judgment_scores is None
