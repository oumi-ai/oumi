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

"""Tests for model swapping edge cases and HuggingFace model handling."""

from unittest.mock import MagicMock, patch

import pytest

from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.commands.handlers.model_management_handler import ModelManagementHandler
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)


class TestModelSwapEdgeCases:
    """Test suite for model swapping edge cases."""

    @pytest.fixture
    def mock_base_config(self):
        """Base mock configuration."""
        model_config = ModelParams(
            model_name="microsoft/Phi-3.5-mini-instruct",
            model_max_length=4096,
            torch_dtype_str="bfloat16",
        )
        generation_config = GenerationParams(max_new_tokens=1024, temperature=0.7)
        return InferenceConfig(
            model=model_config,
            generation=generation_config,
            engine=InferenceEngineType.NATIVE,
        )

    @pytest.fixture
    def mock_context(self, mock_base_config):
        """Mock command context."""
        context = MagicMock()
        context.config = mock_base_config
        context.inference_engine = MagicMock()
        context.system_monitor = MagicMock()
        context.system_monitor.update_max_context_tokens = MagicMock()
        context.system_monitor._last_update_time = 1000
        context._context_window_manager = None
        return context

    @pytest.fixture
    def mock_branch_manager(self):
        """Mock branch manager."""
        branch_manager = MagicMock()
        current_branch = MagicMock()
        current_branch.id = "main"
        current_branch.model_name = None
        current_branch.engine_type = None
        current_branch.model_config = None
        current_branch.generation_config = None
        branch_manager.get_current_branch.return_value = current_branch
        return branch_manager, current_branch

    @pytest.fixture
    def model_handler(self, mock_context, mock_branch_manager):
        """Model management handler with mocked dependencies."""
        # Mock the context attributes that the handler expects
        mock_context.console = MagicMock()
        mock_context.config = mock_context.config
        mock_context.conversation_history = []
        mock_context.inference_engine = mock_context.inference_engine
        mock_context.system_monitor = mock_context.system_monitor
        mock_context._style = MagicMock()

        handler = ModelManagementHandler(mock_context)
        handler._update_context_in_monitor = MagicMock()

        branch_manager, current_branch = mock_branch_manager
        handler.context.branch_manager = branch_manager

        return handler, current_branch

    def test_swap_with_huggingface_model_id_fails(self, model_handler):
        """Test that swapping with a raw HuggingFace model ID fails gracefully."""
        handler, current_branch = model_handler

        test_cases = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "microsoft/Phi-3.5-mini-instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "google/gemma-2-9b-it",
            "microsoft/DialoGPT-large",
        ]

        for model_id in test_cases:
            command = ParsedCommand(
                "swap", args=[model_id], kwargs={}, raw_input=f"/swap({model_id})"
            )
            result = handler._handle_swap(command)

            assert not result.success, f"Should fail for HF model ID: {model_id}"
            # The current implementation treats HF model IDs as config files due to "/"
            # and fails with "Config file not found", which is the expected behavior
            assert (
                "Config file not found" in result.message
                or "Invalid swap target" in result.message
            ), f"Got unexpected message: {result.message}"

    def test_swap_with_config_prefix_succeeds(self, model_handler):
        """Test that swapping with config: prefix works correctly."""
        handler, current_branch = model_handler

        mock_config = InferenceConfig(
            model=ModelParams(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                model_max_length=8192,
                torch_dtype_str="bfloat16",
            ),
            generation=GenerationParams(max_new_tokens=2048),
            engine=InferenceEngineType.VLLM,
        )

        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml", return_value=mock_config
            ),
            patch("oumi.infer.get_engine") as mock_get_engine,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            command = ParsedCommand(
                "swap",
                args=["config:recipes/llama/8b_instruct.yaml"],
                kwargs={},
                raw_input="/swap(config:recipes/llama/8b_instruct.yaml)",
            )
            result = handler._handle_swap(command)

            assert result.success
            assert "Swapped to meta-llama/Llama-3.1-8B-Instruct" in result.message
            assert "VLLM engine" in result.message

    def test_swap_with_yaml_extension_auto_detection(self, model_handler):
        """Test that .yaml/.yml files are auto-detected as config files."""
        handler, current_branch = model_handler

        mock_config = InferenceConfig(
            model=ModelParams(
                model_name="Qwen/Qwen2.5-7B-Instruct", model_max_length=32768
            ),
            generation=GenerationParams(temperature=0.6),
            engine=InferenceEngineType.NATIVE,
        )

        test_paths = [
            "model_config.yaml",
            "inference_config.yml",
            "configs/qwen/7b.yaml",
            "/absolute/path/config.yml",
        ]

        for config_path in test_paths:
            with (
                patch(
                    "oumi.core.configs.InferenceConfig.from_yaml",
                    return_value=mock_config,
                ),
                patch("oumi.infer.get_engine") as mock_get_engine,
                patch("pathlib.Path.exists", return_value=True),
            ):
                mock_engine = MagicMock()
                mock_get_engine.return_value = mock_engine

                command = ParsedCommand(
                    "swap",
                    args=[config_path],
                    kwargs={},
                    raw_input=f"/swap({config_path})",
                )
                result = handler._handle_swap(command)

                assert result.success, f"Should succeed for config path: {config_path}"
                assert "Qwen/Qwen2.5-7B-Instruct" in result.message

    def test_swap_with_path_separators_auto_detection(self, model_handler):
        """Test that paths with / or \\ are auto-detected as config files."""
        handler, current_branch = model_handler

        mock_config = InferenceConfig(
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.NATIVE,
        )

        test_paths = [
            "configs/model",
            "path/to/config",
            "configs\\windows\\path",
            "/absolute/path/config",
            "relative/deep/path/config",
        ]

        for config_path in test_paths:
            with (
                patch(
                    "oumi.core.configs.InferenceConfig.from_yaml",
                    return_value=mock_config,
                ),
                patch("oumi.infer.get_engine") as mock_get_engine,
                patch("pathlib.Path.exists", return_value=True),
            ):
                mock_engine = MagicMock()
                mock_get_engine.return_value = mock_engine

                command = ParsedCommand(
                    "swap",
                    args=[config_path],
                    kwargs={},
                    raw_input=f"/swap({config_path})",
                )
                result = handler._handle_swap(command)

                assert result.success, f"Should succeed for path: {config_path}"

    def test_swap_saves_current_model_state(self, model_handler):
        """Test that swapping saves the current model state to the active branch."""
        handler, current_branch = model_handler

        # Set initial model state
        handler.context.config.model.model_name = "initial-model"
        handler.context.config.engine = InferenceEngineType.NATIVE

        mock_new_config = InferenceConfig(
            model=ModelParams(model_name="new-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.VLLM,
        )

        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml",
                return_value=mock_new_config,
            ),
            patch("oumi.infer.get_engine") as mock_get_engine,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            result = handler._handle_config_swap("test.yaml")

            assert result.success

            # Verify current branch was updated with previous model state
            assert current_branch.model_name == "initial-model"
            assert current_branch.engine_type == "NATIVE"
            assert current_branch.model_config is not None
            assert current_branch.generation_config is not None

    def test_swap_updates_system_monitor_context_length(self, model_handler):
        """Test that swapping updates system monitor with correct context length."""
        handler, current_branch = model_handler

        # Test different engine types and their context calculations
        test_cases = [
            # (engine, model_name, model_max_length, expected_context)
            (InferenceEngineType.NATIVE, "test-model", 8192, 8192),
            (InferenceEngineType.VLLM, "large-model", 32768, 32768),
            (InferenceEngineType.LLAMACPP, "gguf-model", 16384, 16384),
            (InferenceEngineType.ANTHROPIC, "claude-3-5-sonnet-20241022", None, 200000),
            (InferenceEngineType.OPENAI, "gpt-4o", None, 128000),
        ]

        for engine, model_name, max_length, expected_context in test_cases:
            mock_config = InferenceConfig(
                model=ModelParams(model_name=model_name, model_max_length=max_length),
                generation=GenerationParams(),
                engine=engine,
            )

            with (
                patch(
                    "oumi.core.configs.InferenceConfig.from_yaml",
                    return_value=mock_config,
                ),
                patch("oumi.infer.get_engine") as mock_get_engine,
                patch("pathlib.Path.exists", return_value=True),
            ):
                mock_engine = MagicMock()
                mock_get_engine.return_value = mock_engine

                # Reset monitor mock
                handler.context.system_monitor.update_max_context_tokens.reset_mock()
                handler.context.system_monitor._last_update_time = 1000

                result = handler._handle_config_swap("test.yaml")

                assert result.success

                # Verify system monitor was updated with correct context length
                handler.context.system_monitor.update_max_context_tokens.assert_called_once_with(
                    expected_context
                )
                # Verify force refresh was triggered
                assert handler.context.system_monitor._last_update_time == 0

    def test_swap_handles_config_loading_errors(self, model_handler):
        """Test that config loading errors are handled gracefully."""
        handler, current_branch = model_handler

        test_cases = [
            # (exception, expected_message_part)
            (FileNotFoundError("Config not found"), "Error loading config:"),
            (ValueError("Invalid YAML"), "Error loading config: Invalid YAML"),
            (Exception("Generic error"), "Error loading config: Generic error"),
        ]

        for exception, expected_msg_part in test_cases:
            with (
                patch(
                    "oumi.core.configs.InferenceConfig.from_yaml", side_effect=exception
                ),
                patch("pathlib.Path.exists", return_value=True),
            ):
                result = handler._handle_config_swap("test.yaml")

                assert not result.success
                assert expected_msg_part in result.message

    def test_swap_handles_engine_creation_errors(self, model_handler):
        """Test that engine creation errors are handled gracefully."""
        handler, current_branch = model_handler

        mock_config = InferenceConfig(
            model=ModelParams(model_name="problematic-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.VLLM,
        )

        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml", return_value=mock_config
            ),
            patch(
                "oumi.infer.get_engine", side_effect=Exception("Engine creation failed")
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = handler._handle_config_swap("test.yaml")

            assert not result.success
            assert (
                "Error creating inference engine: Engine creation failed"
                in result.message
            )

    def test_swap_with_empty_target_fails(self, model_handler):
        """Test that empty swap targets fail gracefully."""
        handler, current_branch = model_handler

        test_cases = [
            ("", "requires a model name or config path argument"),
            ("   ", "requires a model name or config path argument"),  # whitespace only
            (
                "config:",
                "config: prefix requires a path to a configuration file",
            ),  # config prefix with no path
            (
                "   config:   ",
                "config: prefix requires a path to a configuration file",
            ),  # config prefix with whitespace
        ]

        for target, expected_msg in test_cases:
            command = ParsedCommand(
                "swap", args=[target], kwargs={}, raw_input=f"/swap({target})"
            )
            result = handler._handle_swap(command)

            assert not result.success
            assert expected_msg in result.message

    def test_swap_preserves_ui_and_remote_settings(self, model_handler):
        """Test that swapping preserves UI style and remote parameter settings."""
        handler, current_branch = model_handler

        # Set up original config with UI and remote settings
        original_style = MagicMock()
        original_remote = MagicMock()
        handler.context.config.style = original_style
        handler.context.config.remote_params = original_remote

        mock_new_config = InferenceConfig(
            model=ModelParams(model_name="new-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.VLLM,
        )

        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml",
                return_value=mock_new_config,
            ),
            patch("oumi.infer.get_engine") as mock_get_engine,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            result = handler._handle_config_swap("test.yaml")

            assert result.success

            # Verify new config preserves original UI and remote settings
            # The actual config created during swap preserves original settings
            assert handler.context.config.style is original_style
            assert handler.context.config.remote_params is original_remote

    def test_context_window_manager_reset(self, model_handler):
        """Test that context window manager is properly reset on model swap."""
        handler, current_branch = model_handler

        # Set up existing context window manager
        handler.context._context_window_manager = MagicMock()

        mock_config = InferenceConfig(
            model=ModelParams(model_name="new-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.NATIVE,
        )

        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml", return_value=mock_config
            ),
            patch("oumi.infer.get_engine") as mock_get_engine,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            result = handler._handle_config_swap("test.yaml")

            assert result.success
            # Verify context window manager was reset
            assert handler.context._context_window_manager is None

