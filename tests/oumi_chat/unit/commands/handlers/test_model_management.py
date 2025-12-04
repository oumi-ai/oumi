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

"""Unit tests for model management command handlers."""

from pathlib import Path
from unittest.mock import Mock, patch

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi_chat.commands import ParsedCommand
from oumi_chat.commands.command_context import CommandContext
from oumi_chat.commands.handlers.model_management_handler import ModelManagementHandler
from tests.oumi_chat.utils.chat_test_utils import (
    create_test_chat_config,
    validate_command_result,
)


class TestSwapCommand:
    """Test suite for /swap() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_chat_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ModelManagementHandler(context=self.command_context)

    def test_swap_model_name_treated_as_config_file(self):
        """Test that model name is treated as config file path and fails when file
        doesn't exist."""
        command = ParsedCommand(
            command="swap",
            args=["meta-llama/Llama-3.1-8B-Instruct"],
            kwargs={},
            raw_input="/swap(meta-llama/Llama-3.1-8B-Instruct)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=[
                "Config file not found",
                "meta-llama/Llama-3.1-8B-Instruct",
            ],
        )

    def test_swap_no_arguments(self):
        """Test swap command with no arguments."""
        command = ParsedCommand(
            command="swap",
            args=[],
            kwargs={},
            raw_input="/swap()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=[
                "swap command requires a model name or config path"
            ],
        )

    def test_swap_whitespace_only_argument(self):
        """Test swap command with whitespace-only argument (becomes empty after
        strip)."""
        command = ParsedCommand(
            command="swap",
            args=["  "],  # Whitespace only
            kwargs={},
            raw_input="/swap()",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=[
                "swap command requires a model name or config path argument"
            ],
        )

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    @patch("oumi.infer.get_engine")
    def test_swap_config_file_success(
        self, mock_get_engine, mock_from_yaml, test_file_manager
    ):
        """Test successful config-based model swap."""
        # Create a temporary config file
        config_content = """
model:
  model_name: "test-model"
  model_max_length: 2048

generation:
  temperature: 0.8
  top_p: 0.9

engine: NATIVE
"""
        config_file = test_file_manager.create_temp_file(
            filename="test_config.yaml", content=config_content
        )

        # Mock the config loading
        mock_new_config = InferenceConfig(
            model=ModelParams(model_name="test-model", model_max_length=2048),
            generation=GenerationParams(temperature=0.8, top_p=0.9),
            engine=InferenceEngineType.NATIVE,
        )
        mock_from_yaml.return_value = mock_new_config

        # Mock the engine creation
        mock_new_engine = Mock()
        mock_new_engine.model_name = "test-model"
        mock_get_engine.return_value = mock_new_engine

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=[config_file],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=True,
            expected_message_parts=["Swapped to", "test-model", "NATIVE"],
        )

        # Verify that context was updated
        assert self.command_context.inference_engine == mock_new_engine
        assert self.command_context.config == mock_new_config

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    def test_swap_config_file_not_found(self, mock_from_yaml):
        """Test config swap with non-existent file."""
        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=["nonexistent_config.yaml"],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Config file not found", "nonexistent_config.yaml"],
        )

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    def test_swap_config_file_invalid_yaml(self, mock_from_yaml, test_file_manager):
        """Test config swap with invalid YAML file."""
        # Create invalid YAML file
        invalid_config = """
model:
  model_name: "test
  # Missing closing quote - invalid YAML
"""
        config_file = test_file_manager.create_temp_file(
            filename="invalid_config.yaml", content=invalid_config
        )

        # Mock YAML parsing to raise exception
        mock_from_yaml.side_effect = Exception("Invalid YAML format")

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=[config_file],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Error loading config", "Invalid YAML"],
        )

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    @patch("oumi.infer.get_engine")
    def test_swap_config_engine_creation_fails(
        self, mock_get_engine, mock_from_yaml, test_file_manager
    ):
        """Test config swap when engine creation fails."""
        config_content = """
model:
  model_name: "invalid-model"

engine: INVALID_ENGINE
"""
        config_file = test_file_manager.create_temp_file(
            filename="test_config.yaml", content=config_content
        )

        # Mock successful config loading
        mock_new_config = InferenceConfig(
            model=ModelParams(model_name="invalid-model"),
            engine=InferenceEngineType.NATIVE,
        )
        mock_from_yaml.return_value = mock_new_config

        # Mock engine creation failure
        mock_get_engine.side_effect = Exception("Invalid engine type")

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=[config_file],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=[
                "Error creating inference engine",
                "Invalid engine",
            ],
        )

    def test_swap_config_prefix_detection(self, test_file_manager):
        """Test that 'config:' prefix is properly detected."""
        config_content = "model:\n  model_name: test"
        config_file = test_file_manager.create_temp_file(
            filename="prefixed_config.yaml", content=config_content
        )

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=[f"config:{config_file}"],
            kwargs={},
        )

        # Should attempt config-based swap (will fail due to mocking, but that's
        # expected)
        result = self.handler.handle_command(command)

        # Since we're not mocking the full chain, expect it to fail at config loading
        validate_command_result(
            result,
            expect_success=False,
        )

    def test_swap_yaml_extension_detection(self, test_file_manager):
        """Test that .yaml/.yml extensions are auto-detected as config files."""
        for extension in [".yaml", ".yml"]:
            config_file = test_file_manager.create_temp_file(
                filename=f"auto_detect{extension}",
                content="model:\n  model_name: test",
            )

            command = ParsedCommand(
                raw_input="/swap(...)",
                command="swap",
                args=[config_file],
                kwargs={},
            )

            result = self.handler.handle_command(command)

            # Should attempt config-based swap (fails due to incomplete mocking)
            validate_command_result(result, expect_success=False)

    def test_swap_path_detection(self, test_file_manager):
        """Test that paths with slashes are detected as config files."""
        config_content = "model:\n  model_name: test"

        # Create a config file in a subdirectory-like path
        config_file = test_file_manager.create_temp_file(
            filename="subdir_config.yaml", content=config_content
        )

        # Modify the path to include a slash (simulate subdirectory)
        config_path = str(Path(config_file).parent / "subdir" / Path(config_file).name)

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=[config_path],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        # Should attempt config-based swap but fail since file doesn't exist
        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Config file not found"],
        )

    def test_swap_invalid_model_name(self):
        """Test swap command with invalid model name (no slashes, no config
        extension)."""
        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=["simple-model-name"],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        # Simple model name (no slashes, no .yaml/.yml) gets clear error message
        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Invalid swap target", "simple-model-name"],
        )

    def test_swap_empty_config_prefix(self):
        """Test swap command with empty config: prefix."""
        command = ParsedCommand(
            command="swap",
            args=["config:"],
            kwargs={},
            raw_input="/swap(config:)",
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["config: prefix requires a path"],
        )


class TestListEnginesCommand:
    """Test suite for /list_engines() command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_chat_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ModelManagementHandler(context=self.command_context)

        # Mock the style attributes
        self.handler._style = Mock()
        self.handler._style.use_emoji = True
        self.handler._style.assistant_border_style = "cyan"

    def test_list_engines_success(self):
        """Test successful engines listing."""
        command = ParsedCommand(
            raw_input="/list_engines()",
            command="list_engines",
            args=[],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)

        # Verify console.print was called with a Panel
        self.mock_console.print.assert_called_once()
        printed_arg = self.mock_console.print.call_args[0][0]
        assert hasattr(printed_arg, "renderable")  # Should be a Panel

    def test_list_engines_displays_all_engine_types(self):
        """Test that list_engines displays all expected engine types."""
        engines_info = self.handler._get_engines_info()

        # Check that we have engines of all types
        engine_types = {engine["type"] for engine in engines_info}
        assert "Local" in engine_types
        assert "API" in engine_types
        assert "Remote" in engine_types

        # Check for specific engines we expect
        engine_names = {engine["name"] for engine in engines_info}
        expected_engines = {
            "NATIVE",
            "VLLM",
            "LLAMACPP",
            "ANTHROPIC",
            "OPENAI",
            "TOGETHER",
            "DEEPSEEK",
        }
        assert expected_engines.issubset(engine_names)

    def test_list_engines_api_key_requirements(self):
        """Test that API key requirements are properly marked."""
        engines_info = self.handler._get_engines_info()

        # API engines should require keys
        api_engines = [e for e in engines_info if e["type"] == "API"]
        for engine in api_engines:
            assert engine.get("api_key_required", False) is True

        # Local engines should not require keys
        local_engines = [e for e in engines_info if e["type"] == "Local"]
        for engine in local_engines:
            assert engine.get("api_key_required", False) is False

    def test_list_engines_sample_models(self):
        """Test that engines have appropriate sample models."""
        engines_info = self.handler._get_engines_info()

        for engine in engines_info:
            assert "sample_models" in engine
            assert isinstance(engine["sample_models"], list)
            assert len(engine["sample_models"]) > 0

            # Check for reasonable sample models
            if engine["name"] == "ANTHROPIC":
                sample_models = engine["sample_models"]
                assert any("claude" in model.lower() for model in sample_models)

            elif engine["name"] == "OPENAI":
                sample_models = engine["sample_models"]
                assert any("gpt" in model.lower() for model in sample_models)

    def test_list_engines_without_emoji(self):
        """Test list_engines with emoji disabled."""
        # Mock the style object with use_emoji disabled
        mock_style = Mock()
        mock_style.use_emoji = False
        self.handler._style = mock_style

        command = ParsedCommand(
            raw_input="/list_engines()",
            command="list_engines",
            args=[],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(result, expect_success=True)
        self.mock_console.print.assert_called_once()

    def test_list_engines_exception_handling(self):
        """Test list_engines with exception during execution."""
        # Mock _get_engines_info to raise an exception
        with patch.object(
            self.handler, "_get_engines_info", side_effect=Exception("Test error")
        ):
            command = ParsedCommand(
                raw_input="/list_engines()",
                command="list_engines",
                args=[],
                kwargs={},
            )

            result = self.handler.handle_command(command)

            validate_command_result(
                result,
                expect_success=False,
                expected_message_parts=["Error listing engines", "Test error"],
            )


class TestModelManagementHandler:
    """Test suite for ModelManagementHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_chat_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ModelManagementHandler(context=self.command_context)

    def test_get_supported_commands(self):
        """Test that handler returns correct supported commands."""
        supported = self.handler.get_supported_commands()
        assert set(supported) == {"swap", "list_engines"}

    def test_unsupported_command(self):
        """Test handler with unsupported command."""
        command = ParsedCommand(
            raw_input="/unsupported()",
            command="unsupported",
            args=[],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        validate_command_result(
            result,
            expect_success=False,
            expected_message_parts=["Unsupported command", "unsupported"],
        )


class TestHelperMethods:
    """Test suite for helper methods in ModelManagementHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_chat_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ModelManagementHandler(context=self.command_context)

    def test_serialize_model_config(self):
        """Test model config serialization."""
        model_config = ModelParams(
            model_name="test-model",
            model_max_length=2048,
            torch_dtype_str="float16",
            attn_implementation="sdpa",
        )

        serialized = self.handler._serialize_model_config(model_config)

        assert serialized["model_name"] == "test-model"
        assert serialized["model_max_length"] == "2048"
        assert serialized["torch_dtype_str"] == "float16"
        assert serialized["attn_implementation"] == "sdpa"

    def test_serialize_model_config_none(self):
        """Test model config serialization with None input."""
        serialized = self.handler._serialize_model_config(None)
        assert serialized == {}

    def test_serialize_generation_config(self):
        """Test generation config serialization with all fields."""
        gen_config = GenerationParams(
            max_new_tokens=100,
            batch_size=2,
            exclude_prompt_from_response=False,
            temperature=0.8,
            top_p=0.9,
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=-0.3,
            stop_strings=["END"],
            min_p=0.1,
            use_cache=True,
            num_beams=1,
            use_sampling=True,
        )

        serialized = self.handler._serialize_generation_config(gen_config)

        # Verify all fields are included
        assert serialized["max_new_tokens"] == 100
        assert serialized["batch_size"] == 2
        assert serialized["exclude_prompt_from_response"] is False
        assert serialized["temperature"] == 0.8
        assert serialized["top_p"] == 0.9
        assert serialized["seed"] == 42
        assert serialized["frequency_penalty"] == 0.5
        assert serialized["presence_penalty"] == -0.3
        assert serialized["stop_strings"] == ["END"]
        assert serialized["min_p"] == 0.1
        assert serialized["use_cache"] is True
        assert serialized["num_beams"] == 1
        assert serialized["use_sampling"] is True

    def test_serialize_generation_config_none(self):
        """Test generation config serialization with None input."""
        serialized = self.handler._serialize_generation_config(None)
        assert serialized == {}

    def test_get_context_length_for_engine_local_engines(self):
        """Test context length calculation for local engines."""
        config = InferenceConfig(
            model=ModelParams(model_max_length=4096),
            engine=InferenceEngineType.NATIVE,
        )

        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 4096

        # Test VLLM engine
        config.engine = InferenceEngineType.VLLM
        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 4096

    def test_get_context_length_for_engine_anthropic(self):
        """Test context length calculation for Anthropic models."""
        config = InferenceConfig(
            model=ModelParams(model_name="claude-3-5-sonnet-20241022"),
            engine=InferenceEngineType.ANTHROPIC,
        )

        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 200000  # Anthropic models

    def test_get_context_length_for_engine_openai(self):
        """Test context length calculation for OpenAI models."""
        config = InferenceConfig(
            model=ModelParams(model_name="gpt-4o"),
            engine=InferenceEngineType.OPENAI,
        )

        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 128000  # GPT-4o

        # Test GPT-3.5
        config.model.model_name = "gpt-3.5-turbo"
        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 16385  # GPT-3.5

    def test_get_context_length_for_engine_together(self):
        """Test context length calculation for Together AI models."""
        config = InferenceConfig(
            model=ModelParams(model_name="meta-llama/Llama-3.1-405B-Instruct"),
            engine=InferenceEngineType.TOGETHER,
        )

        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 128000  # Large Llama models

    def test_get_context_length_for_engine_deepseek(self):
        """Test context length calculation for DeepSeek models."""
        config = InferenceConfig(
            model=ModelParams(model_name="deepseek-chat"),
            engine=InferenceEngineType.DEEPSEEK,
        )

        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 32768  # DeepSeek models

    def test_get_context_length_for_engine_default(self):
        """Test context length calculation with default fallback."""
        config = InferenceConfig(
            model=ModelParams(model_name="unknown-model", model_max_length=None),
            engine=InferenceEngineType.NATIVE,
        )

        context_length = self.handler._get_context_length_for_engine(config)
        assert context_length == 4096  # Default fallback

    def test_save_current_model_state_to_branch(self):
        """Test saving model state to branch (should not raise exceptions)."""
        # Test that the method doesn't raise exceptions even without branch manager
        # This tests the defensive programming aspect
        self.handler._save_current_model_state_to_branch()
        self.handler._save_current_model_state_to_branch()

    def test_restore_model_state_from_branch(self):
        """Test restoring model state from branch (should not raise exceptions)."""
        branch_data = {
            "model_config": {"model_name": "test-model"},
            "generation_config": {"temperature": 0.8},
        }

        # Should not raise exception
        self.handler._restore_model_state_from_branch(branch_data)
        self.handler._restore_model_state_from_branch({})


class TestConfigPathResolution:
    """Test suite for config path resolution logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_chat_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        self.handler = ModelManagementHandler(context=self.command_context)

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.cwd")
    def test_config_path_resolution_relative_to_cwd(self, mock_cwd, mock_exists):
        """Test config path resolution relative to current directory."""
        mock_cwd.return_value = Path("/current/dir")
        mock_exists.side_effect = lambda: str(self) == "/current/dir/config.yaml"

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=["config.yaml"],
            kwargs={},
        )

        result = self.handler.handle_command(command)

        # Should attempt to resolve path (will fail later due to incomplete mocking)
        validate_command_result(result, expect_success=False)

    def test_config_path_resolution_absolute_path(self, test_file_manager):
        """Test config path resolution with absolute path."""
        config_content = "model:\n  model_name: test"
        config_file = test_file_manager.create_temp_file(
            filename="absolute_config.yaml", content=config_content
        )

        command = ParsedCommand(
            raw_input="/swap(...)",
            command="swap",
            args=[config_file],  # Already absolute path
            kwargs={},
        )

        result = self.handler.handle_command(command)

        # Should attempt config swap (will fail due to incomplete mocking)
        validate_command_result(result, expect_success=False)


class TestModelSwapEdgeCases:
    """Test suite for model swapping edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.mock_console = Mock()
        self.test_config = create_test_chat_config()

        self.command_context = CommandContext(
            console=self.mock_console,
            config=self.test_config,
            conversation_history=[],
            inference_engine=self.mock_engine,
        )

        # Set up branch manager mock
        self.mock_branch_manager = Mock()
        self.current_branch = Mock()
        self.current_branch.id = "main"
        self.current_branch.model_name = None
        self.current_branch.engine_type = None
        self.current_branch.model_config = None
        self.current_branch.generation_config = None
        self.mock_branch_manager.get_current_branch.return_value = self.current_branch
        self.command_context._branch_manager = self.mock_branch_manager

        self.handler = ModelManagementHandler(context=self.command_context)

        # Set up system monitor mock
        self.command_context.system_monitor = Mock()
        self.command_context.system_monitor.update_max_context_tokens = Mock()
        self.command_context.system_monitor._last_update_time = 1000

    def test_swap_with_multiple_huggingface_model_ids_fails(self):
        """Test that swapping with various HuggingFace model IDs fails gracefully."""
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
            result = self.handler.handle_command(command)

            assert not result.success, f"Should fail for HF model ID: {model_id}"
            assert result.message is not None
            assert (
                "Config file not found" in result.message
                or "Invalid swap target" in result.message
            ), f"Got unexpected message: {result.message}"

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    @patch("oumi.infer.get_engine")
    def test_swap_with_config_prefix_succeeds(self, mock_get_engine, mock_from_yaml):
        """Test that swapping with config: prefix works correctly."""
        mock_config = InferenceConfig(
            model=ModelParams(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                model_max_length=8192,
                torch_dtype_str="bfloat16",
            ),
            generation=GenerationParams(max_new_tokens=2048),
            engine=InferenceEngineType.VLLM,
        )
        mock_from_yaml.return_value = mock_config

        mock_engine = Mock()
        mock_get_engine.return_value = mock_engine

        with patch("pathlib.Path.exists", return_value=True):
            command = ParsedCommand(
                "swap",
                args=["config:recipes/llama/8b_instruct.yaml"],
                kwargs={},
                raw_input="/swap(config:recipes/llama/8b_instruct.yaml)",
            )
            result = self.handler.handle_command(command)

            assert result.success
            assert result.message is not None
            assert "Swapped to meta-llama/Llama-3.1-8B-Instruct" in result.message
            assert "VLLM engine" in result.message

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    @patch("oumi.infer.get_engine")
    def test_swap_saves_current_model_state(self, mock_get_engine, mock_from_yaml):
        """Test that swapping saves the current model state to the active branch."""
        # Set initial model state
        self.handler.context.config.inference.model.model_name = "initial-model"
        self.handler.context.config.inference.engine = InferenceEngineType.NATIVE

        mock_new_config = InferenceConfig(
            model=ModelParams(model_name="new-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.VLLM,
        )
        mock_from_yaml.return_value = mock_new_config

        mock_engine = Mock()
        mock_get_engine.return_value = mock_engine

        with patch("pathlib.Path.exists", return_value=True):
            command = ParsedCommand(
                "swap",
                args=["test.yaml"],
                kwargs={},
                raw_input="/swap(test.yaml)",
            )
            result = self.handler.handle_command(command)

            assert result.success

            # Verify current branch was updated with previous model state
            assert self.current_branch.model_name == "initial-model"
            assert self.current_branch.engine_type == "NATIVE"

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    @patch("oumi.infer.get_engine")
    def test_swap_updates_system_monitor_for_various_engines(
        self, mock_get_engine, mock_from_yaml
    ):
        """Test that swapping updates system monitor with correct context length."""
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
            mock_from_yaml.return_value = mock_config

            mock_engine = Mock()
            mock_get_engine.return_value = mock_engine

            # Reset monitor mock
            assert self.command_context.system_monitor is not None
            self.command_context.system_monitor.update_max_context_tokens.reset_mock()
            self.command_context.system_monitor._last_update_time = 1000

            with patch("pathlib.Path.exists", return_value=True):
                command = ParsedCommand(
                    "swap",
                    args=["test.yaml"],
                    kwargs={},
                    raw_input="/swap(test.yaml)",
                )
                result = self.handler.handle_command(command)

                assert result.success

                # Verify system monitor was updated with correct context length
                assert self.command_context.system_monitor is not None
                self.command_context.system_monitor.update_max_context_tokens.assert_called_once_with(
                    expected_context
                )

    def test_swap_handles_various_config_loading_errors(self):
        """Test that config loading errors are handled gracefully."""
        test_cases = [
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
                command = ParsedCommand(
                    "swap",
                    args=["test.yaml"],
                    kwargs={},
                    raw_input="/swap(test.yaml)",
                )
                result = self.handler.handle_command(command)

                assert not result.success
                assert result.message is not None
                assert expected_msg_part in result.message

    @patch("oumi.core.configs.InferenceConfig.from_yaml")
    def test_swap_handles_engine_creation_errors(self, mock_from_yaml):
        """Test that engine creation errors are handled gracefully."""
        mock_config = InferenceConfig(
            model=ModelParams(model_name="problematic-model"),
            generation=GenerationParams(),
            engine=InferenceEngineType.VLLM,
        )
        mock_from_yaml.return_value = mock_config

        with (
            patch(
                "oumi.infer.get_engine", side_effect=Exception("Engine creation failed")
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            command = ParsedCommand(
                "swap",
                args=["test.yaml"],
                kwargs={},
                raw_input="/swap(test.yaml)",
            )
            result = self.handler.handle_command(command)

            assert not result.success
            assert result.message is not None
            assert (
                "Error creating inference engine: Engine creation failed"
                in result.message
            )

    def test_swap_with_various_empty_targets_fails(self):
        """Test that empty swap targets fail gracefully."""
        test_cases = [
            ("", "requires a model name or config path argument"),
            ("   ", "requires a model name or config path argument"),
            ("config:", "config: prefix requires a path to a configuration file"),
            ("   config:   ", "config: prefix requires a path to a configuration file"),
        ]

        for target, expected_msg in test_cases:
            command = ParsedCommand(
                "swap", args=[target], kwargs={}, raw_input=f"/swap({target})"
            )
            result = self.handler.handle_command(command)

            assert not result.success
            assert result.message is not None
            assert expected_msg in result.message
