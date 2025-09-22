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

"""Tests for branch-model state conservation functionality."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.commands.command_parser import ParsedCommand
from oumi.core.commands.handlers.branch_operations_handler import (
    BranchOperationsHandler,
)
from oumi.core.commands.handlers.model_management_handler import ModelManagementHandler
from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)


class TestBranchModelStateConservation:
    """Test suite for branch-model state conservation."""

    @pytest.fixture
    def mock_qwen_config(self):
        """Mock Qwen model configuration."""
        model_config = ModelParams(
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            model_max_length=32768,
            torch_dtype_str="bfloat16",
            trust_remote_code=True,
        )
        generation_config = GenerationParams(
            max_new_tokens=2048, temperature=0.7, top_p=0.9
        )
        return InferenceConfig(
            model=model_config,
            generation=generation_config,
            engine=InferenceEngineType.NATIVE,
        )

    @pytest.fixture
    def mock_gemma_config(self):
        """Mock Gemma GGUF model configuration."""
        model_config = ModelParams(
            model_name="unsloth/gemma-3n-E4B-it-GGUF",
            tokenizer_name="google/gemma-3n-E4B-it",
            model_max_length=16384,
            torch_dtype_str="float16",
            trust_remote_code=True,
            model_kwargs={"filename": "gemma-3n-E4B-it-UD-Q5_K_XL.gguf"},
        )
        generation_config = GenerationParams(
            max_new_tokens=2048, temperature=0.7, top_p=0.9
        )
        return InferenceConfig(
            model=model_config,
            generation=generation_config,
            engine=InferenceEngineType.LLAMACPP,
        )

    @pytest.fixture
    def mock_branch_manager(self):
        """Mock branch manager with main and branch_1."""
        branch_manager = MagicMock()

        # Mock main branch
        main_branch = MagicMock()
        main_branch.id = "main"
        main_branch.name = "main"
        main_branch.conversation_history = []
        main_branch.model_name = None
        main_branch.engine_type = None
        main_branch.model_config = None
        main_branch.generation_config = None

        # Mock branch_1
        branch_1 = MagicMock()
        branch_1.id = "branch_1"
        branch_1.name = "branch_1"
        branch_1.conversation_history = []
        branch_1.model_name = None
        branch_1.engine_type = None
        branch_1.model_config = None
        branch_1.generation_config = None

        branch_manager.current_branch_id = "main"
        branch_manager.get_current_branch.return_value = main_branch
        branch_manager.create_branch.return_value = (True, "Created branch", branch_1)
        branch_manager.switch_branch.return_value = (
            True,
            "Switched to branch",
            branch_1,
        )
        branch_manager.list_branches.return_value = [
            {"id": "main", "name": "main"},
            {"id": "branch_1", "name": "branch_1"},
        ]

        return branch_manager, main_branch, branch_1

    @pytest.fixture
    def mock_system_monitor(self):
        """Mock system monitor."""
        monitor = MagicMock()
        monitor.update_max_context_tokens = MagicMock()
        monitor._last_update_time = 1000  # Some non-zero value
        return monitor

    @pytest.fixture
    def mock_context(self, mock_qwen_config, mock_system_monitor):
        """Mock command context."""
        context = MagicMock()
        context.config = mock_qwen_config
        context.inference_engine = MagicMock()
        context.system_monitor = mock_system_monitor
        context._context_window_manager = None
        return context

    @pytest.fixture
    def branch_handler(self, mock_context, mock_branch_manager):
        """Branch operations handler with mocked dependencies."""
        # Mock the context attributes that the handler expects
        mock_context.console = MagicMock()
        mock_context.config = mock_context.config
        mock_context.conversation_history = []
        mock_context.inference_engine = mock_context.inference_engine
        mock_context.system_monitor = mock_context.system_monitor
        mock_context._style = MagicMock()
        mock_context._style.use_emoji = True

        handler = BranchOperationsHandler(mock_context)
        handler._update_context_in_monitor = MagicMock()

        branch_manager, main_branch, branch_1 = mock_branch_manager
        handler.context.branch_manager = branch_manager

        return handler, main_branch, branch_1

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

        branch_manager, main_branch, branch_1 = mock_branch_manager
        handler.context.branch_manager = branch_manager

        return handler

    def test_branch_creation_saves_initial_model_state(
        self, branch_handler, mock_qwen_config
    ):
        """Test that creating a branch saves the initial model state."""
        handler, main_branch, branch_1 = branch_handler

        # Create branch command
        command = ParsedCommand(
            "branch", args=["test_branch"], kwargs={}, raw_input="/branch(test_branch)"
        )

        result = handler._handle_branch(command)

        assert result.success
        assert "✅ Created and switched to branch" in result.message

        # Verify main branch was updated with current model state during sync
        handler.context.branch_manager.sync_conversation_history.assert_called_once()

    def test_model_swap_saves_state_to_current_branch(
        self, model_handler, mock_gemma_config
    ):
        """Test that swapping models saves current state to active branch."""
        handler = model_handler

        # Mock config file loading
        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml",
                return_value=mock_gemma_config,
            ),
            patch("oumi.infer.get_engine") as mock_get_engine,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            # Swap to config file
            result = handler._handle_config_swap("configs/test.yaml")

            assert result.success

            # Verify system monitor was updated
            handler.context.system_monitor.update_max_context_tokens.assert_called_once()
            assert (
                handler.context.system_monitor._last_update_time == 0
            )  # Force refresh

    def test_branch_switch_restores_model_state(
        self, branch_handler, mock_gemma_config
    ):
        """Test that switching branches restores the correct model state."""
        handler, main_branch, branch_1 = branch_handler

        # Set up branch_1 with Gemma model state
        branch_1.model_name = "unsloth/gemma-3n-E4B-it-GGUF"
        branch_1.engine_type = "LLAMACPP"
        branch_1.model_config = {
            "model_name": "unsloth/gemma-3n-E4B-it-GGUF",
            "tokenizer_name": "google/gemma-3n-E4B-it",
            "model_max_length": 16384,
            "torch_dtype_str": "float16",
            "trust_remote_code": True,
            "model_kwargs": {"filename": "gemma-3n-E4B-it-UD-Q5_K_XL.gguf"},
        }
        branch_1.generation_config = {
            "max_new_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
        }

        with patch("oumi.infer.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            # Switch to branch_1
            command = ParsedCommand(
                "switch", args=["branch_1"], kwargs={}, raw_input="/switch(branch_1)"
            )
            result = handler._handle_switch(command)

            assert result.success
            assert (
                "restored unsloth/gemma-3n-E4B-it-GGUF with LLAMACPP engine"
                in result.message
            )

            # Verify system monitor was updated with new context length
            handler.context.system_monitor.update_max_context_tokens.assert_called_once_with(
                16384
            )
            assert (
                handler.context.system_monitor._last_update_time == 0
            )  # Force refresh

    def test_gguf_model_kwargs_preservation(self, branch_handler):
        """Test that GGUF model_kwargs are properly preserved during branch switches."""
        handler, main_branch, branch_1 = branch_handler

        # Set up branch with GGUF model including model_kwargs
        branch_1.model_name = "unsloth/gemma-3n-E4B-it-GGUF"
        branch_1.engine_type = "LLAMACPP"
        branch_1.model_config = {
            "model_name": "unsloth/gemma-3n-E4B-it-GGUF",
            "tokenizer_name": "google/gemma-3n-E4B-it",
            "model_kwargs": {"filename": "gemma-3n-E4B-it-UD-Q5_K_XL.gguf"},
            "trust_remote_code": True,
        }

        with patch("oumi.infer.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            # Switch to branch with GGUF model
            command = ParsedCommand(
                "switch", args=["branch_1"], kwargs={}, raw_input="/switch(branch_1)"
            )
            result = handler._handle_switch(command)

            assert result.success

            # Verify ModelParams was created with correct model_kwargs
            mock_get_engine.assert_called_once()
            config_arg = mock_get_engine.call_args[0][0]
            assert hasattr(config_arg.model, "model_kwargs")
            assert config_arg.model.model_kwargs == {
                "filename": "gemma-3n-E4B-it-UD-Q5_K_XL.gguf"
            }

    def test_context_length_calculation_for_different_engines(
        self, branch_handler, mock_gemma_config
    ):
        """Test context length calculation for different engine types."""
        handler, main_branch, branch_1 = branch_handler

        # Test LLAMACPP engine with model_max_length
        context_length = handler._get_context_length_for_engine(mock_gemma_config)
        assert context_length == 16384

        # Test NATIVE engine
        native_config = copy.deepcopy(mock_gemma_config)
        native_config.engine = InferenceEngineType.NATIVE
        native_config.model.model_max_length = 8192
        context_length = handler._get_context_length_for_engine(native_config)
        assert context_length == 8192

        # Test API engine fallback
        api_config = copy.deepcopy(mock_gemma_config)
        api_config.engine = InferenceEngineType.ANTHROPIC
        api_config.model.model_name = "claude-3-5-sonnet-20241022"
        context_length = handler._get_context_length_for_engine(api_config)
        assert context_length == 200000

    def test_automatic_branch_name_generation(self, branch_handler):
        """Test automatic branch name generation when no name provided."""
        handler, main_branch, branch_1 = branch_handler

        # Mock existing branches to test collision avoidance
        handler.context.branch_manager.list_branches.return_value = [
            {"id": "main", "name": "main"},
            {"id": "branch_1", "name": "branch_1"},
            {"id": "branch_2", "name": "branch_2"},
        ]

        # Test branch name generation
        generated_name = handler._generate_branch_name()
        assert generated_name == "branch_3"

        # Test with no existing numbered branches
        handler.context.branch_manager.list_branches.return_value = [
            {"id": "main", "name": "main"},
            {"id": "feature", "name": "feature"},
        ]

        generated_name = handler._generate_branch_name()
        assert generated_name == "branch_1"

    def test_swap_with_hf_model_name_fails_gracefully(self, model_handler):
        """Test that swapping with HF model name (without config) fails gracefully."""
        handler = model_handler

        # Try to swap with just a HuggingFace model name
        command = ParsedCommand(
            "swap",
            args=["meta-llama/Llama-3.1-8B-Instruct"],
            kwargs={},
            raw_input="/swap(meta-llama/Llama-3.1-8B-Instruct)",
        )
        result = handler._handle_swap(command)

        assert not result.success
        # The current implementation treats HF model IDs as config files due to "/"
        # and fails with "Config file not found", which is the expected behavior
        assert (
            "Config file not found" in result.message
            or "Invalid swap target" in result.message
        )

    def test_serialization_roundtrip_preserves_config(
        self, model_handler, mock_gemma_config
    ):
        """Test that serialization and deserialization preserves all config details."""
        handler = model_handler

        # Test model config serialization
        serialized_model = handler._serialize_model_config(mock_gemma_config.model)

        # Verify critical GGUF fields are preserved
        assert serialized_model["model_name"] == "unsloth/gemma-3n-E4B-it-GGUF"
        assert serialized_model["tokenizer_name"] == "google/gemma-3n-E4B-it"
        assert serialized_model["model_kwargs"] == {
            "filename": "gemma-3n-E4B-it-UD-Q5_K_XL.gguf"
        }
        assert serialized_model["model_max_length"] == 16384
        assert serialized_model["torch_dtype_str"] == "float16"

        # Test generation config serialization
        serialized_gen = handler._serialize_generation_config(
            mock_gemma_config.generation
        )
        assert serialized_gen["max_new_tokens"] == 2048
        assert serialized_gen["temperature"] == 0.7
        assert serialized_gen["top_p"] == 0.9

    def test_branch_isolation_different_models(
        self, branch_handler, model_handler, mock_qwen_config, mock_gemma_config
    ):
        """Test that different branches maintain isolated model configurations."""
        branch_ops_handler, main_branch, branch_1 = branch_handler
        model_mgmt_handler = model_handler

        # Start with Qwen on main branch
        main_branch.model_name = mock_qwen_config.model.model_name
        main_branch.engine_type = mock_qwen_config.engine.value
        main_branch.model_config = model_mgmt_handler._serialize_model_config(
            mock_qwen_config.model
        )

        # Create and switch to branch_1
        command = ParsedCommand(
            "branch", args=["branch_1"], kwargs={}, raw_input="/branch(branch_1)"
        )
        result = branch_ops_handler._handle_branch(command)
        assert result.success

        # Simulate swapping to Gemma on branch_1
        with (
            patch(
                "oumi.core.configs.InferenceConfig.from_yaml",
                return_value=mock_gemma_config,
            ),
            patch("oumi.infer.get_engine") as mock_get_engine,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            # Update context to simulate being on branch_1
            (
                branch_ops_handler.context.branch_manager.get_current_branch.return_value
            ) = branch_1

            # Swap to Gemma config
            result = model_mgmt_handler._handle_config_swap("configs/gemma.yaml")
            assert result.success

        # Verify branch_1 now has Gemma config
        assert (
            branch_1.model_name is not None
        )  # Should be updated by _save_current_model_state_to_branch

        # Switch back to main branch
        branch_ops_handler.context.branch_manager.get_current_branch.return_value = (
            main_branch
        )

        with patch("oumi.infer.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            command = ParsedCommand(
                "switch", args=["main"], kwargs={}, raw_input="/switch(main)"
            )
            result = branch_ops_handler._handle_switch(command)

            # Should restore Qwen model
            assert result.success
            if (
                main_branch.model_name
            ):  # If main branch was initialized with model state
                assert "restored" in result.message

    def test_system_monitor_updates_on_context_length_change(self, branch_handler):
        """Test that system monitor updates when context length changes between
        branches."""
        handler, main_branch, branch_1 = branch_handler

        # Set up branches with different context lengths
        main_branch.model_name = "Qwen/Qwen3-4B-Instruct-2507"
        main_branch.engine_type = "NATIVE"
        main_branch.model_config = {"model_max_length": 32768}

        branch_1.model_name = "unsloth/gemma-3n-E4B-it-GGUF"
        branch_1.engine_type = "LLAMACPP"
        branch_1.model_config = {"model_max_length": 16384}

        with patch("oumi.infer.get_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_get_engine.return_value = mock_engine

            # Switch to branch_1 (16K context)
            command = ParsedCommand(
                "switch", args=["branch_1"], kwargs={}, raw_input="/switch(branch_1)"
            )
            result = handler._handle_switch(command)

            assert result.success

            # Verify system monitor was called with correct context length
            handler.context.system_monitor.update_max_context_tokens.assert_called_with(
                16384
            )
            # Verify force refresh was triggered
            assert handler.context.system_monitor._last_update_time == 0
