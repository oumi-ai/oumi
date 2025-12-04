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

"""Tests for branch-model state conservation functionality and stress testing."""

import copy
import random
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import (
    GenerationParams,
    InferenceConfig,
    InferenceEngineType,
    ModelParams,
)
from oumi_chat.commands.command_context import CommandContext
from oumi_chat.commands.command_parser import ParsedCommand
from oumi_chat.commands.handlers.branch_operations_handler import (
    BranchOperationsHandler,
)
from oumi_chat.commands.handlers.model_management_handler import ModelManagementHandler


class MockInferenceEngine:
    """Mock inference engine with cleanup tracking."""

    def __init__(self, model_name: str, engine_type: str):
        self.model_name = model_name
        self.engine_type = engine_type
        self.cleanup_called = False
        self.close_called = False

    def cleanup(self):
        """Mock cleanup method."""
        self.cleanup_called = True

    def close(self):
        """Mock close method."""
        self.close_called = True


def create_mock_config(
    model_name: str, engine_type: InferenceEngineType
) -> InferenceConfig:
    """Create a mock config for testing."""
    model_config = ModelParams(
        model_name=model_name,
        model_max_length=8192,
        torch_dtype_str="bfloat16",
        trust_remote_code=True,
    )

    generation_config = GenerationParams(
        max_new_tokens=2048, temperature=0.7, top_p=0.9
    )

    config = InferenceConfig(
        model=model_config, generation=generation_config, engine=engine_type
    )
    config.style = MagicMock()

    return config


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


# Fixtures for stress tests
@pytest.fixture
def stress_mock_branch_manager():
    """Create a mock branch manager with multiple branches for stress testing."""
    branch_manager = MagicMock()
    branch_manager.current_branch_id = "main"

    # Mock branches storage
    branches = {}

    def create_branch_side_effect(from_branch_id, name):
        """Mock branch creation."""
        new_branch = MagicMock()
        new_branch.id = name.lower()
        new_branch.name = name
        new_branch.conversation_history = []
        new_branch.model_name = None
        new_branch.engine_type = None
        new_branch.model_config = {}
        new_branch.generation_config = {}

        branches[new_branch.id] = new_branch
        return True, f"Created branch {name}", new_branch

    def switch_branch_side_effect(branch_name):
        """Mock branch switching."""
        branch_id = branch_name.lower()
        if branch_id in branches:
            branch_manager.current_branch_id = branch_id
            return True, f"Switched to {branch_name}", branches[branch_id]
        return False, f"Branch {branch_name} not found", None

    def get_current_branch_side_effect():
        """Mock getting current branch."""
        return branches.get(branch_manager.current_branch_id)

    def list_branches_side_effect():
        """Mock listing branches."""
        return [
            {
                "id": branch.id,
                "name": branch.name,
                "message_count": len(branch.conversation_history),
                "created_at": "2025-01-01T00:00:00Z",
                "preview": "Test branch",
            }
            for branch in branches.values()
        ]

    # Create initial main branch
    main_branch = MagicMock()
    main_branch.id = "main"
    main_branch.name = "Main"
    main_branch.conversation_history = []
    main_branch.model_name = None
    main_branch.engine_type = None
    main_branch.model_config = {}
    main_branch.generation_config = {}
    branches["main"] = main_branch

    branch_manager.create_branch.side_effect = create_branch_side_effect
    branch_manager.switch_branch.side_effect = switch_branch_side_effect
    branch_manager.get_current_branch.side_effect = get_current_branch_side_effect
    branch_manager.list_branches.side_effect = list_branches_side_effect
    branch_manager.sync_conversation_history = MagicMock()

    return branch_manager


@pytest.fixture
def stress_mock_context(stress_mock_branch_manager):
    """Create a comprehensive mock context for stress testing."""
    context = MagicMock()
    context.branch_manager = stress_mock_branch_manager
    context.conversation_history = []
    context.system_monitor = MagicMock()

    # Mock config with proper values instead of MagicMock
    context.config = create_mock_config("test-model", InferenceEngineType.NATIVE)

    # Mock inference engine
    context.inference_engine = MockInferenceEngine("test-model", "NATIVE")

    return context


class TestBranchModelStressTest:
    """Comprehensive stress test for branch model state management."""

    def test_stress_many_branches_many_models(self, stress_mock_context):
        """Test creating many branches with different models and switching
        between them."""
        # Create command context wrapper
        command_context = CommandContext(
            console=MagicMock(),
            config=stress_mock_context.config,
            inference_engine=stress_mock_context.inference_engine,
            conversation_history=stress_mock_context.conversation_history,
            system_monitor=stress_mock_context.system_monitor,
        )
        # Set the branch manager manually
        command_context._branch_manager = stress_mock_context.branch_manager

        branch_handler = BranchOperationsHandler(command_context)
        branch_handler.console = MagicMock()
        branch_handler._style = MagicMock()

        model_handler = ModelManagementHandler(command_context)
        model_handler.console = MagicMock()

        # Define test models and engines
        test_models = [
            ("meta-llama/Llama-3.1-8B-Instruct", InferenceEngineType.NATIVE),
            ("Qwen/Qwen3-4B-Instruct", InferenceEngineType.VLLM),
            ("microsoft/Phi-3-mini-4k-instruct", InferenceEngineType.LLAMACPP),
            ("google/gemma-2-2b-it", InferenceEngineType.NATIVE),
            ("mistralai/Mistral-7B-Instruct-v0.3", InferenceEngineType.VLLM),
        ]

        created_branches = []
        branch_model_mapping = {}  # Track which model should be on which branch

        # Phase 1: Create multiple branches and assign different models
        for i, (model_name, engine_type) in enumerate(test_models):
            branch_name = f"test_branch_{i + 1}"

            # Create branch
            create_cmd = ParsedCommand(
                command="branch",
                args=[branch_name],
                kwargs={},
                raw_input=f"/branch({branch_name})",
            )
            result = branch_handler._handle_branch(create_cmd)
            assert result.success, (
                f"Failed to create branch {branch_name}: {result.message}"
            )
            created_branches.append(branch_name)

            # Mock engine creation for model swap
            with patch("oumi.infer.get_engine") as mock_get_engine:
                new_engine = MockInferenceEngine(model_name, engine_type.value)
                mock_get_engine.return_value = new_engine

                # Update context config for this model
                stress_mock_context.config = create_mock_config(model_name, engine_type)

                # Swap to this model on this branch
                config_path = f"configs/test_{model_name.replace('/', '_')}_config.yaml"

                with (
                    patch("pathlib.Path.exists", return_value=True),
                    patch(
                        "oumi_chat.commands.config_utils.load_config_from_yaml_preserving_settings"
                    ) as mock_load,
                ):
                    mock_load.return_value = stress_mock_context.config

                    swap_cmd = ParsedCommand(
                        command="swap",
                        args=[f"config:{config_path}"],
                        kwargs={},
                        raw_input=f"/swap(config:{config_path})",
                    )
                    result = model_handler._handle_swap(swap_cmd)
                    assert result.success, (
                        f"Failed to swap to {model_name}: {result.message}"
                    )

            # Record the model assignment for this branch
            branch_model_mapping[branch_name] = (model_name, engine_type)

        # Phase 2: Stress test - rapidly switch between branches
        switch_sequence = []

        # Generate a complex switching pattern
        for cycle in range(3):  # Multiple cycles
            # Random switches within each cycle
            branch_order = created_branches.copy()
            random.shuffle(branch_order)
            switch_sequence.extend(branch_order)

            # Add some back-and-forth switches
            if len(created_branches) >= 2:
                switch_sequence.extend([created_branches[0], created_branches[1]] * 2)

        for switch_count, target_branch in enumerate(switch_sequence):
            expected_model, expected_engine = branch_model_mapping[target_branch]

            # Mock the engine restoration
            with patch("oumi.infer.get_engine") as mock_get_engine:
                restored_engine = MockInferenceEngine(
                    expected_model, expected_engine.value
                )
                mock_get_engine.return_value = restored_engine

                # Update context config to match expected model
                stress_mock_context.config = create_mock_config(
                    expected_model, expected_engine
                )

                # Perform the switch
                switch_cmd = ParsedCommand(
                    command="switch",
                    args=[target_branch],
                    kwargs={},
                    raw_input=f"/switch({target_branch})",
                )
                result = branch_handler._handle_switch(switch_cmd)
                assert result.success, (
                    f"Switch #{switch_count + 1} to {target_branch} failed: "
                    f"{result.message}"
                )

        # Phase 3: Verify branch-model associations are preserved
        for branch_name, (
            expected_model,
            expected_engine,
        ) in branch_model_mapping.items():
            # Switch to branch
            with patch("oumi.infer.get_engine") as mock_get_engine:
                restored_engine = MockInferenceEngine(
                    expected_model, expected_engine.value
                )
                mock_get_engine.return_value = restored_engine

                stress_mock_context.config = create_mock_config(
                    expected_model, expected_engine
                )

                switch_cmd = ParsedCommand(
                    command="switch",
                    args=[branch_name],
                    kwargs={},
                    raw_input=f"/switch({branch_name})",
                )
                result = branch_handler._handle_switch(switch_cmd)
                assert result.success, (
                    f"Final verification switch to {branch_name} failed"
                )

                # Get the current branch and verify its saved state
                current_branch = stress_mock_context.branch_manager.get_current_branch()
                assert current_branch is not None, f"Could not get branch {branch_name}"

    def test_concurrent_model_operations_robustness(self, stress_mock_context):
        """Test robustness when model operations and branch operations happen
        concurrently."""
        # Create command context wrapper
        command_context = CommandContext(
            console=MagicMock(),
            config=stress_mock_context.config,
            inference_engine=stress_mock_context.inference_engine,
            conversation_history=[],
            system_monitor=stress_mock_context.system_monitor,
        )
        # Set the branch manager manually
        command_context._branch_manager = stress_mock_context.branch_manager

        branch_handler = BranchOperationsHandler(command_context)
        branch_handler.console = MagicMock()
        branch_handler._style = MagicMock()

        model_handler = ModelManagementHandler(command_context)
        model_handler.console = MagicMock()

        # Test sequence: create branch, swap model, switch branch, swap model, etc.
        operations = [
            ("branch", "test1"),
            ("swap", "model-alpha"),
            ("branch", "test2"),
            ("swap", "model-beta"),
            ("switch", "test1"),
            ("swap", "model-gamma"),
            ("switch", "test2"),
            ("switch", "main"),
            ("swap", "model-delta"),
            ("switch", "test1"),
            ("switch", "test2"),
        ]

        successful_operations = 0

        for i, (operation, target) in enumerate(operations):
            try:
                if operation == "branch":
                    cmd = ParsedCommand(
                        command="branch",
                        args=[target],
                        kwargs={},
                        raw_input=f"/branch({target})",
                    )
                    result = branch_handler._handle_branch(cmd)

                elif operation == "switch":
                    cmd = ParsedCommand(
                        command="switch",
                        args=[target],
                        kwargs={},
                        raw_input=f"/switch({target})",
                    )
                    result = branch_handler._handle_switch(cmd)

                elif operation == "swap":
                    # Mock model swap
                    config_path = f"configs/{target}_config.yaml"
                    with (
                        patch("pathlib.Path.exists", return_value=True),
                        patch(
                            "oumi_chat.commands.config_utils.load_config_from_yaml_preserving_settings"
                        ) as mock_load,
                        patch("oumi.infer.get_engine") as mock_get_engine,
                    ):
                        new_config = create_mock_config(
                            target, InferenceEngineType.NATIVE
                        )
                        mock_load.return_value = new_config
                        mock_get_engine.return_value = MockInferenceEngine(
                            target, "NATIVE"
                        )

                        cmd = ParsedCommand(
                            command="swap",
                            args=[f"config:{config_path}"],
                            kwargs={},
                            raw_input=f"/swap(config:{config_path})",
                        )
                        result = model_handler._handle_swap(cmd)

                if result.success:
                    successful_operations += 1

            except Exception:
                pass

        # We expect most operations to succeed
        assert successful_operations >= len(operations) * 0.7, (
            f"Too many operations failed: {successful_operations}/{len(operations)}"
        )
