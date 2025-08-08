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

"""Test branch model state preservation and restoration."""

import pytest
from unittest.mock import MagicMock, patch

from src.oumi.core.commands.conversation_branches import ConversationBranch, ConversationBranchManager
from src.oumi.core.commands.command_handler import CommandHandler
from src.oumi.core.commands.command_parser import ParsedCommand
from src.oumi.core.configs import InferenceConfig


@pytest.fixture
def mock_inference_config():
    """Mock inference configuration."""
    config = MagicMock(spec=InferenceConfig)
    config.engine = "OPENAI"
    config.model = MagicMock()
    config.model.model_name = "gpt-5"
    config.generation = MagicMock()
    config.generation.max_new_tokens = 4096
    config.generation.temperature = 1.0
    config.style = MagicMock()
    return config


@pytest.fixture
def mock_inference_engine():
    """Mock inference engine."""
    engine = MagicMock()
    engine.get_model_name.return_value = "gpt-5"
    return engine


@pytest.fixture
def command_handler(mock_inference_config, mock_inference_engine):
    """Command handler with mocked dependencies."""
    handler = CommandHandler(
        console=MagicMock(),
        config=mock_inference_config,
        inference_engine=mock_inference_engine,
        conversation_history=[],
        system_monitor=MagicMock()
    )
    return handler


class TestBranchModelState:
    """Test branch model state preservation and restoration."""

    def test_branch_model_state_saved_on_creation(self, command_handler):
        """Test that model state is saved when creating a branch."""
        # Start with a model
        command_handler.config.model.model_name = "deepseek-r1"
        command_handler.config.engine = "DEEPSEEK"
        
        # Create branch
        branch_command = ParsedCommand("branch", [])
        result = command_handler._handle_branch(branch_command)
        
        # Check that branch was created successfully
        assert result.success
        assert "Created branch" in result.message
        
        # Check that model state was saved to the new branch
        branches = list(command_handler.branch_manager.branches.values())
        new_branch = [b for b in branches if b.id != "main"][0]
        
        assert new_branch.model_name == "deepseek-r1"
        assert new_branch.engine_type == "DEEPSEEK"
        assert new_branch.model_config is not None


    def test_branch_model_state_restored_on_switch(self, command_handler):
        """Test that model state is restored when switching branches."""
        # Start with DeepSeek model
        command_handler.config.model.model_name = "deepseek-r1"
        command_handler.config.engine = "DEEPSEEK"
        
        # Create branch (which saves current state)
        branch_result = command_handler._handle_branch()
        assert branch_result.success
        
        # Switch to GPT-5 on main
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            mock_new_engine = MagicMock()
            mock_new_engine.get_model_name.return_value = "gpt-5"
            mock_build.return_value = mock_new_engine
            
            swap_result = command_handler._handle_swap("gpt-5")
            assert swap_result.success
            assert command_handler.config.model.model_name == "gpt-5"
        
        # Switch to the branch (should restore DeepSeek)
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            mock_deepseek_engine = MagicMock()
            mock_deepseek_engine.get_model_name.return_value = "deepseek-r1"
            mock_build.return_value = mock_deepseek_engine
            
            # Get the branch ID
            branch_id = list(command_handler.branch_manager.branches.keys())[1]  # Not main
            
            switch_result = command_handler._handle_branch_switch(branch_id)
            assert switch_result.success
            
            # Verify model was restored
            assert command_handler.config.model.model_name == "deepseek-r1"
            assert command_handler.config.engine == "DEEPSEEK"


    def test_branch_switch_preserves_current_model_state(self, command_handler):
        """Test that switching away from a branch preserves its current model state."""
        # Start with model A
        command_handler.config.model.model_name = "claude-3-5-sonnet"
        command_handler.config.engine = "ANTHROPIC"
        
        # Create branch
        branch_result = command_handler._handle_branch()
        assert branch_result.success
        branch_id = list(command_handler.branch_manager.branches.keys())[1]  # Not main
        
        # Switch to branch
        with patch('oumi.builders.inference_engines.build_inference_engine'):
            switch_result = command_handler._handle_branch_switch(branch_id)
            assert switch_result.success
        
        # Change model on the branch
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            mock_new_engine = MagicMock()
            mock_new_engine.get_model_name.return_value = "gpt-5-nano"
            mock_build.return_value = mock_new_engine
            
            swap_result = command_handler._handle_swap("gpt-5-nano")
            assert swap_result.success
        
        # Switch back to main
        with patch('oumi.builders.inference_engines.build_inference_engine'):
            main_switch_result = command_handler._handle_branch_switch("main")
            assert main_switch_result.success
        
        # Switch back to branch - should have preserved the gpt-5-nano state
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            mock_nano_engine = MagicMock()
            mock_nano_engine.get_model_name.return_value = "gpt-5-nano"
            mock_build.return_value = mock_nano_engine
            
            branch_switch_result = command_handler._handle_branch_switch(branch_id)
            assert branch_switch_result.success
            
            # Should have restored gpt-5-nano, not the original claude model
            assert command_handler.config.model.model_name == "gpt-5-nano"
            assert command_handler.config.engine == "OPENAI"


    def test_branch_model_state_copied_on_creation(self, command_handler):
        """Test that branch creation copies model state from parent."""
        # Set up parent with specific model
        command_handler.config.model.model_name = "llama-3.1-70b"
        command_handler.config.engine = "VLLM"
        command_handler.config.generation.temperature = 0.7
        
        # Create branch
        branch_result = command_handler._handle_branch("test_branch")
        assert branch_result.success
        
        # Get the new branch
        branch = command_handler.branch_manager.get_branch_by_name("test_branch")
        assert branch is not None
        
        # Verify model state was copied
        assert branch.model_name == "llama-3.1-70b"
        assert branch.engine_type == "VLLM"
        assert branch.model_config is not None
        assert branch.generation_config is not None


    def test_no_engine_rebuild_if_same_model(self, command_handler):
        """Test that inference engine is not rebuilt if switching to same model."""
        # Set up initial model
        command_handler.config.model.model_name = "qwen-2.5-3b"
        command_handler.config.engine = "NATIVE"
        
        # Create branch with same model
        branch_result = command_handler._handle_branch()
        assert branch_result.success
        branch_id = list(command_handler.branch_manager.branches.keys())[1]  # Not main
        
        # Switch to branch - should not rebuild engine since model is same
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            switch_result = command_handler._handle_branch_switch(branch_id)
            assert switch_result.success
            
            # Engine should not have been rebuilt
            mock_build.assert_not_called()


    def test_engine_rebuild_if_different_model(self, command_handler):
        """Test that inference engine is rebuilt when switching to different model."""
        # Start with model A
        command_handler.config.model.model_name = "gpt-4"
        command_handler.config.engine = "OPENAI"
        
        # Create branch
        branch_result = command_handler._handle_branch()
        assert branch_result.success
        branch_id = list(command_handler.branch_manager.branches.keys())[1]  # Not main
        
        # Change main to different model
        with patch('oumi.builders.inference_engines.build_inference_engine'):
            swap_result = command_handler._handle_swap("claude-3-5-sonnet")
            assert swap_result.success
            command_handler.config.engine = "ANTHROPIC"  # Simulate the swap
        
        # Switch to branch - should rebuild engine since model differs
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            mock_gpt_engine = MagicMock()
            mock_gpt_engine.get_model_name.return_value = "gpt-4"
            mock_build.return_value = mock_gpt_engine
            
            switch_result = command_handler._handle_branch_switch(branch_id)
            assert switch_result.success
            
            # Engine should have been rebuilt
            mock_build.assert_called_once()


    def test_branch_model_display_consistency(self, command_handler):
        """Test the specific scenario from user's bug report."""
        # Start with DeepSeek
        command_handler.config.model.model_name = "deepseek-r1"
        command_handler.config.engine = "DEEPSEEK"
        
        # Create branch
        branch_result = command_handler._handle_branch()
        assert branch_result.success
        branch_id = list(command_handler.branch_manager.branches.keys())[1]  # Not main
        
        # Swap to GPT-5-mini on main
        with patch('oumi.builders.inference_engines.build_inference_engine'):
            swap_result = command_handler._handle_swap("gpt-5-mini")
            assert swap_result.success
            command_handler.config.engine = "OPENAI"
        
        # Swap to Claude Opus on main
        with patch('oumi.builders.inference_engines.build_inference_engine'):
            swap_result = command_handler._handle_swap("claude-opus-4-1-20250805")
            assert swap_result.success
            command_handler.config.engine = "ANTHROPIC"
        
        # Switch back to branch - should restore DeepSeek, not show Claude
        with patch('oumi.builders.inference_engines.build_inference_engine') as mock_build:
            mock_deepseek_engine = MagicMock()
            mock_deepseek_engine.get_model_name.return_value = "deepseek-r1"
            mock_build.return_value = mock_deepseek_engine
            
            switch_result = command_handler._handle_branch_switch(branch_id)
            assert switch_result.success
            
            # Should have DeepSeek model, not Claude
            assert command_handler.config.model.model_name == "deepseek-r1"
            assert command_handler.config.engine == "DEEPSEEK"
            
            # Should have rebuilt engine since models differ
            mock_build.assert_called_once()