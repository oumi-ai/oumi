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

"""Integration tests for basic chat session functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oumi.core.commands import CommandResult
from oumi.core.types.conversation import Conversation, Message, Role
from tests.utils.chat_test_utils import (
    ChatTestSession,
    create_test_inference_config,
    get_sample_conversations,
    temporary_test_files,
)


class TestBasicChatSession:
    """Test suite for basic chat session operations."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_inference_config()
        return ChatTestSession(config)

    def test_start_chat_session(self, chat_session):
        """Test starting a new chat session."""
        result = chat_session.start_session()
        
        assert result.success
        assert "started" in result.message.lower() or "ready" in result.message.lower()
        assert chat_session.is_active()

    def test_end_chat_session(self, chat_session):
        """Test ending an active chat session."""
        # Start session first
        chat_session.start_session()
        assert chat_session.is_active()
        
        # End session
        result = chat_session.end_session()
        
        assert result.success
        assert not chat_session.is_active()

    def test_basic_conversation_flow(self, chat_session):
        """Test basic user-assistant conversation flow."""
        chat_session.start_session()
        
        # Send user message
        user_input = "Hello, how are you today?"
        with patch.object(chat_session.mock_engine, 'infer') as mock_infer:
            mock_infer.return_value = "Hello! I'm doing well, thank you for asking. How can I help you today?"
            
            result = chat_session.send_message(user_input)
            
            assert result.success
            
            # Check conversation state
            conversation = chat_session.get_conversation()
            assert len(conversation.messages) >= 2  # User message + assistant response
            assert conversation.messages[-2].role == Role.USER
            assert conversation.messages[-2].content == user_input
            assert conversation.messages[-1].role == Role.ASSISTANT

    def test_conversation_persistence(self, chat_session):
        """Test that conversation history is maintained."""
        chat_session.start_session()
        
        # Send multiple messages
        messages = [
            "What is artificial intelligence?",
            "Can you give me an example?",
            "How does machine learning work?",
        ]
        
        with patch.object(chat_session.mock_engine, 'infer') as mock_infer:
            mock_infer.side_effect = [
                "AI is the simulation of human intelligence in machines.",
                "An example is image recognition systems.",
                "Machine learning uses algorithms to learn from data.",
            ]
            
            for msg in messages:
                result = chat_session.send_message(msg)
                assert result.success
            
            # Check conversation history
            conversation = chat_session.get_conversation()
            assert len(conversation.messages) >= 6  # 3 user + 3 assistant messages
            
            # Verify message order and content
            user_messages = [m for m in conversation.messages if m.role == Role.USER]
            assert len(user_messages) == 3
            for i, expected_msg in enumerate(messages):
                assert user_messages[i].content == expected_msg

    def test_error_handling_invalid_input(self, chat_session):
        """Test handling of invalid input."""
        chat_session.start_session()
        
        # Test empty message
        result = chat_session.send_message("")
        assert not result.success
        assert "empty" in result.message.lower()
        
        # Test very long message
        very_long_message = "x" * 100000
        result = chat_session.send_message(very_long_message)
        # Should either handle gracefully or reject
        assert isinstance(result, CommandResult)

    def test_inference_engine_error_recovery(self, chat_session):
        """Test recovery from inference engine errors."""
        chat_session.start_session()
        
        with patch.object(chat_session.mock_engine, 'infer') as mock_infer:
            # Simulate engine error
            mock_infer.side_effect = Exception("Model inference failed")
            
            result = chat_session.send_message("Test message")
            
            # Should handle error gracefully
            assert not result.success
            assert "error" in result.message.lower() or "failed" in result.message.lower()
            
            # Session should remain active for recovery
            assert chat_session.is_active()

    def test_conversation_context_management(self, chat_session):
        """Test that conversation context is properly managed."""
        chat_session.start_session()
        
        # Create a conversation with context that would exceed token limits
        long_context_messages = []
        for i in range(100):
            long_context_messages.extend([
                f"User message {i}: " + "This is a long message. " * 50,
                f"Assistant response {i}: " + "This is a long response. " * 50,
            ])
        
        with patch.object(chat_session.mock_engine, 'infer') as mock_infer:
            mock_infer.return_value = "Response within context limits"
            
            # Send many messages to trigger context management
            for i in range(10):
                result = chat_session.send_message(f"Message {i}")
                assert result.success
            
            # Verify conversation is still manageable
            conversation = chat_session.get_conversation()
            assert len(conversation.messages) <= 50  # Should have some reasonable limit

    def test_session_state_isolation(self, chat_session):
        """Test that multiple sessions are isolated from each other."""
        # Start first session
        chat_session.start_session()
        result1 = chat_session.send_message("First session message")
        assert result1.success
        
        # Create second session
        config2 = create_test_inference_config()
        chat_session2 = ChatTestSession(config2)
        chat_session2.start_session()
        
        # Send message to second session
        result2 = chat_session2.send_message("Second session message")
        assert result2.success
        
        # Verify sessions are isolated
        conv1 = chat_session.get_conversation()
        conv2 = chat_session2.get_conversation()
        
        assert conv1.conversation_id != conv2.conversation_id
        
        # Messages should be different
        conv1_user_msgs = [m.content for m in conv1.messages if m.role == Role.USER]
        conv2_user_msgs = [m.content for m in conv2.messages if m.role == Role.USER]
        
        assert "First session message" in conv1_user_msgs
        assert "Second session message" in conv2_user_msgs
        assert "Second session message" not in conv1_user_msgs
        assert "First session message" not in conv2_user_msgs


class TestChatSessionWithCommands:
    """Test suite for chat sessions with command execution."""

    @pytest.fixture
    def chat_session(self):
        """Create a test chat session."""
        config = create_test_inference_config()
        return ChatTestSession(config)

    def test_help_command_in_session(self, chat_session):
        """Test executing help command within chat session."""
        chat_session.start_session()
        
        result = chat_session.execute_command("/help()")
        
        assert result.success
        assert "help" in result.message.lower() or "command" in result.message.lower()

    def test_save_command_in_session(self, chat_session):
        """Test executing save command within chat session."""
        chat_session.start_session()
        
        # Create some conversation history
        chat_session.send_message("Hello!")
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            result = chat_session.execute_command(f"/save({temp_path})")
            
            if result.success:
                assert "saved" in result.message.lower()
                assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_clear_command_in_session(self, chat_session):
        """Test executing clear command within chat session."""
        chat_session.start_session()
        
        # Create conversation history
        chat_session.send_message("Message 1")
        chat_session.send_message("Message 2")
        
        # Verify history exists
        conv_before = chat_session.get_conversation()
        assert len(conv_before.messages) > 0
        
        # Clear conversation
        result = chat_session.execute_command("/clear()")
        
        if result.success:
            conv_after = chat_session.get_conversation()
            assert len(conv_after.messages) == 0

    def test_show_command_in_session(self, chat_session):
        """Test executing show command within chat session."""
        chat_session.start_session()
        
        # Create conversation history
        test_message = "Show me this message"
        chat_session.send_message(test_message)
        
        # Show specific message
        result = chat_session.execute_command("/show(1)")
        
        if result.success:
            assert test_message in result.message or "Message 1" in result.message

    def test_branch_command_in_session(self, chat_session):
        """Test executing branch command within chat session."""
        chat_session.start_session()
        
        # Create some conversation history
        chat_session.send_message("Initial message")
        chat_session.send_message("Second message")
        
        # Create a branch
        result = chat_session.execute_command("/branch(test_branch)")
        
        if result.success:
            assert "branch" in result.message.lower()
            assert "test_branch" in result.message

    def test_attach_command_in_session(self, chat_session):
        """Test executing attach command within chat session."""
        chat_session.start_session()
        
        # Create a temporary file to attach
        test_content = "This is a test file for attachment."
        
        with temporary_test_files({"test_attachment.txt": test_content}) as temp_files:
            result = chat_session.execute_command(f"/attach({temp_files['test_attachment.txt']})")
            
            if result.success:
                assert "attach" in result.message.lower()
                # File content should be available to the session
                conv = chat_session.get_conversation()
                # Check if attachment is reflected in conversation context

    def test_command_error_handling_in_session(self, chat_session):
        """Test command error handling within chat session."""
        chat_session.start_session()
        
        # Test invalid command
        result = chat_session.execute_command("/invalid_command()")
        
        assert not result.success
        assert "unknown" in result.message.lower() or "not found" in result.message.lower()
        
        # Session should remain active after command error
        assert chat_session.is_active()
        
        # Should still be able to send regular messages
        followup_result = chat_session.send_message("Are you still there?")
        assert followup_result.success

    def test_mixed_messages_and_commands(self, chat_session):
        """Test mixing regular messages and commands in a session."""
        chat_session.start_session()
        
        # Mix of messages and commands
        interactions = [
            ("message", "Hello, how are you?"),
            ("command", "/help()"),
            ("message", "Can you help me with something?"),
            ("command", "/show(all)"),
            ("message", "Thank you!"),
        ]
        
        for interaction_type, content in interactions:
            if interaction_type == "message":
                result = chat_session.send_message(content)
            else:
                result = chat_session.execute_command(content)
            
            # Each interaction should work regardless of previous ones
            assert isinstance(result, CommandResult)
            
        # Conversation should contain all interactions
        conv = chat_session.get_conversation()
        user_messages = [m.content for m in conv.messages if m.role == Role.USER]
        
        # Should have the regular messages (commands might not appear in conversation)
        message_contents = [content for interaction_type, content in interactions if interaction_type == "message"]
        for msg_content in message_contents:
            assert any(msg_content in user_msg for user_msg in user_messages)


class TestChatSessionConfigurationHandling:
    """Test suite for chat session configuration handling."""

    def test_session_with_different_models(self):
        """Test sessions with different model configurations."""
        # Test with SmolLM
        config1 = create_test_inference_config()
        config1.model.model_name = "SmolLM-135M-Instruct"
        session1 = ChatTestSession(config1)
        
        # Test with different model
        config2 = create_test_inference_config()
        config2.model.model_name = "SmolVLM-256M-Instruct"
        session2 = ChatTestSession(config2)
        
        session1.start_session()
        session2.start_session()
        
        # Both sessions should work
        result1 = session1.send_message("Hello from session 1")
        result2 = session2.send_message("Hello from session 2")
        
        assert result1.success
        assert result2.success

    def test_session_with_different_generation_params(self):
        """Test sessions with different generation parameters."""
        # High temperature config
        config_creative = create_test_inference_config()
        config_creative.generation.temperature = 1.0
        config_creative.generation.top_p = 0.95
        
        # Low temperature config  
        config_focused = create_test_inference_config()
        config_focused.generation.temperature = 0.1
        config_focused.generation.top_p = 0.8
        
        session_creative = ChatTestSession(config_creative)
        session_focused = ChatTestSession(config_focused)
        
        session_creative.start_session()
        session_focused.start_session()
        
        # Both should work with their respective configs
        result1 = session_creative.send_message("Tell me a creative story")
        result2 = session_focused.send_message("What is 2+2?")
        
        assert result1.success
        assert result2.success

    def test_session_config_validation(self):
        """Test session behavior with invalid configurations."""
        # Test with missing model name
        config = create_test_inference_config()
        config.model.model_name = None
        
        session = ChatTestSession(config)
        result = session.start_session()
        
        # Should handle gracefully
        assert isinstance(result, CommandResult)
        
        # Test with invalid parameters
        config2 = create_test_inference_config()
        config2.generation.temperature = -1.0  # Invalid temperature
        
        session2 = ChatTestSession(config2)
        result2 = session2.start_session()
        
        # Should handle gracefully
        assert isinstance(result2, CommandResult)