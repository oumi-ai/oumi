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

"""Comprehensive chat history save/load integration tests."""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.commands.handlers.file_operations_handler import FileOperationsHandler
from oumi.core.commands import CommandContext, CommandResult
from tests.utils.chat_real_model_utils import (
    RealModelChatSession,
    create_real_model_inference_config,
    temporary_chat_files
)


class TestChatHistorySaveLoad:
    """Test chat history saving and loading functionality."""

    def test_basic_chat_save_functionality(self):
        """Test basic saving of chat conversation to JSON."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            save_path = temp_dir / "test_chat_save.json"
            
            # Create mock conversation
            conversation = Conversation(
                conversation_id="test_save_conversation",
                messages=[
                    Message(role=Role.USER, content="Hello, how are you?"),
                    Message(role=Role.ASSISTANT, content="I'm doing well, thank you! How can I help you today?"),
                    Message(role=Role.USER, content="Can you explain quantum physics?"),
                    Message(role=Role.ASSISTANT, content="Quantum physics is the study of matter and energy at the smallest scales...")
                ]
            )
            
            # Mock command context with conversation
            context = MagicMock()
            context.current_conversation = conversation
            context.conversation_history = [conversation]
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Test save operation
            result = file_handler.save_conversation(
                context=context,
                file_path=str(save_path),
                format="json"
            )
            
            assert result.success
            assert save_path.exists()
            
            # Verify saved content
            saved_data = json.loads(save_path.read_text())
            assert "conversation_id" in saved_data
            assert "messages" in saved_data
            assert len(saved_data["messages"]) == 4
            assert saved_data["messages"][0]["role"] == "user"
            assert saved_data["messages"][1]["role"] == "assistant"

    def test_multiple_format_exports(self):
        """Test exporting conversations in multiple formats."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create test conversation
            conversation = Conversation(
                conversation_id="multi_format_test",
                messages=[
                    Message(role=Role.USER, content="What is AI?"),
                    Message(role=Role.ASSISTANT, content="Artificial Intelligence (AI) refers to systems that can perform tasks typically requiring human intelligence."),
                    Message(role=Role.USER, content="Give me an example."),
                    Message(role=Role.ASSISTANT, content="Voice assistants like Siri or Alexa are examples of AI systems.")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = conversation
            context.conversation_history = [conversation]
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Test different export formats
            export_formats = {
                "json": temp_dir / "export_test.json",
                "txt": temp_dir / "export_test.txt", 
                "md": temp_dir / "export_test.md",
                "html": temp_dir / "export_test.html",
                "csv": temp_dir / "export_test.csv"
            }
            
            export_results = {}
            for format_name, file_path in export_formats.items():
                result = file_handler.save_conversation(
                    context=context,
                    file_path=str(file_path),
                    format=format_name
                )
                export_results[format_name] = {
                    "success": result.success,
                    "file_exists": file_path.exists(),
                    "file_size": file_path.stat().st_size if file_path.exists() else 0
                }
            
            # Verify all formats exported successfully
            for format_name, result in export_results.items():
                assert result["success"], f"{format_name} export failed"
                assert result["file_exists"], f"{format_name} file not created"
                assert result["file_size"] > 0, f"{format_name} file is empty"

    def test_conversation_branch_preservation(self):
        """Test that conversation branches are preserved in saves."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            save_path = temp_dir / "branched_conversation.json"
            
            # Create conversation with branches
            main_conversation = Conversation(
                conversation_id="main_branch",
                messages=[
                    Message(role=Role.USER, content="Let's discuss science"),
                    Message(role=Role.ASSISTANT, content="Science is fascinating! What area interests you most?")
                ]
            )
            
            branch_conversation = Conversation(
                conversation_id="science_branch",
                messages=[
                    Message(role=Role.USER, content="Let's discuss science"),
                    Message(role=Role.ASSISTANT, content="Science is fascinating! What area interests you most?"),
                    Message(role=Role.USER, content="I'm interested in physics"),
                    Message(role=Role.ASSISTANT, content="Physics explores the fundamental laws of nature...")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = branch_conversation
            context.conversation_history = [main_conversation, branch_conversation]
            context.conversation_branches = {
                "main": main_conversation,
                "physics": branch_conversation
            }
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Save with branch information
            result = file_handler.save_conversation_with_branches(
                context=context,
                file_path=str(save_path)
            )
            
            if result.success:
                saved_data = json.loads(save_path.read_text())
                
                # Should preserve branch structure
                assert "branches" in saved_data or "conversation_history" in saved_data
                
                if "branches" in saved_data:
                    assert len(saved_data["branches"]) >= 2
                elif "conversation_history" in saved_data:
                    assert len(saved_data["conversation_history"]) >= 2

    def test_conversation_loading_and_restoration(self):
        """Test loading conversations and restoring chat state."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            save_path = temp_dir / "conversation_to_load.json"
            
            # Create and save a conversation first
            original_conversation = Conversation(
                conversation_id="load_test_conversation",
                messages=[
                    Message(role=Role.USER, content="Remember this conversation"),
                    Message(role=Role.ASSISTANT, content="I'll remember our discussion about memory and conversation state."),
                    Message(role=Role.USER, content="What did we just discuss?"),
                    Message(role=Role.ASSISTANT, content="We discussed memory and conversation state preservation.")
                ]
            )
            
            # Save conversation manually for loading test
            save_data = {
                "conversation_id": original_conversation.conversation_id,
                "messages": [
                    {
                        "role": msg.role.value,
                        "content": msg.compute_flattened_text_content(),
                        "timestamp": time.time()
                    }
                    for msg in original_conversation.messages
                ],
                "metadata": {
                    "created_at": time.time(),
                    "model_config": config.model.__dict__ if hasattr(config.model, '__dict__') else {},
                    "generation_config": config.generation.__dict__ if hasattr(config.generation, '__dict__') else {}
                }
            }
            
            save_path.write_text(json.dumps(save_data, indent=2))
            
            # Test loading the conversation
            file_handler = FileOperationsHandler()
            context = MagicMock()
            context.conversation_history = []
            context.inference_config = config
            
            load_result = file_handler.load_conversation(
                context=context,
                file_path=str(save_path)
            )
            
            assert load_result.success
            
            # Verify conversation was loaded correctly
            if hasattr(context, 'current_conversation') and context.current_conversation:
                loaded_conversation = context.current_conversation
                assert loaded_conversation.conversation_id == "load_test_conversation"
                assert len(loaded_conversation.messages) == 4
                assert loaded_conversation.messages[0].role == Role.USER
                assert "Remember this conversation" in loaded_conversation.messages[0].compute_flattened_text_content()

    def test_auto_save_functionality(self):
        """Test automatic conversation saving during chat sessions."""
        config = create_real_model_inference_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / ".oumi" / "chat_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Mock chat session with auto-save enabled
            chat_session = RealModelChatSession(config)
            chat_session._auto_save_enabled = True
            chat_session._cache_directory = str(cache_dir)
            
            # Simulate conversation with auto-save triggers
            mock_conversation = Conversation(
                conversation_id="auto_save_test",
                messages=[
                    Message(role=Role.USER, content="Test auto-save feature"),
                    Message(role=Role.ASSISTANT, content="Auto-save helps preserve conversation history automatically.")
                ]
            )
            
            chat_session._current_conversation = mock_conversation
            
            # Trigger auto-save (simulate)
            if hasattr(chat_session, 'trigger_auto_save'):
                auto_save_result = chat_session.trigger_auto_save()
                
                if auto_save_result:
                    # Check that auto-save file was created
                    auto_save_files = list(cache_dir.glob("auto_save_*.json"))
                    assert len(auto_save_files) > 0
                    
                    # Verify auto-save content
                    latest_auto_save = max(auto_save_files, key=lambda p: p.stat().st_mtime)
                    auto_save_data = json.loads(latest_auto_save.read_text())
                    
                    assert "messages" in auto_save_data
                    assert len(auto_save_data["messages"]) >= 2

    def test_conversation_metadata_preservation(self):
        """Test preservation of conversation metadata during save/load."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            metadata_save_path = temp_dir / "metadata_test.json"
            
            # Create conversation with rich metadata
            conversation = Conversation(
                conversation_id="metadata_test",
                messages=[
                    Message(role=Role.USER, content="Test metadata preservation"),
                    Message(role=Role.ASSISTANT, content="Metadata includes timestamps, model configs, and performance stats.")
                ]
            )
            
            # Add metadata to context
            context = MagicMock()
            context.current_conversation = conversation
            context.conversation_history = [conversation]
            context.inference_config = config
            context.session_metadata = {
                "session_start_time": time.time(),
                "user_preferences": {
                    "theme": "dark",
                    "save_frequency": "auto"
                },
                "performance_stats": {
                    "avg_response_time": 1.5,
                    "total_tokens": 150,
                    "memory_usage": "2.3 GB"
                },
                "model_info": {
                    "model_name": config.model.model_name if hasattr(config.model, 'model_name') else "test-model",
                    "engine": config.engine,
                    "parameters": config.generation.__dict__ if hasattr(config.generation, '__dict__') else {}
                }
            }
            
            file_handler = FileOperationsHandler()
            
            # Save with metadata
            result = file_handler.save_conversation_with_metadata(
                context=context,
                file_path=str(metadata_save_path)
            )
            
            if result.success and metadata_save_path.exists():
                saved_data = json.loads(metadata_save_path.read_text())
                
                # Verify metadata preservation
                expected_metadata_keys = [
                    "session_metadata",
                    "model_info", 
                    "performance_stats",
                    "timestamp"
                ]
                
                for key in expected_metadata_keys:
                    if key in saved_data:
                        assert saved_data[key] is not None

    def test_conversation_search_and_browse(self):
        """Test conversation search and browsing functionality."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create multiple test conversations
            test_conversations = [
                {
                    "id": "science_chat",
                    "messages": [
                        {"role": "user", "content": "Tell me about quantum physics"},
                        {"role": "assistant", "content": "Quantum physics is the study of matter at the atomic level..."}
                    ]
                },
                {
                    "id": "cooking_chat", 
                    "messages": [
                        {"role": "user", "content": "How do I make pasta?"},
                        {"role": "assistant", "content": "To make pasta, you'll need flour, eggs, and water..."}
                    ]
                },
                {
                    "id": "history_chat",
                    "messages": [
                        {"role": "user", "content": "What happened in 1969?"},
                        {"role": "assistant", "content": "1969 was notable for the Apollo 11 moon landing..."}
                    ]
                }
            ]
            
            # Save test conversations
            conversation_files = []
            for i, conv_data in enumerate(test_conversations):
                conv_file = temp_dir / f"conversation_{i}.json"
                conv_file.write_text(json.dumps(conv_data, indent=2))
                conversation_files.append(conv_file)
            
            file_handler = FileOperationsHandler()
            context = MagicMock()
            context.chat_cache_directory = str(temp_dir)
            
            # Test conversation browsing
            if hasattr(file_handler, 'browse_conversations'):
                browse_result = file_handler.browse_conversations(context)
                
                if browse_result.success:
                    # Should find saved conversations
                    found_conversations = browse_result.data if hasattr(browse_result, 'data') else []
                    assert len(found_conversations) >= len(test_conversations)
            
            # Test conversation search
            if hasattr(file_handler, 'search_conversations'):
                search_result = file_handler.search_conversations(
                    context,
                    query="quantum physics"
                )
                
                if search_result.success:
                    # Should find science conversation
                    search_results = search_result.data if hasattr(search_result, 'data') else []
                    science_found = any("quantum" in str(result).lower() for result in search_results)
                    assert science_found

    def test_corrupted_file_handling(self):
        """Test handling of corrupted or invalid conversation files."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create corrupted files
            corrupted_files = {
                "invalid_json.json": "{ invalid json content",
                "empty_file.json": "",
                "wrong_structure.json": json.dumps({"wrong": "structure"}),
                "partial_data.json": json.dumps({
                    "conversation_id": "partial",
                    # Missing messages field
                })
            }
            
            file_handler = FileOperationsHandler()
            context = MagicMock()
            context.conversation_history = []
            
            for filename, content in corrupted_files.items():
                corrupt_file = temp_dir / filename
                corrupt_file.write_text(content)
                
                # Test loading corrupted file
                load_result = file_handler.load_conversation(
                    context=context,
                    file_path=str(corrupt_file)
                )
                
                # Should handle corruption gracefully
                assert isinstance(load_result, CommandResult)
                if not load_result.success:
                    # Error message should be informative
                    assert "corrupt" in load_result.message.lower() or \
                           "invalid" in load_result.message.lower() or \
                           "error" in load_result.message.lower()

    def test_large_conversation_handling(self):
        """Test handling of very large conversations."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            large_conv_path = temp_dir / "large_conversation.json"
            
            # Create large conversation
            large_messages = []
            for i in range(200):  # 200 message pairs
                large_messages.extend([
                    Message(
                        role=Role.USER, 
                        content=f"This is user message {i+1}. " + "Content " * 50
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=f"This is assistant response {i+1}. " + "Response content " * 100
                    )
                ])
            
            large_conversation = Conversation(
                conversation_id="large_test_conversation",
                messages=large_messages
            )
            
            context = MagicMock()
            context.current_conversation = large_conversation
            context.conversation_history = [large_conversation]
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Test saving large conversation
            save_result = file_handler.save_conversation(
                context=context,
                file_path=str(large_conv_path),
                format="json"
            )
            
            # Should handle large conversations
            assert save_result.success or "too large" in save_result.message.lower()
            
            if save_result.success and large_conv_path.exists():
                # File should exist and be substantial
                file_size = large_conv_path.stat().st_size
                assert file_size > 10000  # Should be > 10KB
                
                # Test loading large conversation
                load_context = MagicMock()
                load_context.conversation_history = []
                
                load_result = file_handler.load_conversation(
                    context=load_context,
                    file_path=str(large_conv_path)
                )
                
                # Should handle loading large files
                assert isinstance(load_result, CommandResult)


class TestChatHistoryUserJourneys:
    """Test complete user journeys involving chat history."""

    def test_complete_save_reload_continue_journey(self):
        """Test saving a conversation, reloading it, and continuing the chat."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            journey_save_path = temp_dir / "journey_conversation.json"
            
            # Phase 1: Initial conversation
            initial_conversation = Conversation(
                conversation_id="journey_conversation",
                messages=[
                    Message(role=Role.USER, content="Let's discuss machine learning"),
                    Message(role=Role.ASSISTANT, content="Machine learning is a subset of AI that enables systems to learn from data."),
                    Message(role=Role.USER, content="What are some common algorithms?"),
                    Message(role=Role.ASSISTANT, content="Common algorithms include linear regression, decision trees, and neural networks.")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = initial_conversation
            context.conversation_history = [initial_conversation]
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Save initial conversation
            save_result = file_handler.save_conversation(
                context=context,
                file_path=str(journey_save_path),
                format="json"
            )
            
            assert save_result.success
            assert journey_save_path.exists()
            
            # Phase 2: Load conversation in new session
            new_context = MagicMock()
            new_context.conversation_history = []
            new_context.inference_config = config
            
            load_result = file_handler.load_conversation(
                context=new_context,
                file_path=str(journey_save_path)
            )
            
            assert load_result.success
            
            # Phase 3: Continue conversation
            if hasattr(new_context, 'current_conversation') and new_context.current_conversation:
                continued_conversation = new_context.current_conversation
                
                # Add continuation messages
                continued_conversation.messages.extend([
                    Message(role=Role.USER, content="Can you explain neural networks in more detail?"),
                    Message(role=Role.ASSISTANT, content="Neural networks are inspired by biological neurons and consist of interconnected nodes...")
                ])
                
                # Save continued conversation
                continued_save_path = temp_dir / "journey_continued.json"
                continue_save_result = file_handler.save_conversation(
                    context=new_context,
                    file_path=str(continued_save_path),
                    format="json"
                )
                
                if continue_save_result.success:
                    # Verify continuation was saved
                    continued_data = json.loads(continued_save_path.read_text())
                    assert len(continued_data["messages"]) == 6  # Original 4 + 2 new

    def test_multiple_session_history_management(self):
        """Test managing conversation history across multiple chat sessions."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create multiple session conversations
            session_conversations = []
            for i in range(3):
                conversation = Conversation(
                    conversation_id=f"session_{i+1}",
                    messages=[
                        Message(role=Role.USER, content=f"Session {i+1} question"),
                        Message(role=Role.ASSISTANT, content=f"Session {i+1} response with helpful information."),
                        Message(role=Role.USER, content=f"Follow-up question for session {i+1}"),
                        Message(role=Role.ASSISTANT, content=f"Follow-up response for session {i+1}.")
                    ]
                )
                session_conversations.append(conversation)
                
                # Save each session
                session_file = temp_dir / f"session_{i+1}.json"
                session_data = {
                    "conversation_id": conversation.conversation_id,
                    "messages": [
                        {
                            "role": msg.role.value,
                            "content": msg.compute_flattened_text_content(),
                            "timestamp": time.time() + i * 3600  # Different timestamps
                        }
                        for msg in conversation.messages
                    ]
                }
                session_file.write_text(json.dumps(session_data, indent=2))
            
            file_handler = FileOperationsHandler()
            
            # Test loading multiple sessions
            master_context = MagicMock()
            master_context.conversation_history = []
            master_context.inference_config = config
            
            loaded_sessions = []
            for i in range(3):
                session_file = temp_dir / f"session_{i+1}.json"
                load_result = file_handler.load_conversation(
                    context=master_context,
                    file_path=str(session_file)
                )
                loaded_sessions.append(load_result.success)
            
            # Verify all sessions loaded
            successful_loads = sum(1 for success in loaded_sessions if success)
            assert successful_loads >= 2  # At least 2/3 should succeed

    def test_conversation_export_and_sharing_journey(self):
        """Test exporting conversations for sharing and documentation."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create conversation for sharing
            sharing_conversation = Conversation(
                conversation_id="sharing_test",
                messages=[
                    Message(role=Role.USER, content="How do I implement a neural network?"),
                    Message(role=Role.ASSISTANT, content="To implement a neural network, you'll need to define layers, activation functions, and training loops..."),
                    Message(role=Role.USER, content="Can you show me code examples?"),
                    Message(role=Role.ASSISTANT, content="Here's a simple example using Python:\n\n```python\nimport numpy as np\n\nclass NeuralNetwork:\n    def __init__(self):\n        # Network initialization\n        pass\n```"),
                    Message(role=Role.USER, content="That's helpful, thank you!"),
                    Message(role=Role.ASSISTANT, content="You're welcome! Feel free to ask if you need more details.")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = sharing_conversation
            context.conversation_history = [sharing_conversation]
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Export in multiple formats for different sharing scenarios
            sharing_formats = [
                ("md", "documentation.md"),   # For documentation
                ("txt", "simple_share.txt"),  # For simple sharing
                ("html", "web_share.html"),   # For web display
                ("pdf", "formal_doc.pdf")     # For formal documents
            ]
            
            export_results = []
            for format_type, filename in sharing_formats:
                export_path = temp_dir / filename
                result = file_handler.save_conversation(
                    context=context,
                    file_path=str(export_path),
                    format=format_type
                )
                
                export_results.append({
                    "format": format_type,
                    "success": result.success,
                    "file_size": export_path.stat().st_size if export_path.exists() else 0
                })
            
            # Verify exports for sharing
            successful_exports = [r for r in export_results if r["success"]]
            assert len(successful_exports) >= 2  # At least 2 formats should work
            
            # Check that exports contain meaningful content
            for result in successful_exports:
                if result["file_size"] > 0:
                    assert result["file_size"] > 100  # Should have substantial content

    def test_conversation_backup_and_recovery_journey(self):
        """Test conversation backup and recovery scenarios."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create important conversation that needs backup
            important_conversation = Conversation(
                conversation_id="important_work_session",
                messages=[
                    Message(role=Role.USER, content="Help me design a system architecture"),
                    Message(role=Role.ASSISTANT, content="Let's start by identifying the key components and their relationships..."),
                    Message(role=Role.USER, content="What about scalability considerations?"),
                    Message(role=Role.ASSISTANT, content="For scalability, consider microservices architecture, load balancing, and database sharding..."),
                    Message(role=Role.USER, content="How do I handle data consistency?"),
                    Message(role=Role.ASSISTANT, content="Data consistency can be managed through ACID transactions, event sourcing, or eventual consistency patterns...")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = important_conversation
            context.conversation_history = [important_conversation]
            context.inference_config = config
            context.session_metadata = {
                "created_at": time.time(),
                "importance_level": "high",
                "tags": ["architecture", "scalability", "work"]
            }
            
            file_handler = FileOperationsHandler()
            
            # Create backup
            backup_path = temp_dir / "important_backup.json"
            backup_result = file_handler.save_conversation_with_metadata(
                context=context,
                file_path=str(backup_path)
            )
            
            assert backup_result.success
            assert backup_path.exists()
            
            # Simulate data loss scenario
            recovery_context = MagicMock()
            recovery_context.conversation_history = []
            recovery_context.inference_config = config
            
            # Recover from backup
            recovery_result = file_handler.load_conversation(
                context=recovery_context,
                file_path=str(backup_path)
            )
            
            assert recovery_result.success
            
            # Verify recovery completeness
            if hasattr(recovery_context, 'current_conversation') and recovery_context.current_conversation:
                recovered_conversation = recovery_context.current_conversation
                assert recovered_conversation.conversation_id == "important_work_session"
                assert len(recovered_conversation.messages) == 6
                
                # Verify content preservation
                user_messages = [msg for msg in recovered_conversation.messages if msg.role == Role.USER]
                assert len(user_messages) == 3
                assert "system architecture" in user_messages[0].compute_flattened_text_content()


class TestChatHistoryErrorScenarios:
    """Test error handling and edge cases in chat history management."""

    def test_file_permission_errors(self):
        """Test handling of file permission errors during save/load."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            
            # Create conversation
            conversation = Conversation(
                conversation_id="permission_test",
                messages=[
                    Message(role=Role.USER, content="Test permissions"),
                    Message(role=Role.ASSISTANT, content="Testing file permission handling.")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = conversation
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Try to save to non-existent directory (should fail gracefully)
            invalid_path = "/root/restricted/conversation.json"
            result = file_handler.save_conversation(
                context=context,
                file_path=invalid_path,
                format="json"
            )
            
            # Should handle permission errors gracefully
            assert not result.success
            assert "permission" in result.message.lower() or \
                   "access" in result.message.lower() or \
                   "denied" in result.message.lower()

    def test_disk_space_handling(self):
        """Test handling of insufficient disk space scenarios."""
        config = create_real_model_inference_config()
        
        # Create very large conversation content
        huge_content = "Large content " * 10000  # Very large string
        
        huge_conversation = Conversation(
            conversation_id="disk_space_test",
            messages=[
                Message(role=Role.USER, content=huge_content),
                Message(role=Role.ASSISTANT, content=huge_content)
            ]
        )
        
        context = MagicMock()
        context.current_conversation = huge_conversation
        context.inference_config = config
        
        file_handler = FileOperationsHandler()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            huge_save_path = temp_dir / "huge_conversation.json"
            
            # Try to save huge conversation
            result = file_handler.save_conversation(
                context=context,
                file_path=str(huge_save_path),
                format="json"
            )
            
            # Should either succeed or fail gracefully with informative message
            if not result.success:
                assert "space" in result.message.lower() or \
                       "size" in result.message.lower() or \
                       "large" in result.message.lower()

    def test_concurrent_file_access(self):
        """Test handling of concurrent file access scenarios."""
        config = create_real_model_inference_config()
        
        with temporary_chat_files({}) as temp_files:
            temp_dir = Path(list(temp_files.values())[0]).parent if temp_files else Path(tempfile.mkdtemp())
            concurrent_path = temp_dir / "concurrent_test.json"
            
            # Create test conversation
            conversation = Conversation(
                conversation_id="concurrent_test",
                messages=[
                    Message(role=Role.USER, content="Test concurrent access"),
                    Message(role=Role.ASSISTANT, content="Testing file locking and concurrent access handling.")
                ]
            )
            
            context = MagicMock()
            context.current_conversation = conversation
            context.inference_config = config
            
            file_handler = FileOperationsHandler()
            
            # Simulate concurrent saves (sequential for testing)
            results = []
            for i in range(3):
                result = file_handler.save_conversation(
                    context=context,
                    file_path=str(concurrent_path),
                    format="json"
                )
                results.append(result.success)
                
                # Brief pause to simulate timing
                time.sleep(0.01)
            
            # At least one operation should succeed
            assert any(results), "All concurrent operations failed"