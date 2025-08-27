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

"""
Comprehensive tests for WebChat branch functionality fixes.

This test suite focuses on the specific branch issues identified in the debug logs:
1. Branch history not being retained from the branch point
2. GUI not showing all active branches  
3. Branch switching not working (branches not clickable)
4. Regeneration button being broken

These tests use verbose debug logging to catch issues that would be missed
in traditional UI-only testing.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from oumi.core.configs import InferenceConfig
from oumi.webchat.server import WebChatSession
from tests.utils.chat_test_utils import create_test_inference_config
from tests.unit.webchat.utils.webchat_test_utils import (
    MockWebSocketClient,
    assert_session_state,
    assert_websocket_message,
    mock_webchat_server,
    wait_for_condition,
)

# Enable verbose debug logging for these tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBranchHistoryRetention:
    """Test that branch operations properly retain conversation history from branch point."""

    def test_branch_creation_preserves_full_conversation_history(self):
        """Test that creating a branch preserves the full conversation up to the branch point."""
        # Create a real WebChatSession for testing branch functionality
        config = create_test_inference_config()
        session = WebChatSession(session_id="history_test", config=config)
        
        # Build up a conversation history like in the debug logs
        conversation_messages = [
            {"role": "user", "content": "Hi Gemma!"},
            {"role": "assistant", "content": "Thank you for asking! 😊"},
            {"role": "user", "content": "How are you today?"},
            {"role": "assistant", "content": "I am doing well, thank you for asking! How are you doing?"},
            {"role": "user", "content": "Which muppet is your favorite?"},
            {"role": "assistant", "content": "That's a tough one! I like Kermit because he's optimistic."}
        ]
        
        # Add messages to session conversation history directly
        for msg in conversation_messages:
            session.conversation_history.append({
                "role": msg["role"], 
                "content": msg["content"], 
                "timestamp": time.time()
            })
        
        # Verify initial conversation length
        assert len(session.conversation_history) == 6
        logger.info(f"🧪 TEST: Initial conversation length: {len(session.conversation_history)}")
        
        # Debug the branch manager state
        main_branch = session.branch_manager.branches.get("main")
        if main_branch:
            logger.info(f"🧪 TEST: Main branch conversation length: {len(main_branch.conversation_history)}")
        else:
            logger.info("🧪 TEST: Main branch not found in branch manager")
        
        # Create branch at message 5 (after "Which muppet is your favorite?") 
        branch_point = 5  # Up to but not including the 5th message (0-indexed)
        logger.info(f"🧪 TEST: Attempting to create branch at point {branch_point}")
        success, message, new_branch = session.branch_manager.create_branch(
            from_branch_id="main",
            name="muppet_discussion", 
            branch_point=branch_point
        )
        
        assert success, f"Branch creation failed: {message}"
        assert new_branch is not None
        logger.info(f"🧪 TEST: Branch created - {new_branch.id}")
        
        # Verify branch preserves history up to branch point
        expected_branch_history_length = branch_point  # [:branch_point] excludes the message at branch_point
        actual_branch_history_length = len(new_branch.conversation_history)
        
        logger.info(f"🧪 TEST: Expected branch history length: {expected_branch_history_length}")
        logger.info(f"🧪 TEST: Actual branch history length: {actual_branch_history_length}")
        
        # Log each message in the branch for debugging
        for i, msg in enumerate(new_branch.conversation_history):
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))[:50]
            logger.info(f"🧪 TEST: Branch Message {i}: [{role}] {content}...")
        
        assert actual_branch_history_length == expected_branch_history_length, \
            f"Branch should preserve {expected_branch_history_length} messages, but got {actual_branch_history_length}"
        
        # Verify the specific messages are preserved correctly
        for i in range(expected_branch_history_length):
            original_msg = session.conversation_history[i] 
            branch_msg = new_branch.conversation_history[i]
            assert branch_msg["role"] == original_msg["role"], \
                f"Message {i} role mismatch: {branch_msg['role']} != {original_msg['role']}"
            assert branch_msg["content"] == original_msg["content"], \
                f"Message {i} content mismatch: {branch_msg['content']} != {original_msg['content']}"

    def test_branch_switch_preserves_history_from_branch_point(self):
        """Test that switching to a branch preserves the conversation history from the branch point."""
        with mock_webchat_server() as server:
            session_id = server.create_session("switch_test")
            session = server.get_session(session_id)
            
            # Create conversation like in debug logs
            original_messages = [
                {"role": "user", "content": "Hi Gemma!"},
                {"role": "assistant", "content": "Thank you for asking! 😊"},
                {"role": "user", "content": "How are you today?"},
                {"role": "assistant", "content": "I am doing well, thank you for asking! How are you doing?"},
                {"role": "user", "content": "Which muppet is your favorite?"},
                {"role": "assistant", "content": "That's a tough one! I like Kermit."},
            ]
            
            # Add messages to main branch
            for msg in original_messages:
                session.add_message(msg["role"], msg["content"])
            
            logger.info(f"🧪 TEST: Main branch conversation length: {len(session.conversation_history)}")
            assert len(session.conversation_history) == 6
            
            # Create branch from message 4 ("Which muppet is your favorite?")
            success, message, feature_branch = session.branch_manager.create_branch(
                from_branch_id="main",
                name="feature_branch",
                branch_point=4
            )
            assert success, f"Branch creation failed: {message}"
            
            # Add additional messages to feature branch
            session.branch_manager.switch_branch("feature_branch")
            session.add_message("user", "Actually, tell me more about Kermit")
            session.add_message("assistant", "Kermit is the main character and leader of the Muppets.")
            
            feature_length_before_switch = len(session.conversation_history)
            logger.info(f"🧪 TEST: Feature branch length before switching back: {feature_length_before_switch}")
            
            # Switch back to main branch and verify no history loss
            success, message, main_branch = session.branch_manager.switch_branch("main")
            assert success, f"Switch to main failed: {message}"
            
            main_length_after_switch = len(session.conversation_history)
            logger.info(f"🧪 TEST: Main branch length after switch: {main_length_after_switch}")
            
            # Main should still have all 6 original messages
            assert main_length_after_switch == 6, \
                f"Main branch should have 6 messages, but has {main_length_after_switch}"
            
            # Switch back to feature branch and verify it has the full history from branch point + new messages
            success, message, feature_branch = session.branch_manager.switch_branch("feature_branch") 
            assert success, f"Switch to feature failed: {message}"
            
            feature_length_after_switch = len(session.conversation_history)
            logger.info(f"🧪 TEST: Feature branch length after switch back: {feature_length_after_switch}")
            
            # Feature should have: branch point messages (5) + new messages (2) = 7
            expected_feature_length = 7  # 5 from branch point + 2 new
            assert feature_length_after_switch == expected_feature_length, \
                f"Feature branch should have {expected_feature_length} messages, but has {feature_length_after_switch}"
            
            # Verify first 5 messages match the original conversation up to branch point
            for i in range(5):
                original_msg = original_messages[i]
                feature_msg = session.conversation_history[i]
                assert feature_msg["role"] == original_msg["role"], \
                    f"Feature branch message {i} role mismatch after switch"
                assert feature_msg["content"] == original_msg["content"], \
                    f"Feature branch message {i} content mismatch after switch"


class TestBranchListingAndDisplay:
    """Test that branch listing shows all active branches correctly."""

    def test_rest_api_returns_all_active_branches(self):
        """Test that the REST API returns all active branches."""
        with mock_webchat_server() as server:
            session_id = server.create_session("listing_test")
            session = server.get_session(session_id)
            
            # Create multiple branches
            branch_names = ["feature_1", "feature_2", "hotfix_1", "experiment_1"]
            created_branches = []
            
            for name in branch_names:
                success, message, branch = session.branch_manager.create_branch(
                    from_branch_id="main",
                    name=name
                )
                assert success, f"Failed to create branch {name}: {message}"
                created_branches.append(branch.branch_id)
                logger.info(f"🧪 TEST: Created branch {branch.branch_id}")
            
            # Get branches via REST API (simulating frontend request)
            response = server.handle_rest_request("GET", f"/branches?session_id={session_id}")
            
            assert "branches" in response
            assert "current_branch" in response
            
            returned_branches = [b["branch_id"] for b in response["branches"]]
            logger.info(f"🧪 TEST: Returned branches: {returned_branches}")
            logger.info(f"🧪 TEST: Created branches: {created_branches}")
            
            # Should include main branch + all created branches
            expected_branches = ["main"] + created_branches
            
            for expected_branch in expected_branches:
                assert expected_branch in returned_branches, \
                    f"Branch {expected_branch} missing from API response. Got: {returned_branches}"
            
            # Should have correct count
            assert len(returned_branches) == len(expected_branches), \
                f"Expected {len(expected_branches)} branches, got {len(returned_branches)}"

    @pytest.mark.asyncio
    async def test_websocket_branch_updates(self):
        """Test that WebSocket properly sends branch updates."""
        with mock_webchat_server() as server:
            session_id = server.create_session("websocket_test")
            session = server.get_session(session_id)
            
            client = MockWebSocketClient()
            await client.connect()
            
            # Mock the server's branch update response
            initial_branches = session.branch_manager.list_branches()
            branch_update_response = {
                "type": "branches_update",
                "branches": [{"branch_id": b.branch_id, "created_at": b.created_at} for b in initial_branches],
                "current_branch": session.branch_manager.current_branch_id,
            }
            client.add_auto_response("get_branches", branch_update_response)
            
            # Request branch update
            request_message = {"type": "get_branches", "session_id": session_id}
            response = await client.send_message(request_message)
            
            assert response["type"] == "branches_update"
            assert "branches" in response
            assert "current_branch" in response
            
            # Create a new branch and verify it appears in updates
            session.branch_manager.create_branch(from_branch_id="main", name="new_branch")
            updated_branches = session.branch_manager.list_branches()
            
            updated_response = {
                "type": "branches_update", 
                "branches": [{"branch_id": b.branch_id, "created_at": b.created_at} for b in updated_branches],
                "current_branch": session.branch_manager.current_branch_id,
            }
            client.add_auto_response("get_branches", updated_response)
            
            response = await client.send_message(request_message)
            returned_branch_ids = [b["branch_id"] for b in response["branches"]]
            
            assert "new_branch" in returned_branch_ids, \
                f"New branch not in update. Got: {returned_branch_ids}"


class TestBranchSwitching:
    """Test branch switching functionality."""

    def test_rest_api_branch_switching(self):
        """Test branch switching through REST API."""
        with mock_webchat_server() as server:
            session_id = server.create_session("switching_test")
            session = server.get_session(session_id)
            
            # Setup conversation and create branch
            session.add_message("user", "Test message 1")
            session.add_message("assistant", "Test response 1")
            
            success, message, feature_branch = session.branch_manager.create_branch(
                from_branch_id="main",
                name="test_branch"
            )
            assert success, f"Branch creation failed: {message}"
            
            # Add message to feature branch
            session.add_message("user", "Feature message")
            session.add_message("assistant", "Feature response")
            
            initial_feature_length = len(session.conversation_history)
            logger.info(f"🧪 TEST: Feature branch conversation length: {initial_feature_length}")
            
            # Switch back to main via REST API
            switch_data = {
                "action": "switch",
                "branch_id": "main",
                "session_id": session_id
            }
            
            response = server.handle_rest_request("POST", "/branches", switch_data)
            
            assert response["success"] is True, f"Branch switch failed: {response.get('message')}"
            assert response["current_branch"] == "main"
            
            # Verify conversation was switched correctly
            main_length = len(session.conversation_history)
            logger.info(f"🧪 TEST: Main branch conversation length after switch: {main_length}")
            
            # Main should have 2 messages (the original conversation)
            assert main_length == 2, f"Expected 2 messages in main, got {main_length}"
            
            # Switch back to feature branch
            switch_data["branch_id"] = "test_branch"
            response = server.handle_rest_request("POST", "/branches", switch_data)
            
            assert response["success"] is True
            assert response["current_branch"] == "test_branch"
            
            # Verify feature branch conversation restored
            restored_feature_length = len(session.conversation_history)
            logger.info(f"🧪 TEST: Feature branch conversation length after restore: {restored_feature_length}")
            
            assert restored_feature_length == initial_feature_length, \
                f"Feature branch conversation not properly restored: {restored_feature_length} != {initial_feature_length}"

    @pytest.mark.asyncio
    async def test_websocket_branch_switching(self):
        """Test branch switching through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock successful switch response
        switch_response = {
            "type": "branch_switched",
            "previous_branch": "main",
            "current_branch": "test_branch",
            "conversation": [
                {"role": "user", "content": "Switched message"},
                {"role": "assistant", "content": "Switched response"}
            ],
            "success": True
        }
        client.add_auto_response("branch_switch", switch_response)
        
        # Send branch switch message
        switch_message = {
            "type": "branch_switch",
            "branch_id": "test_branch",
            "session_id": "test_session"
        }
        
        response = await client.send_message(switch_message)
        
        assert response["type"] == "branch_switched"
        assert response["current_branch"] == "test_branch"
        assert response["success"] is True
        assert len(response["conversation"]) == 2


class TestRegenerationFunctionality:
    """Test regeneration button functionality."""

    def test_regeneration_preserves_branch_state(self):
        """Test that regeneration works correctly and preserves branch state."""
        with mock_webchat_server() as server:
            session_id = server.create_session("regen_test")
            session = server.get_session(session_id)
            
            # Setup conversation
            session.add_message("user", "Tell me about AI")
            session.add_message("assistant", "AI is artificial intelligence...")
            session.add_message("user", "Tell me more")
            session.add_message("assistant", "Original response about AI")
            
            # Create branch and switch to it
            success, message, branch = session.branch_manager.create_branch(
                from_branch_id="main",
                name="regen_branch"
            )
            assert success, f"Branch creation failed: {message}"
            
            original_length = len(session.conversation_history)
            original_branch = session.branch_manager.current_branch_id
            
            logger.info(f"🧪 TEST: Pre-regen - Branch: {original_branch}, Length: {original_length}")
            
            # Regenerate last message (should replace assistant's response)
            regen_data = {
                "action": "regenerate",
                "session_id": session_id,
                "message_index": -1  # Last message
            }
            
            # Mock the regeneration to produce a different response
            with patch('oumi.webchat.server.OumiWebServer.handle_regeneration') as mock_regen:
                mock_regen.return_value = {
                    "success": True,
                    "regenerated_message": "Regenerated response about AI",
                    "message_index": 3
                }
                
                response = server.handle_rest_request("POST", "/regenerate", regen_data)
            
            # Verify regeneration success
            assert response["success"] is True
            
            # Verify conversation length unchanged
            post_regen_length = len(session.conversation_history) 
            assert post_regen_length == original_length, \
                f"Conversation length changed during regeneration: {post_regen_length} != {original_length}"
            
            # Verify still on correct branch
            current_branch = session.branch_manager.current_branch_id
            assert current_branch == original_branch, \
                f"Branch changed during regeneration: {current_branch} != {original_branch}"
            
            # Verify last message was actually regenerated
            last_message = session.conversation_history[-1]
            assert last_message["content"] == "Regenerated response about AI", \
                f"Message not regenerated correctly: {last_message['content']}"

    @pytest.mark.asyncio 
    async def test_regeneration_via_websocket(self):
        """Test regeneration through WebSocket interface."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock regeneration response
        regen_response = {
            "type": "message_regenerated",
            "message_index": 3,
            "old_content": "Original response",
            "new_content": "Regenerated response",
            "branch_preserved": True,
            "success": True
        }
        client.add_auto_response("regenerate", regen_response)
        
        # Send regeneration request
        regen_message = {
            "type": "regenerate",
            "message_index": 3,
            "session_id": "test_session"
        }
        
        response = await client.send_message(regen_message)
        
        assert response["type"] == "message_regenerated"
        assert response["success"] is True
        assert response["branch_preserved"] is True
        assert response["new_content"] == "Regenerated response"


class TestBranchDebugLogging:
    """Test debug logging functionality for branch operations."""

    def test_debug_logging_captures_branch_operations(self):
        """Test that debug logging captures detailed branch operation information."""
        with mock_webchat_server() as server:
            session_id = server.create_session("debug_test")
            session = server.get_session(session_id)
            
            # Setup conversation
            session.add_message("user", "Test message")
            session.add_message("assistant", "Test response")
            
            # Capture logs during branch operations
            with patch('oumi.webchat.server.logger') as mock_logger:
                # Create branch
                success, message, branch = session.branch_manager.create_branch(
                    from_branch_id="main",
                    name="debug_branch"
                )
                assert success
                
                # Switch branch
                success, message, switched_branch = session.branch_manager.switch_branch("debug_branch")
                assert success
                
                # Verify debug logging was called
                mock_logger.info.assert_called()
                log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                
                # Check that specific debug messages were logged
                branch_create_logs = [log for log in log_calls if "Branch create" in log]
                branch_switch_logs = [log for log in log_calls if "Branch switch" in log]
                
                assert len(branch_create_logs) > 0, "No branch creation debug logs found"
                assert len(branch_switch_logs) > 0, "No branch switch debug logs found"
                
                # Check for specific debug information
                debug_info_found = any(
                    "conversation length" in log.lower() for log in log_calls
                )
                assert debug_info_found, "Debug logs should include conversation length information"

    def test_verbose_debug_mode_provides_detailed_information(self):
        """Test that verbose debug mode provides comprehensive branch operation details."""
        with mock_webchat_server() as server:
            session_id = server.create_session("verbose_test")
            session = server.get_session(session_id)
            
            # Build detailed conversation
            conversation = [
                {"role": "user", "content": "Hi there!"},
                {"role": "assistant", "content": "Hello! How can I help?"},
                {"role": "user", "content": "What's the weather like?"},
                {"role": "assistant", "content": "I don't have access to weather data."},
            ]
            
            for msg in conversation:
                session.add_message(msg["role"], msg["content"])
            
            # Test with verbose logging
            with patch('oumi.webchat.server.logger') as mock_logger:
                # Perform branch operation 
                success, message, branch = session.branch_manager.create_branch(
                    from_branch_id="main",
                    name="verbose_branch",
                    branch_point=2  # Branch after weather question
                )
                assert success
                
                # Verify detailed logging
                log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
                
                # Should log individual messages
                message_logs = [log for log in log_calls if "Branch-point Message" in log]
                assert len(message_logs) >= 3, f"Should log individual messages, got {len(message_logs)} logs"
                
                # Should log conversation lengths
                length_logs = [log for log in log_calls if "conversation length" in log.lower()]
                assert len(length_logs) > 0, "Should log conversation lengths"
                
                # Should log branch creation result
                result_logs = [log for log in log_calls if "Branch create result" in log]
                assert len(result_logs) > 0, "Should log branch creation result"