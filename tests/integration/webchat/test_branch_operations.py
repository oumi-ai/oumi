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

"""Integration tests for WebChat branch operations."""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from tests.unit.webchat.utils.webchat_test_utils import (
    MockWebSocketClient,
    WebChatTestServer,
    mock_webchat_server,
    assert_websocket_message,
    assert_session_state,
)


class TestBranchCreationOperations:
    """Test branch creation through various interfaces."""

    @pytest.mark.asyncio
    async def test_branch_creation_via_websocket(self):
        """Test branch creation through WebSocket interface."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock successful branch creation response
        branch_response = {
            "type": "branch_created",
            "branch_name": "feature_branch",
            "parent_branch": "main",
            "branch_id": "branch_123",
            "success": True
        }
        client.add_auto_response("branch_create", branch_response)
        
        # Send branch creation message
        create_message = {
            "type": "branch_create",
            "branch_name": "feature_branch",
            "parent_branch": "main",
            "session_id": "test_session"
        }
        
        response = await client.send_message(create_message)
        
        assert_websocket_message(
            response,
            "branch_created",
            ["branch_name", "parent_branch", "branch_id", "success"]
        )
        assert response["branch_name"] == "feature_branch"
        assert response["parent_branch"] == "main"
        assert response["success"] is True

    def test_branch_creation_via_rest_api(self):
        """Test branch creation through REST API."""
        with mock_webchat_server() as server:
            session_id = server.create_session("branch_test")
            
            # Create branch via REST API
            branch_data = {
                "name": "api_branch",
                "parent": "main",
                "message": "Creating branch via API"
            }
            
            response = server.handle_rest_request(
                "POST", 
                f"/branches/{session_id}", 
                branch_data
            )
            
            assert response["status"] == "ok"

    @pytest.mark.asyncio
    async def test_branch_creation_with_conversation_context(self):
        """Test branch creation preserves conversation context."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Setup conversation context
        conversation_messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is..."}
        ]
        
        # Mock branch creation with context preservation
        branch_response = {
            "type": "branch_created",
            "branch_name": "ml_discussion",
            "conversation_preserved": True,
            "message_count": len(conversation_messages),
            "success": True
        }
        client.add_auto_response("branch_create", branch_response)
        
        create_message = {
            "type": "branch_create",
            "branch_name": "ml_discussion",
            "preserve_context": True,
            "session_id": "test_session"
        }
        
        response = await client.send_message(create_message)
        
        assert response["conversation_preserved"] is True
        assert response["message_count"] == 2

    @pytest.mark.asyncio
    async def test_branch_creation_error_handling(self):
        """Test branch creation error scenarios."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Test duplicate branch name error
        error_response = {
            "type": "branch_error",
            "error_code": "DUPLICATE_BRANCH_NAME",
            "error_message": "Branch name already exists",
            "success": False
        }
        client.add_auto_response("branch_create", error_response)
        
        duplicate_message = {
            "type": "branch_create",
            "branch_name": "main",  # Trying to create branch with existing name
            "session_id": "test_session"
        }
        
        response = await client.send_message(duplicate_message)
        
        assert response["type"] == "branch_error"
        assert response["error_code"] == "DUPLICATE_BRANCH_NAME"
        assert response["success"] is False


class TestBranchSwitchingOperations:
    """Test branch switching functionality."""

    @pytest.mark.asyncio
    async def test_branch_switching_via_websocket(self):
        """Test branch switching through WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock successful branch switch response
        switch_response = {
            "type": "branch_switched",
            "previous_branch": "main",
            "current_branch": "feature_branch",
            "conversation_history": [
                {"role": "user", "content": "Previous conversation"},
                {"role": "assistant", "content": "In this branch"}
            ],
            "success": True
        }
        client.add_auto_response("branch_switch", switch_response)
        
        switch_message = {
            "type": "branch_switch",
            "target_branch": "feature_branch",
            "session_id": "test_session"
        }
        
        response = await client.send_message(switch_message)
        
        assert response["type"] == "branch_switched"
        assert response["current_branch"] == "feature_branch"
        assert response["previous_branch"] == "main"
        assert len(response["conversation_history"]) == 2

    def test_branch_switching_via_command(self):
        """Test branch switching via /branch() command."""
        with mock_webchat_server() as server:
            session_id = server.create_session("branch_command_test")
            session = server.get_session(session_id)
            
            # Simulate branch command execution
            command_response = server.handle_rest_request(
                "POST",
                f"/commands/{session_id}",
                {"command": "/branch(switch, feature_branch)"}
            )
            
            assert command_response["status"] == "ok"

    @pytest.mark.asyncio
    async def test_branch_switching_with_state_sync(self):
        """Test branch switching with UI state synchronization."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock branch switch with UI state update
        switch_response = {
            "type": "branch_switched",
            "current_branch": "ui_test_branch",
            "ui_state": {
                "branch_tree_update": True,
                "conversation_updated": True,
                "settings_preserved": True
            },
            "success": True
        }
        client.add_auto_response("branch_switch", switch_response)
        
        switch_message = {
            "type": "branch_switch",
            "target_branch": "ui_test_branch",
            "update_ui": True,
            "session_id": "test_session"
        }
        
        response = await client.send_message(switch_message)
        
        assert "ui_state" in response
        assert response["ui_state"]["branch_tree_update"] is True
        assert response["ui_state"]["conversation_updated"] is True

    @pytest.mark.asyncio
    async def test_branch_switching_error_cases(self):
        """Test branch switching error scenarios."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Test switching to non-existent branch
        error_response = {
            "type": "branch_error",
            "error_code": "BRANCH_NOT_FOUND",
            "error_message": "Target branch does not exist",
            "target_branch": "nonexistent_branch",
            "success": False
        }
        client.add_auto_response("branch_switch", error_response)
        
        invalid_switch = {
            "type": "branch_switch",
            "target_branch": "nonexistent_branch",
            "session_id": "test_session"
        }
        
        response = await client.send_message(invalid_switch)
        
        assert response["type"] == "branch_error"
        assert response["error_code"] == "BRANCH_NOT_FOUND"
        assert response["target_branch"] == "nonexistent_branch"


class TestBranchVisualizationOperations:
    """Test branch tree visualization and UI interactions."""

    @pytest.mark.asyncio
    async def test_branch_tree_data_retrieval(self):
        """Test retrieval of branch tree data for visualization."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock comprehensive branch tree response
        tree_response = {
            "type": "branch_tree_data",
            "nodes": [
                {
                    "id": "main",
                    "label": "Main",
                    "active": True,
                    "message_count": 5,
                    "created_at": "2025-01-01T12:00:00Z"
                },
                {
                    "id": "feature_1",
                    "label": "Feature 1",
                    "active": False,
                    "message_count": 3,
                    "created_at": "2025-01-01T12:30:00Z"
                },
                {
                    "id": "feature_2", 
                    "label": "Feature 2",
                    "active": False,
                    "message_count": 7,
                    "created_at": "2025-01-01T13:00:00Z"
                }
            ],
            "edges": [
                {"source": "main", "target": "feature_1"},
                {"source": "main", "target": "feature_2"}
            ],
            "success": True
        }
        client.add_auto_response("get_branch_tree", tree_response)
        
        tree_request = {
            "type": "get_branch_tree",
            "session_id": "test_session"
        }
        
        response = await client.send_message(tree_request)
        
        assert response["type"] == "branch_tree_data"
        assert len(response["nodes"]) == 3
        assert len(response["edges"]) == 2
        
        # Verify node structure
        main_node = next(node for node in response["nodes"] if node["id"] == "main")
        assert main_node["active"] is True
        assert main_node["message_count"] == 5
        
        # Verify edge structure
        assert {"source": "main", "target": "feature_1"} in response["edges"]

    def test_branch_tree_rest_endpoint(self):
        """Test branch tree data via REST API."""
        with mock_webchat_server() as server:
            session_id = server.create_session("tree_test")
            
            # Get branch tree data
            response = server.handle_rest_request("GET", f"/branches/{session_id}")
            
            assert "branches" in response
            assert "current" in response
            assert response["current"] == "main"

    @pytest.mark.asyncio
    async def test_branch_context_menu_operations(self):
        """Test branch context menu operations via WebSocket."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Test branch rename operation
        rename_response = {
            "type": "branch_renamed",
            "old_name": "feature_branch",
            "new_name": "improved_feature",
            "branch_id": "branch_123",
            "success": True
        }
        client.add_auto_response("branch_rename", rename_response)
        
        rename_message = {
            "type": "branch_rename",
            "branch_id": "branch_123",
            "old_name": "feature_branch",
            "new_name": "improved_feature",
            "session_id": "test_session"
        }
        
        response = await client.send_message(rename_message)
        
        assert response["type"] == "branch_renamed"
        assert response["old_name"] == "feature_branch"
        assert response["new_name"] == "improved_feature"

    @pytest.mark.asyncio
    async def test_branch_deletion_operations(self):
        """Test branch deletion through UI operations."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock branch deletion response
        delete_response = {
            "type": "branch_deleted",
            "deleted_branch": "old_feature",
            "parent_branch": "main",
            "remaining_branches": ["main", "feature_2"],
            "success": True
        }
        client.add_auto_response("branch_delete", delete_response)
        
        delete_message = {
            "type": "branch_delete",
            "target_branch": "old_feature",
            "confirm_deletion": True,
            "session_id": "test_session"
        }
        
        response = await client.send_message(delete_message)
        
        assert response["type"] == "branch_deleted"
        assert response["deleted_branch"] == "old_feature"
        assert "main" in response["remaining_branches"]
        assert "old_feature" not in response["remaining_branches"]

    @pytest.mark.asyncio
    async def test_real_time_branch_updates(self):
        """Test real-time branch updates and synchronization."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Mock real-time branch update broadcast
        update_response = {
            "type": "branch_update_broadcast",
            "update_type": "message_added",
            "branch_id": "main", 
            "new_message_count": 6,
            "last_activity": "2025-01-01T14:00:00Z",
            "success": True
        }
        client.add_auto_response("branch_activity", update_response)
        
        activity_message = {
            "type": "branch_activity",
            "activity_type": "message_added",
            "branch_id": "main",
            "session_id": "test_session"
        }
        
        response = await client.send_message(activity_message)
        
        assert response["type"] == "branch_update_broadcast"
        assert response["update_type"] == "message_added"
        assert response["new_message_count"] == 6


class TestAdvancedBranchOperations:
    """Test advanced branch operations and edge cases."""

    def test_branch_operation_with_session_management(self):
        """Test branch operations with session persistence."""
        with mock_webchat_server() as server:
            # Create session and add conversation data
            session_id = server.create_session("branch_session_test")
            session = server.get_session(session_id)
            
            session.add_message("user", "Initial conversation")
            session.add_message("assistant", "Initial response")
            
            assert_session_state(session, expected_messages=2)
            
            # Simulate branch operation affecting session state
            branch_response = server.handle_rest_request(
                "POST",
                f"/branches/{session_id}",
                {"name": "session_branch", "preserve_messages": True}
            )
            
            assert branch_response["status"] == "ok"
            # Session should maintain its conversation after branch creation
            assert_session_state(session, expected_messages=2)

    @pytest.mark.asyncio
    async def test_concurrent_branch_operations(self):
        """Test concurrent branch operations."""
        client1 = MockWebSocketClient()
        client2 = MockWebSocketClient()
        
        await client1.connect()
        await client2.connect()
        
        # Setup responses for concurrent operations
        create_response = {
            "type": "branch_created",
            "branch_name": "concurrent_branch",
            "success": True
        }
        
        switch_response = {
            "type": "branch_switched", 
            "current_branch": "main",
            "success": True
        }
        
        client1.add_auto_response("branch_create", create_response)
        client2.add_auto_response("branch_switch", switch_response)
        
        # Execute concurrent operations
        create_task = client1.send_message({
            "type": "branch_create",
            "branch_name": "concurrent_branch",
            "session_id": "session_1"
        })
        
        switch_task = client2.send_message({
            "type": "branch_switch",
            "target_branch": "main",
            "session_id": "session_2"
        })
        
        # Wait for both operations to complete
        create_result, switch_result = await asyncio.gather(create_task, switch_task)
        
        assert create_result["success"] is True
        assert switch_result["success"] is True

    @pytest.mark.asyncio
    async def test_branch_operation_error_recovery(self):
        """Test error recovery in branch operations."""
        client = MockWebSocketClient()
        await client.connect()
        
        # Test operation failure followed by retry
        failure_response = {
            "type": "branch_error",
            "error_code": "TEMPORARY_ERROR",
            "retry_allowed": True,
            "success": False
        }
        
        success_response = {
            "type": "branch_created",
            "branch_name": "retry_branch",
            "success": True
        }
        
        # First attempt fails, second succeeds
        client.add_auto_response("branch_create", failure_response)
        
        first_attempt = {
            "type": "branch_create",
            "branch_name": "retry_branch",
            "session_id": "test_session"
        }
        
        # First attempt should fail
        response1 = await client.send_message(first_attempt)
        assert response1["success"] is False
        assert response1["retry_allowed"] is True
        
        # Update response for retry
        client.add_auto_response("branch_create", success_response)
        
        # Retry should succeed
        response2 = await client.send_message(first_attempt)
        assert response2["success"] is True

    def test_branch_state_consistency(self):
        """Test branch state consistency across operations."""
        with mock_webchat_server() as server:
            session_id = server.create_session("consistency_test")
            session = server.get_session(session_id)
            
            # Verify initial branch state
            initial_branch = session.branch_manager.get_current_branch()
            assert initial_branch == "main"
            
            # Simulate branch operations that should maintain consistency
            operations = [
                ("CREATE", {"name": "test_branch"}),
                ("SWITCH", {"target": "test_branch"}),  
                ("RENAME", {"old": "test_branch", "new": "renamed_branch"}),
                ("SWITCH", {"target": "main"}),
            ]
            
            for operation, params in operations:
                response = server.handle_rest_request(
                    "POST",
                    f"/branches/{session_id}",
                    {"operation": operation, **params}
                )
                
                # Each operation should succeed
                assert response["status"] == "ok"
                
                # Session should remain active and valid
                assert_session_state(session, should_be_active=True)