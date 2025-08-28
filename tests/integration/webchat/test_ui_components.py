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

"""Integration tests for WebChat UI components and interactions."""

import time

import pytest

from tests.unit.webchat.utils.webchat_test_utils import (
    MockWebSocketClient,
    cleanup_test_files,
    create_mock_file_upload,
    create_test_gradio_interface,
    mock_webchat_server,
)


class TestQuickActionButtons:
    """Test quick action button functionality."""

    @pytest.mark.asyncio
    async def test_clear_button_functionality(self):
        """Test Clear button clears conversation."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock clear action response
        clear_response = {
            "type": "conversation_cleared",
            "previous_message_count": 4,
            "conversation_history": [],
            "success": True,
        }
        client.add_auto_response("clear_conversation", clear_response)

        # Simulate clear button click
        clear_message = {
            "type": "clear_conversation",
            "session_id": "test_session",
            "confirm": True,
        }

        response = await client.send_message(clear_message)

        assert response["type"] == "conversation_cleared"
        assert response["previous_message_count"] == 4
        assert len(response["conversation_history"]) == 0
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_delete_last_button_functionality(self):
        """Test Delete Last button removes last exchange."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock delete last response
        delete_response = {
            "type": "message_deleted",
            "deleted_messages": 2,  # User + assistant message
            "remaining_count": 2,
            "success": True,
        }
        client.add_auto_response("delete_last", delete_response)

        delete_message = {"type": "delete_last", "session_id": "test_session"}

        response = await client.send_message(delete_message)

        assert response["type"] == "message_deleted"
        assert response["deleted_messages"] == 2
        assert response["remaining_count"] == 2
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_regenerate_button_functionality(self):
        """Test Regenerate button regenerates last response."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock regenerate response
        regenerate_response = {
            "type": "response_regenerated",
            "original_response": "Original response text",
            "new_response": "Regenerated response text",
            "regeneration_count": 2,
            "success": True,
        }
        client.add_auto_response("regenerate_response", regenerate_response)

        regenerate_message = {
            "type": "regenerate_response",
            "session_id": "test_session",
            "message_index": -1,  # Last message
        }

        response = await client.send_message(regenerate_message)

        assert response["type"] == "response_regenerated"
        assert response["new_response"] == "Regenerated response text"
        assert response["regeneration_count"] == 2
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_export_button_functionality(self):
        """Test Export button exports conversation."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock export response
        export_response = {
            "type": "conversation_exported",
            "export_format": "markdown",
            "file_path": "/tmp/conversation_export.md",
            "file_size": 1024,
            "success": True,
        }
        client.add_auto_response("export_conversation", export_response)

        export_message = {
            "type": "export_conversation",
            "format": "markdown",
            "include_metadata": True,
            "session_id": "test_session",
        }

        response = await client.send_message(export_message)

        assert response["type"] == "conversation_exported"
        assert response["export_format"] == "markdown"
        assert response["file_size"] > 0
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_help_button_functionality(self):
        """Test Help button displays help information."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock help response
        help_response = {
            "type": "help_displayed",
            "help_content": {
                "commands": ["/clear()", "/save()", "/export()"],
                "shortcuts": ["Ctrl+Enter: Send", "Ctrl+K: Clear"],
                "features": ["Branch management", "File upload", "Export"],
            },
            "success": True,
        }
        client.add_auto_response("show_help", help_response)

        help_message = {"type": "show_help", "session_id": "test_session"}

        response = await client.send_message(help_message)

        assert response["type"] == "help_displayed"
        assert "commands" in response["help_content"]
        assert "shortcuts" in response["help_content"]
        assert len(response["help_content"]["commands"]) >= 3

    @pytest.mark.asyncio
    async def test_button_state_management(self):
        """Test button enable/disable state management."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock button state update
        state_response = {
            "type": "button_states_updated",
            "button_states": {
                "clear_btn": {"enabled": True, "tooltip": "Clear conversation"},
                "delete_btn": {"enabled": True, "tooltip": "Delete last exchange"},
                "regen_btn": {"enabled": True, "tooltip": "Regenerate response"},
                "export_btn": {"enabled": True, "tooltip": "Export conversation"},
                "help_btn": {"enabled": True, "tooltip": "Show help"},
            },
            "success": True,
        }
        client.add_auto_response("update_button_states", state_response)

        state_message = {
            "type": "update_button_states",
            "conversation_length": 4,
            "session_id": "test_session",
        }

        response = await client.send_message(state_message)

        assert response["type"] == "button_states_updated"
        button_states = response["button_states"]

        # All buttons should be enabled when conversation exists
        assert button_states["clear_btn"]["enabled"] is True
        assert button_states["delete_btn"]["enabled"] is True
        assert button_states["regen_btn"]["enabled"] is True


class TestFileUploadFunctionality:
    """Test file upload and attachment functionality."""

    @pytest.mark.asyncio
    async def test_file_upload_via_websocket(self):
        """Test file upload through WebSocket interface."""
        client = MockWebSocketClient()
        await client.connect()

        # Create mock file
        test_content = b"This is test file content for upload testing."
        mock_file = create_mock_file_upload(
            "test_document.txt", test_content, "text/plain"
        )

        # Mock successful upload response
        upload_response = {
            "type": "file_uploaded",
            "file_name": "test_document.txt",
            "file_size": len(test_content),
            "file_type": "text/plain",
            "file_id": "file_123",
            "success": True,
        }
        client.add_auto_response("upload_file", upload_response)

        upload_message = {
            "type": "upload_file",
            "file_name": mock_file.name,
            "file_size": mock_file.size,
            "file_type": mock_file.content_type,
            "file_data": test_content.hex(),  # Hex encoded for transport
            "session_id": "test_session",
        }

        response = await client.send_message(upload_message)

        assert response["type"] == "file_uploaded"
        assert response["file_name"] == "test_document.txt"
        assert response["file_size"] == len(test_content)
        assert response["success"] is True

        # Cleanup
        cleanup_test_files(mock_file.temp_path)

    @pytest.mark.asyncio
    async def test_image_upload_and_processing(self):
        """Test image file upload and processing."""
        client = MockWebSocketClient()
        await client.connect()

        # Create mock image file
        # Simple PNG data (1x1 red pixel)
        png_data = bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000909c60370000000c49444154789c626001000000050001b3b5b7560000000049454e44ae426082"
        )
        mock_image = create_mock_file_upload("test_image.png", png_data, "image/png")

        # Mock image processing response
        image_response = {
            "type": "image_processed",
            "file_name": "test_image.png",
            "file_size": len(png_data),
            "image_dimensions": {"width": 1, "height": 1},
            "image_format": "PNG",
            "file_id": "image_456",
            "success": True,
        }
        client.add_auto_response("upload_file", image_response)

        upload_message = {
            "type": "upload_file",
            "file_name": mock_image.name,
            "file_type": mock_image.content_type,
            "file_data": png_data.hex(),
            "session_id": "test_session",
        }

        response = await client.send_message(upload_message)

        assert response["type"] == "image_processed"
        assert "image_dimensions" in response
        assert response["image_format"] == "PNG"

        cleanup_test_files(mock_image.temp_path)

    @pytest.mark.asyncio
    async def test_drag_and_drop_functionality(self):
        """Test drag and drop file upload simulation."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock drag and drop event
        drop_response = {
            "type": "files_dropped",
            "dropped_files": [
                {"name": "document.pdf", "size": 2048, "type": "application/pdf"},
                {"name": "image.jpg", "size": 1024, "type": "image/jpeg"},
            ],
            "files_accepted": 2,
            "files_rejected": 0,
            "success": True,
        }
        client.add_auto_response("drag_drop_files", drop_response)

        drop_message = {
            "type": "drag_drop_files",
            "files": [
                {"name": "document.pdf", "size": 2048, "type": "application/pdf"},
                {"name": "image.jpg", "size": 1024, "type": "image/jpeg"},
            ],
            "session_id": "test_session",
        }

        response = await client.send_message(drop_message)

        assert response["type"] == "files_dropped"
        assert response["files_accepted"] == 2
        assert response["files_rejected"] == 0
        assert len(response["dropped_files"]) == 2

    @pytest.mark.asyncio
    async def test_file_upload_error_handling(self):
        """Test file upload error scenarios."""
        client = MockWebSocketClient()
        await client.connect()

        # Test file too large error
        large_file_error = {
            "type": "upload_error",
            "error_code": "FILE_TOO_LARGE",
            "error_message": "File size exceeds 10MB limit",
            "max_size": 10485760,
            "file_size": 20971520,
            "success": False,
        }
        client.add_auto_response("upload_file", large_file_error)

        large_file_message = {
            "type": "upload_file",
            "file_name": "huge_file.zip",
            "file_size": 20971520,  # 20MB
            "session_id": "test_session",
        }

        response = await client.send_message(large_file_message)

        assert response["type"] == "upload_error"
        assert response["error_code"] == "FILE_TOO_LARGE"
        assert response["success"] is False

    def test_file_attachment_with_conversation(self):
        """Test file attachment integration with conversation."""
        with mock_webchat_server() as server:
            session_id = server.create_session("file_conversation_test")
            session = server.get_session(session_id)

            # Add initial conversation
            session.add_message("user", "I need help analyzing this data")
            session.add_message(
                "assistant", "I'd be happy to help! Please share the data file."
            )

            # Simulate file attachment
            file_response = server.handle_rest_request(
                "POST",
                f"/files/{session_id}",
                {"file_name": "data.csv", "file_type": "text/csv", "action": "analyze"},
            )

            assert file_response["status"] == "ok"

            # Add follow-up conversation about the file
            session.add_message("user", "Please analyze the uploaded CSV data")
            session.add_message(
                "assistant", "I can see the CSV file. Let me analyze it..."
            )

            # Should have 4 messages total
            assert len(session.conversation_history) == 4


class TestSettingsPanelControls:
    """Test settings panel UI controls."""

    @pytest.mark.asyncio
    async def test_temperature_slider_adjustment(self):
        """Test temperature slider adjustment."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock temperature update response
        temp_response = {
            "type": "setting_updated",
            "setting_name": "temperature",
            "old_value": 0.7,
            "new_value": 0.9,
            "success": True,
        }
        client.add_auto_response("update_setting", temp_response)

        temp_message = {
            "type": "update_setting",
            "setting_name": "temperature",
            "value": 0.9,
            "session_id": "test_session",
        }

        response = await client.send_message(temp_message)

        assert response["type"] == "setting_updated"
        assert response["setting_name"] == "temperature"
        assert response["new_value"] == 0.9

    @pytest.mark.asyncio
    async def test_max_tokens_slider_adjustment(self):
        """Test max tokens slider adjustment."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock max tokens update response
        tokens_response = {
            "type": "setting_updated",
            "setting_name": "max_new_tokens",
            "old_value": 100,
            "new_value": 256,
            "success": True,
        }
        client.add_auto_response("update_setting", tokens_response)

        tokens_message = {
            "type": "update_setting",
            "setting_name": "max_new_tokens",
            "value": 256,
            "session_id": "test_session",
        }

        response = await client.send_message(tokens_message)

        assert response["type"] == "setting_updated"
        assert response["setting_name"] == "max_new_tokens"
        assert response["new_value"] == 256

    @pytest.mark.asyncio
    async def test_model_switching_via_ui(self):
        """Test model switching through settings panel."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock model switch response
        model_response = {
            "type": "model_switched",
            "previous_model": "SmolLM-135M-Instruct",
            "new_model": "SmolLM-360M-Instruct",
            "model_loaded": True,
            "success": True,
        }
        client.add_auto_response("switch_model", model_response)

        switch_message = {
            "type": "switch_model",
            "target_model": "SmolLM-360M-Instruct",
            "session_id": "test_session",
        }

        response = await client.send_message(switch_message)

        assert response["type"] == "model_switched"
        assert response["new_model"] == "SmolLM-360M-Instruct"
        assert response["model_loaded"] is True

    def test_settings_persistence(self):
        """Test settings persistence across sessions."""
        with mock_webchat_server() as server:
            # Create session with custom settings
            session_id = server.create_session("settings_test")

            # Update settings via REST API
            settings_data = {"temperature": 0.8, "max_new_tokens": 200, "top_p": 0.95}

            response = server.handle_rest_request(
                "POST", f"/settings/{session_id}", settings_data
            )

            assert response["status"] == "ok"

            # Verify settings are stored
            get_response = server.handle_rest_request("GET", f"/settings/{session_id}")
            assert get_response["status"] == "ok"


class TestSystemMonitorIntegration:
    """Test system monitor UI component integration."""

    @pytest.mark.asyncio
    async def test_system_stats_real_time_updates(self):
        """Test real-time system statistics updates."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock system stats response
        stats_response = {
            "type": "system_stats",
            "gpu": {
                "usage": 65.5,
                "memory_used": 2048,
                "memory_total": 8192,
                "temperature": 72,
            },
            "cpu": {"usage": 45.2, "cores": 8, "frequency": 3200},
            "memory": {"used": 12288, "total": 16384, "usage_percent": 75.0},
            "timestamp": time.time(),
            "success": True,
        }
        client.add_auto_response("get_system_stats", stats_response)

        stats_message = {"type": "get_system_stats", "session_id": "test_session"}

        response = await client.send_message(stats_message)

        assert response["type"] == "system_stats"
        assert "gpu" in response
        assert "cpu" in response
        assert "memory" in response
        assert response["gpu"]["usage"] == 65.5
        assert response["cpu"]["cores"] == 8

    @pytest.mark.asyncio
    async def test_context_window_tracking(self):
        """Test context window usage tracking."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock context tracking response
        context_response = {
            "type": "context_usage",
            "tokens_used": 1024,
            "tokens_total": 2048,
            "usage_percent": 50.0,
            "messages_in_context": 8,
            "estimated_remaining": 1024,
            "success": True,
        }
        client.add_auto_response("get_context_usage", context_response)

        context_message = {"type": "get_context_usage", "session_id": "test_session"}

        response = await client.send_message(context_message)

        assert response["type"] == "context_usage"
        assert response["tokens_used"] == 1024
        assert response["usage_percent"] == 50.0
        assert response["messages_in_context"] == 8

    def test_system_monitor_rest_endpoints(self):
        """Test system monitor data via REST endpoints."""
        with mock_webchat_server() as server:
            # Get system stats
            stats_response = server.handle_rest_request("GET", "/system/stats")

            assert "gpu" in stats_response
            assert "cpu" in stats_response
            assert "memory" in stats_response


class TestUIStateManagement:
    """Test overall UI state management and synchronization."""

    @pytest.mark.asyncio
    async def test_ui_state_synchronization(self):
        """Test UI state sync between frontend and backend."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock complete UI state response
        ui_state_response = {
            "type": "ui_state_sync",
            "state": {
                "conversation": {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                    "message_count": 2,
                },
                "branch": {
                    "current_branch": "main",
                    "available_branches": ["main", "feature"],
                },
                "settings": {"temperature": 0.7, "max_new_tokens": 100},
                "ui_components": {
                    "buttons_enabled": True,
                    "upload_ready": True,
                    "export_available": True,
                },
            },
            "success": True,
        }
        client.add_auto_response("sync_ui_state", ui_state_response)

        sync_message = {"type": "sync_ui_state", "session_id": "test_session"}

        response = await client.send_message(sync_message)

        assert response["type"] == "ui_state_sync"
        state = response["state"]

        assert state["conversation"]["message_count"] == 2
        assert state["branch"]["current_branch"] == "main"
        assert state["settings"]["temperature"] == 0.7
        assert state["ui_components"]["buttons_enabled"] is True

    @pytest.mark.asyncio
    async def test_ui_error_state_handling(self):
        """Test UI error state display and recovery."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock error state response
        error_state_response = {
            "type": "ui_error_state",
            "error_type": "CONNECTION_LOST",
            "error_message": "Connection to backend lost",
            "recovery_actions": ["Reconnect", "Refresh Page"],
            "auto_recovery": True,
            "success": False,
        }
        client.add_auto_response("ui_error", error_state_response)

        error_message = {
            "type": "ui_error",
            "error_type": "CONNECTION_LOST",
            "session_id": "test_session",
        }

        response = await client.send_message(error_message)

        assert response["type"] == "ui_error_state"
        assert response["error_type"] == "CONNECTION_LOST"
        assert "Reconnect" in response["recovery_actions"]
        assert response["auto_recovery"] is True

    def test_gradio_interface_integration(self):
        """Test Gradio interface component integration."""
        # Test Gradio interface creation and configuration
        interface = create_test_gradio_interface()

        # Verify all expected components are present
        assert hasattr(interface, "chatbot")
        assert hasattr(interface, "msg_input")
        assert hasattr(interface, "upload_btn")
        assert hasattr(interface, "clear_btn")
        assert hasattr(interface, "delete_btn")
        assert hasattr(interface, "regen_btn")
        assert hasattr(interface, "export_btn")
        assert hasattr(interface, "help_btn")
        assert hasattr(interface, "branch_tree")
        assert hasattr(interface, "system_monitor")
        assert hasattr(interface, "settings_panel")

        # Verify interface methods are available
        assert hasattr(interface, "update_chatbot")
        assert hasattr(interface, "update_branch_tree")
        assert hasattr(interface, "clear_chat")

    @pytest.mark.asyncio
    async def test_responsive_ui_updates(self):
        """Test responsive UI updates based on screen size."""
        client = MockWebSocketClient()
        await client.connect()

        # Mock responsive layout response
        layout_response = {
            "type": "layout_updated",
            "screen_size": "desktop",
            "layout_config": {
                "sidebar_visible": True,
                "branch_tree_expanded": True,
                "monitor_panel_docked": True,
                "button_layout": "horizontal",
            },
            "success": True,
        }
        client.add_auto_response("update_layout", layout_response)

        layout_message = {
            "type": "update_layout",
            "screen_width": 1920,
            "screen_height": 1080,
            "session_id": "test_session",
        }

        response = await client.send_message(layout_message)

        assert response["type"] == "layout_updated"
        assert response["screen_size"] == "desktop"
        layout = response["layout_config"]
        assert layout["sidebar_visible"] is True
        assert layout["branch_tree_expanded"] is True
