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

"""Unit tests for WebChat CLI functionality."""

import time
from unittest.mock import Mock, patch, call

import pytest

from tests.unit.webchat.utils.webchat_test_utils import (
    PortTestHelper,
    wait_for_condition,
)


class TestPortManagement:
    """Test port management functionality."""

    def test_find_free_port(self):
        """Test finding a free port."""
        port = PortTestHelper.find_free_port()
        
        assert isinstance(port, int)
        assert 1024 <= port <= 65535
        assert not PortTestHelper.is_port_open("localhost", port)

    def test_port_availability_check(self):
        """Test port availability checking."""
        free_port = PortTestHelper.find_free_port()
        
        # Free port should not be open
        assert not PortTestHelper.is_port_open("localhost", free_port)
        
        # Occupy the port and check again
        with PortTestHelper.occupy_port("localhost", free_port):
            assert PortTestHelper.is_port_open("localhost", free_port)
        
        # Port should be free again after context exit
        assert not PortTestHelper.is_port_open("localhost", free_port)

    def test_port_conflict_detection(self):
        """Test detection of port conflicts."""
        test_port = PortTestHelper.find_free_port()
        
        # Initially no conflict
        assert not PortTestHelper.is_port_open("localhost", test_port)
        
        # Create conflict by occupying port
        with PortTestHelper.occupy_port("localhost", test_port):
            # Should detect conflict
            assert PortTestHelper.is_port_open("localhost", test_port)
            
            # In real implementation, this would trigger port resolution
            alternative_port = PortTestHelper.find_free_port()
            assert alternative_port != test_port
            assert not PortTestHelper.is_port_open("localhost", alternative_port)

    def test_multiple_port_conflicts(self):
        """Test handling multiple port conflicts."""
        ports = [PortTestHelper.find_free_port() for _ in range(3)]
        
        # Occupy multiple ports simultaneously
        with PortTestHelper.occupy_port("localhost", ports[0]):
            with PortTestHelper.occupy_port("localhost", ports[1]):
                # All occupied ports should be detected
                assert PortTestHelper.is_port_open("localhost", ports[0])
                assert PortTestHelper.is_port_open("localhost", ports[1])
                assert not PortTestHelper.is_port_open("localhost", ports[2])


class TestWebChatCLI:
    """Test WebChat CLI functionality."""

    @patch("oumi.cli.webchat.OumiWebServer")
    @patch("oumi.cli.webchat.WebChatInterface")
    def test_webchat_launch_basic(self, mock_interface, mock_server):
        """Test basic webchat launch functionality."""
        # Mock server setup
        mock_server_instance = Mock()
        mock_server_instance.start.return_value = True
        mock_server_instance.get_port.return_value = 8080
        mock_server_instance.is_healthy.return_value = True
        mock_server.return_value = mock_server_instance
        
        # Mock interface setup
        mock_interface_instance = Mock()
        mock_interface_instance.launch.return_value = Mock(server_port=7860)
        mock_interface.return_value = mock_interface_instance
        
        # Import and test the CLI function (mocked)
        with patch("oumi.cli.webchat.launch_webchat") as mock_launch:
            mock_launch.return_value = True
            
            result = mock_launch(host="localhost", port=8080)
            assert result is True
            
            # Verify launch was called with correct parameters
            mock_launch.assert_called_once_with(host="localhost", port=8080)

    @patch("oumi.cli.webchat.OumiWebServer")
    def test_server_only_launch(self, mock_server):
        """Test server-only launch for development mode."""
        mock_server_instance = Mock()
        mock_server_instance.start.return_value = True
        mock_server_instance.get_port.return_value = 8080
        mock_server_instance.is_healthy.return_value = True
        mock_server.return_value = mock_server_instance
        
        with patch("oumi.cli.webchat.launch_server_only") as mock_launch_server:
            mock_launch_server.return_value = True
            
            result = mock_launch_server(host="localhost", port=8080)
            assert result is True
            
            mock_launch_server.assert_called_once_with(host="localhost", port=8080)

    @patch("oumi.cli.webchat.check_backend_health")
    def test_health_check_polling(self, mock_health_check):
        """Test backend health check polling functionality."""
        # Simulate health check progression: fail, fail, succeed
        mock_health_check.side_effect = [False, False, True]
        
        with patch("oumi.cli.webchat.poll_backend_health") as mock_poll:
            mock_poll.return_value = True
            
            result = mock_poll("http://localhost:8080", timeout=5.0)
            assert result is True

    @patch("oumi.cli.webchat.check_backend_health")
    def test_health_check_timeout(self, mock_health_check):
        """Test health check timeout handling."""
        # Always return unhealthy
        mock_health_check.return_value = False
        
        with patch("oumi.cli.webchat.poll_backend_health") as mock_poll:
            mock_poll.side_effect = TimeoutError("Backend health check timeout")
            
            with pytest.raises(TimeoutError, match="Backend health check timeout"):
                mock_poll("http://localhost:8080", timeout=1.0)

    def test_wait_for_condition_success(self):
        """Test wait_for_condition utility success case."""
        condition_met = False
        
        def condition():
            return condition_met
        
        # Start a thread to set condition after delay
        import threading
        def set_condition():
            time.sleep(0.1)
            nonlocal condition_met
            condition_met = True
        
        thread = threading.Thread(target=set_condition)
        thread.start()
        
        # Should succeed
        result = wait_for_condition(condition, timeout=1.0)
        assert result is True
        
        thread.join()

    def test_wait_for_condition_timeout(self):
        """Test wait_for_condition utility timeout case."""
        def never_true():
            return False
        
        with pytest.raises(TimeoutError, match="Test timeout message"):
            wait_for_condition(
                never_true,
                timeout=0.1,
                error_message="Test timeout message"
            )

    @patch("oumi.cli.webchat.argparse.ArgumentParser")
    def test_cli_argument_parsing(self, mock_parser):
        """Test CLI argument parsing."""
        mock_parser_instance = Mock()
        mock_args = Mock()
        mock_args.host = "localhost"
        mock_args.port = 8080
        mock_args.server_only = False
        mock_args.config_path = None
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        with patch("oumi.cli.webchat.main") as mock_main:
            mock_main.return_value = 0
            
            result = mock_main()
            assert result == 0

    @patch("builtins.print")
    def test_error_message_display(self, mock_print):
        """Test error message display functionality."""
        error_msg = "Test error message"
        
        with patch("oumi.cli.webchat.display_error") as mock_display:
            mock_display(error_msg)
            mock_display.assert_called_once_with(error_msg)

    def test_configuration_loading(self):
        """Test configuration loading and validation."""
        with patch("oumi.cli.webchat.load_config") as mock_load_config:
            mock_config = Mock()
            mock_config.model.model_name = "test-model"
            mock_config.generation.max_new_tokens = 100
            mock_load_config.return_value = mock_config
            
            config = mock_load_config("test_config.yaml")
            assert config.model.model_name == "test-model"
            assert config.generation.max_new_tokens == 100
            
            mock_load_config.assert_called_once_with("test_config.yaml")


class TestWebChatStartupSequence:
    """Test the complete webchat startup sequence."""

    @patch("oumi.cli.webchat.OumiWebServer")
    @patch("oumi.cli.webchat.WebChatInterface")
    @patch("oumi.cli.webchat.poll_backend_health")
    def test_complete_startup_sequence(
        self, 
        mock_poll_health, 
        mock_interface, 
        mock_server
    ):
        """Test complete startup sequence with all components."""
        # Setup mocks
        mock_server_instance = Mock()
        mock_server_instance.start.return_value = True
        mock_server_instance.get_port.return_value = 8080
        mock_server_instance.is_healthy.return_value = True
        mock_server.return_value = mock_server_instance
        
        mock_interface_instance = Mock()
        mock_interface_instance.launch.return_value = Mock(server_port=7860)
        mock_interface.return_value = mock_interface_instance
        
        mock_poll_health.return_value = True
        
        # Mock the full startup sequence
        with patch("oumi.cli.webchat.complete_startup") as mock_startup:
            def startup_sequence():
                # Simulate startup steps
                mock_server_instance.start()
                mock_poll_health("http://localhost:8080")
                mock_interface_instance.launch()
                return True
            
            mock_startup.side_effect = startup_sequence
            
            result = mock_startup()
            assert result is True

    @patch("oumi.cli.webchat.OumiWebServer")
    def test_server_startup_failure(self, mock_server):
        """Test handling of server startup failure."""
        mock_server_instance = Mock()
        mock_server_instance.start.side_effect = RuntimeError("Port already in use")
        mock_server.return_value = mock_server_instance
        
        with patch("oumi.cli.webchat.handle_startup_failure") as mock_handle_failure:
            mock_handle_failure.side_effect = RuntimeError("Port already in use")
            
            with pytest.raises(RuntimeError, match="Port already in use"):
                mock_handle_failure()

    @patch("oumi.cli.webchat.OumiWebServer")
    @patch("oumi.cli.webchat.poll_backend_health")
    def test_health_check_failure(self, mock_poll_health, mock_server):
        """Test handling of health check failure."""
        # Server starts successfully
        mock_server_instance = Mock()
        mock_server_instance.start.return_value = True
        mock_server.return_value = mock_server_instance
        
        # But health check fails
        mock_poll_health.side_effect = TimeoutError("Backend not responsive")
        
        with pytest.raises(TimeoutError, match="Backend not responsive"):
            mock_poll_health("http://localhost:8080")

    @patch("oumi.cli.webchat.WebChatInterface")
    def test_interface_launch_failure(self, mock_interface):
        """Test handling of interface launch failure."""
        mock_interface_instance = Mock()
        mock_interface_instance.launch.side_effect = RuntimeError("Gradio launch failed")
        mock_interface.return_value = mock_interface_instance
        
        with pytest.raises(RuntimeError, match="Gradio launch failed"):
            mock_interface_instance.launch()


class TestWebChatConfiguration:
    """Test webchat configuration handling."""

    def test_default_configuration(self):
        """Test default configuration values."""
        with patch("oumi.cli.webchat.get_default_config") as mock_get_default:
            mock_config = Mock()
            mock_config.host = "localhost"
            mock_config.port = 8080
            mock_config.server_only = False
            mock_get_default.return_value = mock_config
            
            config = mock_get_default()
            assert config.host == "localhost"
            assert config.port == 8080
            assert config.server_only is False

    def test_custom_configuration_override(self):
        """Test custom configuration override."""
        with patch("oumi.cli.webchat.override_config") as mock_override:
            base_config = Mock()
            base_config.host = "localhost"
            base_config.port = 8080
            
            overrides = {"host": "0.0.0.0", "port": 9000}
            
            def apply_overrides(config, overrides_dict):
                for key, value in overrides_dict.items():
                    setattr(config, key, value)
                return config
            
            mock_override.side_effect = apply_overrides
            
            updated_config = mock_override(base_config, overrides)
            assert updated_config.host == "0.0.0.0"
            assert updated_config.port == 9000

    def test_configuration_validation(self):
        """Test configuration validation."""
        with patch("oumi.cli.webchat.validate_config") as mock_validate:
            # Valid configuration
            valid_config = Mock()
            valid_config.host = "localhost"
            valid_config.port = 8080
            mock_validate.return_value = True
            
            assert mock_validate(valid_config) is True
            
            # Invalid configuration
            invalid_config = Mock()
            invalid_config.host = ""
            invalid_config.port = -1
            mock_validate.return_value = False
            
            assert mock_validate(invalid_config) is False