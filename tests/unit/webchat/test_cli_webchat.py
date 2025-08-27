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

import asyncio
import time
from unittest.mock import Mock, patch, call, AsyncMock

import pytest
from typer.testing import CliRunner

from oumi.cli.webchat import (
    check_port_availability,
    find_available_port,
    wait_for_backend_health,
    webchat,
    webchat_server,
)
from tests.unit.webchat.utils.webchat_test_utils import (
    PortTestHelper,
    wait_for_condition,
)


class TestWebChatCLIUtilities:
    """Test WebChat CLI utility functions."""

    def test_check_port_availability(self):
        """Test port availability checking."""
        free_port = PortTestHelper.find_free_port()
        
        # Free port should be available
        assert check_port_availability("localhost", free_port) is True
        
        # Occupy the port and check again
        with PortTestHelper.occupy_port("localhost", free_port):
            assert check_port_availability("localhost", free_port) is False

    def test_find_available_port(self):
        """Test finding an available port."""
        port = find_available_port()
        
        assert isinstance(port, int)
        assert 1024 <= port <= 65535
        assert check_port_availability("localhost", port) is True

    @pytest.mark.asyncio
    async def test_wait_for_backend_health_success(self):
        """Test successful backend health check."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock successful health check response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await wait_for_backend_health("http://localhost:8000/health", timeout=1)
            assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_backend_health_timeout(self):
        """Test backend health check timeout."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            # Mock connection error
            mock_get.side_effect = Exception("Connection failed")
            
            result = await wait_for_backend_health("http://localhost:8000/health", timeout=0.1)
            assert result is False

    def test_port_conflict_detection(self):
        """Test detection of port conflicts."""
        test_port = PortTestHelper.find_free_port()
        
        # Initially no conflict
        assert check_port_availability("localhost", test_port) is True
        
        # Create conflict by occupying port
        with PortTestHelper.occupy_port("localhost", test_port):
            # Should detect conflict
            assert check_port_availability("localhost", test_port) is False
            
            # Find alternative port
            alternative_port = find_available_port()
            assert alternative_port != test_port
            assert check_port_availability("localhost", alternative_port) is True

    def test_multiple_port_conflicts(self):
        """Test handling multiple port conflicts."""
        ports = [PortTestHelper.find_free_port() for _ in range(3)]
        
        # Occupy multiple ports simultaneously
        with PortTestHelper.occupy_port("localhost", ports[0]):
            with PortTestHelper.occupy_port("localhost", ports[1]):
                # Occupied ports should not be available
                assert check_port_availability("localhost", ports[0]) is False
                assert check_port_availability("localhost", ports[1]) is False
                assert check_port_availability("localhost", ports[2]) is True


class TestWebChatTyperCommands:
    """Test WebChat Typer CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing Typer commands."""
        return CliRunner()

    @patch("oumi.webchat.interface.launch_webchat")
    @patch("oumi.cli.webchat.find_available_port")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_webchat_command_basic(self, mock_check_port, mock_find_port, mock_launch, cli_runner):
        """Test basic webchat command execution."""
        # Mock port availability
        mock_check_port.return_value = True
        mock_find_port.return_value = 8080
        
        # Mock successful launch
        mock_launch.return_value = None
        
        # Execute webchat command
        result = cli_runner.invoke(webchat, ["--host", "localhost", "--port", "8080"])
        
        # Command should execute without errors
        assert result.exit_code == 0
        mock_launch.assert_called_once()

    @patch("oumi.webchat.server.run_webchat_server")
    @patch("oumi.cli.webchat.find_available_port")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_webchat_server_command(self, mock_check_port, mock_find_port, mock_run_server, cli_runner):
        """Test webchat-server command execution."""
        # Mock port availability
        mock_check_port.return_value = True
        mock_find_port.return_value = 8080
        
        # Mock successful server run
        mock_run_server.return_value = None
        
        # Execute webchat-server command
        result = cli_runner.invoke(webchat_server, ["--host", "localhost", "--port", "8080"])
        
        # Command should execute without errors
        assert result.exit_code == 0
        mock_run_server.assert_called_once()

    @patch("oumi.webchat.interface.launch_webchat")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_webchat_port_conflict_resolution(self, mock_check_port, mock_launch, cli_runner):
        """Test automatic port conflict resolution."""
        # Simulate port conflict on default port
        mock_check_port.side_effect = [False, False, True]  # First two ports busy, third available
        
        with patch("oumi.cli.webchat.find_available_port", return_value=8082):
            mock_launch.return_value = None
            
            # Execute webchat command with conflicted port
            result = cli_runner.invoke(webchat, ["--port", "8080"])
            
            # Should succeed despite port conflict
            assert result.exit_code == 0
            mock_launch.assert_called_once()

    @patch("oumi.webchat.interface.launch_webchat")
    def test_webchat_command_with_config(self, mock_launch, cli_runner):
        """Test webchat command with configuration file."""
        mock_launch.return_value = None
        
        # Execute webchat command with config
        result = cli_runner.invoke(webchat, ["-c", "test_config.yaml"])
        
        # Should pass config to launch function
        assert result.exit_code == 0
        mock_launch.assert_called_once()

    @patch("oumi.webchat.server.run_webchat_server")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_server_command_error_handling(self, mock_check_port, mock_run_server, cli_runner):
        """Test server command error handling."""
        # Mock port check success but server failure
        mock_check_port.return_value = True
        mock_run_server.side_effect = RuntimeError("Server startup failed")
        
        # Execute command and expect error
        result = cli_runner.invoke(webchat_server, ["--port", "8080"])
        
        # Should exit with error code
        assert result.exit_code != 0
        assert "Server startup failed" in str(result.exception) or result.exit_code == 1

    def test_cli_help_messages(self, cli_runner):
        """Test CLI help messages."""
        # Test main webchat command help
        result = cli_runner.invoke(webchat, ["--help"])
        assert result.exit_code == 0
        assert "webchat" in result.output.lower()
        
        # Test server command help
        result = cli_runner.invoke(webchat_server, ["--help"])
        assert result.exit_code == 0
        assert "server" in result.output.lower()


class TestWebChatStartupSequence:
    """Test webchat startup sequence and health checking."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @patch("oumi.webchat.interface.launch_webchat")
    @patch("oumi.cli.webchat.wait_for_backend_health")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_complete_startup_sequence(
        self, 
        mock_check_port,
        mock_wait_health, 
        mock_launch,
        cli_runner
    ):
        """Test complete startup sequence with health checking."""
        # Mock successful port check and health check
        mock_check_port.return_value = True
        
        # Mock async health check
        async def mock_health_check(*args, **kwargs):
            return True
        mock_wait_health.return_value = asyncio.create_task(mock_health_check())
        
        mock_launch.return_value = None
        
        # Execute webchat command
        result = cli_runner.invoke(webchat, ["--host", "localhost", "--port", "8080"])
        
        # Should complete successfully
        assert result.exit_code == 0
        mock_launch.assert_called_once()

    @patch("oumi.webchat.interface.launch_webchat")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_startup_with_port_conflict(self, mock_check_port, mock_launch, cli_runner):
        """Test startup handling port conflicts."""
        # Simulate port conflict
        mock_check_port.side_effect = [False, True]  # First port busy, second available
        
        with patch("oumi.cli.webchat.find_available_port", return_value=8081):
            mock_launch.return_value = None
            
            result = cli_runner.invoke(webchat, ["--port", "8080"])
            
            # Should resolve conflict and succeed
            assert result.exit_code == 0
            mock_launch.assert_called_once()

    @patch("oumi.webchat.server.run_webchat_server")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_server_startup_failure_handling(self, mock_check_port, mock_run_server, cli_runner):
        """Test server startup failure handling."""
        mock_check_port.return_value = True
        mock_run_server.side_effect = Exception("Startup failed")
        
        result = cli_runner.invoke(webchat_server, ["--port", "8080"])
        
        # Should handle error gracefully
        assert result.exit_code != 0


class TestWebChatConfiguration:
    """Test webchat configuration handling."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @patch("oumi.webchat.interface.launch_webchat")
    @patch("oumi.cli.webchat.check_port_availability")
    def test_configuration_file_loading(self, mock_check_port, mock_launch, cli_runner):
        """Test loading configuration from file."""
        mock_check_port.return_value = True
        mock_launch.return_value = None
        
        # Mock config file existence and loading
        with patch("pathlib.Path.exists", return_value=True):
            result = cli_runner.invoke(webchat, ["-c", "test_config.yaml"])
            
            assert result.exit_code == 0
            mock_launch.assert_called_once()

    @patch("oumi.webchat.interface.launch_webchat")
    def test_default_configuration_values(self, mock_launch, cli_runner):
        """Test default configuration values."""
        mock_launch.return_value = None
        
        # Execute without explicit parameters
        result = cli_runner.invoke(webchat, [])
        
        # Should use default values
        assert result.exit_code == 0
        mock_launch.assert_called_once()
        
        # Check that launch was called (default values used internally)
        call_args = mock_launch.call_args
        assert call_args is not None

    def test_configuration_validation(self, cli_runner):
        """Test configuration parameter validation."""
        # Test invalid port (negative)
        result = cli_runner.invoke(webchat, ["--port", "-1"])
        assert result.exit_code != 0
        
        # Test invalid port (too high)
        result = cli_runner.invoke(webchat, ["--port", "99999"])
        assert result.exit_code != 0

    @patch("oumi.webchat.interface.launch_webchat")
    def test_environment_variable_configuration(self, mock_launch, cli_runner):
        """Test configuration via environment variables."""
        mock_launch.return_value = None
        
        # Mock environment variables
        with patch.dict("os.environ", {"WEBCHAT_HOST": "0.0.0.0", "WEBCHAT_PORT": "9000"}):
            # Note: Actual implementation may or may not support env vars
            # This test documents expected behavior
            result = cli_runner.invoke(webchat, [])
            
            assert result.exit_code == 0
            mock_launch.assert_called_once()


class TestWebChatIntegration:
    """Test WebChat CLI integration scenarios."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_wait_for_condition_utility(self):
        """Test wait_for_condition utility function."""
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

    @patch("oumi.webchat.interface.launch_webchat")
    @patch("oumi.webchat.server.run_webchat_server")
    def test_concurrent_command_execution(self, mock_run_server, mock_launch, cli_runner):
        """Test that CLI commands can handle concurrent execution scenarios."""
        # This tests the CLI's robustness when multiple commands might be executed
        mock_launch.return_value = None
        mock_run_server.return_value = None
        
        # Execute webchat command
        result1 = cli_runner.invoke(webchat, ["--port", "8080"])
        assert result1.exit_code == 0
        
        # Execute server command
        result2 = cli_runner.invoke(webchat_server, ["--port", "8081"])
        assert result2.exit_code == 0
        
        mock_launch.assert_called_once()
        mock_run_server.assert_called_once()