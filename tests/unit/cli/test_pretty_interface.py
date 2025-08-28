"""Tests for the pretty chat interface to prevent regressions."""

import inspect
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams


class TestPrettyInterfaceRegression:
    """Tests to ensure pretty interface features are present and working."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock inference config for testing."""
        return InferenceConfig(
            model=ModelParams(
                model_name="microsoft/DialoGPT-medium",  # Use an actual model for testing
                trust_remote_code=True,
            ),
            generation=GenerationParams(max_new_tokens=10),
        )

    def test_infer_interactive_uses_rich_console(self, mock_config):
        """Test that infer_interactive initializes Rich console components."""
        from oumi import infer_interactive

        with patch("oumi.infer.Console") as mock_console_class:
            with patch("oumi.infer.get_engine") as mock_get_engine:
                with patch("oumi.infer.SystemMonitor") as mock_monitor_class:
                    with patch("oumi.infer.EnhancedInput") as mock_enhanced_input:
                        mock_console = MagicMock()
                        mock_console_class.return_value = mock_console

                        mock_engine = MagicMock()
                        mock_get_engine.return_value = mock_engine

                        mock_monitor = MagicMock()
                        mock_monitor_class.return_value = mock_monitor

                        mock_enhanced_input_instance = MagicMock()
                        mock_enhanced_input.return_value = mock_enhanced_input_instance
                        mock_enhanced_input_instance.get_input.side_effect = EOFError

                        try:
                            infer_interactive(mock_config)
                        except EOFError:
                            pass  # Expected from input mock

                        # Verify Rich Console was initialized
                        mock_console_class.assert_called_once()

                        # Verify console methods were called (indicating pretty interface)
                        assert mock_console.print.called

    def test_infer_interactive_has_system_monitor(self, mock_config):
        """Test that system monitoring is initialized in pretty interface."""
        from oumi import infer_interactive

        with patch("oumi.infer.SystemMonitor") as mock_monitor_class:
            with patch("oumi.infer.Console"):
                with patch("oumi.infer.get_engine"):
                    with patch("oumi.infer.EnhancedInput") as mock_enhanced_input:
                        mock_monitor = MagicMock()
                        mock_monitor_class.return_value = mock_monitor

                        mock_enhanced_input_instance = MagicMock()
                        mock_enhanced_input.return_value = mock_enhanced_input_instance
                        mock_enhanced_input_instance.get_input.side_effect = EOFError

                        try:
                            infer_interactive(mock_config)
                        except EOFError:
                            pass  # Expected from input mock

                        # Verify SystemMonitor was initialized
                        mock_monitor_class.assert_called_once()

    def test_infer_interactive_has_thinking_animation(self, mock_config):
        """Test that thinking animation is present in pretty interface."""
        from oumi import infer_interactive
        from oumi.core.input import InputAction, InputResult

        with patch("oumi.infer.Live") as mock_live_class:
            with patch("oumi.infer.Console"):
                with patch("oumi.infer.get_engine") as mock_get_engine:
                    with patch("oumi.infer.SystemMonitor"):
                        mock_live = MagicMock()
                        mock_live_class.return_value = mock_live

                        # Mock engine to return a simple response
                        mock_engine = MagicMock()
                        mock_get_engine.return_value = mock_engine
                        
                        # Mock a simple conversation response
                        from oumi.core.types.conversation import Conversation, Message, Role
                        mock_response = [Conversation(messages=[
                            Message(role=Role.ASSISTANT, content="Hello!")
                        ])]
                        mock_engine.infer.return_value = mock_response

                        # Mock input to provide one message then exit
                        with patch("oumi.infer.EnhancedInput") as mock_enhanced_input:
                            mock_enhanced_input_instance = MagicMock()
                            mock_enhanced_input.return_value = (
                                mock_enhanced_input_instance
                            )
                            
                            # Provide one input, then EOF
                            input_results = [
                                InputResult(
                                    text="Hello",
                                    action=InputAction.SUBMIT,
                                    cancelled=False,
                                    should_exit=False,
                                    multiline_toggled=False
                                ),
                                InputResult(
                                    text="",
                                    action=InputAction.EXIT,
                                    cancelled=False,
                                    should_exit=True,
                                    multiline_toggled=False
                                )
                            ]
                            mock_enhanced_input_instance.get_input.side_effect = input_results

                            try:
                                infer_interactive(mock_config)
                            except (EOFError, SystemExit):
                                pass  # Expected from input mock

                        # Verify Live context manager was used (for thinking animation)
                        # Note: Live is used in the thinking_with_monitor context manager
                        mock_live_class.assert_called()

    def test_infer_interactive_has_command_parsing(self, mock_config):
        """Test that command parsing is present for /save, /load, etc."""
        from oumi import infer_interactive

        with patch("oumi.infer.CommandRouter") as mock_router_class:
            with patch("oumi.infer.Console"):
                with patch("oumi.infer.get_engine"):
                    with patch("oumi.infer.SystemMonitor"):
                        with patch("oumi.infer.EnhancedInput") as mock_enhanced_input:
                            mock_router = MagicMock()
                            mock_router_class.return_value = mock_router

                            mock_enhanced_input_instance = MagicMock()
                            mock_enhanced_input.return_value = mock_enhanced_input_instance
                            mock_enhanced_input_instance.get_input.side_effect = EOFError

                            try:
                                infer_interactive(mock_config)
                            except EOFError:
                                pass  # Expected from input mock

                            # Verify CommandRouter was initialized
                            mock_router_class.assert_called_once()

    def test_infer_interactive_imports_pretty_dependencies(self, mock_config):
        """Test that all pretty interface dependencies can be imported."""
        # This test ensures all the Rich/pretty interface imports are available
        from oumi.infer import (
            Console,  # Rich console
            Live,  # Thinking animation
            Panel,  # UI panels
            Text,  # Text rendering
            Spinner,  # Spinner animation
        )

        # If we get here without ImportError, the pretty interface imports are available
        assert Console is not None
        assert Live is not None
        assert Panel is not None
        assert Text is not None
        assert Spinner is not None

    def test_legacy_interface_not_used(self, mock_config):
        """Test that the legacy simple interface is NOT being used."""
        # Get the source code of infer_interactive
        import inspect

        import oumi.infer as infer_module

        source = inspect.getsource(infer_module.infer_interactive)

        # Legacy interface indicators that should NOT be present
        legacy_patterns = [
            'print(f"Assistant: {response}")',  # Simple print statements
            'input("You: ")',  # Basic input without enhancement
            # Note: removed "while True:" as pretty interface legitimately uses it
        ]

        # Pretty interface indicators that SHOULD be present
        pretty_patterns = [
            "Console(",  # Rich console
            "SystemMonitor(",  # System monitoring instantiation
            "thinking_with_monitor",  # Thinking animation
            "CommandRouter(",  # Command parsing instantiation
        ]

        # Check that legacy patterns are NOT present
        for pattern in legacy_patterns:
            assert pattern not in source, f"Legacy pattern found: {pattern}"

        # Check that pretty patterns ARE present
        for pattern in pretty_patterns:
            assert pattern in source, f"Pretty pattern missing: {pattern}"

    def test_cli_chat_command_uses_interactive(self):
        """Test that the CLI chat command properly calls infer_interactive."""
        import typer
        from typer.testing import CliRunner

        from oumi.cli.infer import chat

        app = typer.Typer()
        app.command()(chat)
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Create a minimal config
            config = InferenceConfig(
                model=ModelParams(model_name="test-model"),
                generation=GenerationParams(max_new_tokens=10),
            )
            config.to_yaml(config_path)

            with patch("oumi.infer_interactive") as mock_interactive:
                with patch("oumi.infer") as mock_infer:
                    # Test that chat command calls infer_interactive
                    runner.invoke(app, ["--config", str(config_path)])

                    # Should call infer_interactive, not infer
                    mock_interactive.assert_called_once()
                    mock_infer.assert_not_called()

    def test_regression_detection_via_function_signature(self):
        """Test function signature to detect if infer_interactive was replaced."""
        import inspect

        from oumi.infer import infer_interactive

        # Get the signature of infer_interactive
        sig = inspect.signature(infer_interactive)
        params = list(sig.parameters.keys())

        # The pretty interface should accept specific parameters
        expected_params = ["config", "input_image_bytes", "system_prompt"]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        # Should have reasonable number of lines (pretty interface is substantial)
        source = inspect.getsource(infer_interactive)
        line_count = len(source.split("\n"))

        # Pretty interface should be substantial (more than simple legacy version)
        assert line_count > 50, (
            f"infer_interactive too short ({line_count} lines), might be legacy version"
        )

    def test_auto_save_functionality_present(self, mock_config):
        """Test that auto-save functionality is available."""
        import oumi.infer as infer_module

        # Check that auto-save related code is present
        source = inspect.getsource(infer_module)

        auto_save_patterns = [
            "auto_save_chat",  # Auto-save functionality
            "Auto-save chat",  # Auto-save comments
            "file_operations_handler",  # File operations for saving
        ]

        for pattern in auto_save_patterns:
            assert pattern in source, f"Auto-save pattern missing: {pattern}"
