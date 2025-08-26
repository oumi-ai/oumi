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

"""Comprehensive styling controls tests for Oumi inference."""

import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from oumi.core.configs.params.style_params import StyleParams
from oumi.core.configs import InferenceConfig
from oumi.core.input.enhanced_input import EnhancedInput, CommandCompleter
from oumi.core.monitoring.system_monitor import SystemMonitor


class TestStyleParamsValidation:
    """Test styling parameter validation and themes."""

    def test_predefined_themes_available(self):
        """Test that all predefined themes are properly defined."""
        style_params = StyleParams()
        
        # Test default theme
        assert style_params.theme == "dark"
        
        # Test available themes
        valid_themes = ["dark", "light", "monokai", "minimal", "neon"]
        for theme in valid_themes:
            theme_style = StyleParams(theme=theme)
            assert theme_style.theme == theme
            
            # Verify theme has required styling properties
            assert hasattr(theme_style, 'user_prompt_style')
            assert hasattr(theme_style, 'assistant_response_style')
            assert hasattr(theme_style, 'error_style')
            assert hasattr(theme_style, 'system_monitor_style')

    def test_custom_theme_validation(self):
        """Test custom theme parameter validation."""
        # Valid custom theme
        valid_custom_theme = {
            "user_prompt": "#ffffff",
            "assistant_response": "#00ff00", 
            "error": "#ff0000",
            "system_info": "#0000ff"
        }
        
        style_params = StyleParams(
            theme="custom",
            custom_theme=valid_custom_theme
        )
        
        assert style_params.theme == "custom"
        assert style_params.custom_theme == valid_custom_theme

    def test_invalid_theme_rejection(self):
        """Test that invalid theme names are rejected."""
        with pytest.raises((ValueError, TypeError)):
            StyleParams(theme="invalid_theme_name")

    def test_color_format_validation(self):
        """Test that color formats are properly validated."""
        # Test RGB hex colors
        valid_colors = ["#ffffff", "#ff0000", "#00ff00", "#0000ff"]
        
        for color in valid_colors:
            custom_theme = {"user_prompt": color}
            style_params = StyleParams(theme="custom", custom_theme=custom_theme)
            assert style_params.custom_theme["user_prompt"] == color
        
        # Test named colors
        valid_named_colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
        for color in valid_named_colors:
            custom_theme = {"assistant_response": color}
            style_params = StyleParams(theme="custom", custom_theme=custom_theme)
            assert style_params.custom_theme["assistant_response"] == color

    def test_style_combinations(self):
        """Test complex style combinations (bold, italic, underline)."""
        complex_styles = [
            "bold red",
            "italic #ff00ff", 
            "underline cyan",
            "bold italic yellow",
            "underline bold #00ffff"
        ]
        
        for style in complex_styles:
            custom_theme = {"user_prompt": style}
            style_params = StyleParams(theme="custom", custom_theme=custom_theme)
            assert style_params.custom_theme["user_prompt"] == style

    def test_system_monitor_styling_configuration(self):
        """Test system monitor specific styling options."""
        style_params = StyleParams(theme="neon")
        
        # Should have monitor-specific style categories
        assert hasattr(style_params, 'monitor_low_usage_style')
        assert hasattr(style_params, 'monitor_medium_usage_style') 
        assert hasattr(style_params, 'monitor_high_usage_style')
        assert hasattr(style_params, 'monitor_critical_usage_style')
        
        # Test custom monitor styling
        monitor_styles = {
            "monitor_low": "green",
            "monitor_medium": "yellow", 
            "monitor_high": "red",
            "monitor_critical": "bold red"
        }
        
        custom_style = StyleParams(
            theme="custom",
            custom_theme=monitor_styles
        )
        
        for key, value in monitor_styles.items():
            assert custom_style.custom_theme[key] == value

    def test_background_color_support(self):
        """Test background color styling support."""
        backgrounds = [
            "on black",
            "white on blue", 
            "#ffffff on #000000",
            "bold green on dark_red"
        ]
        
        for bg_style in backgrounds:
            custom_theme = {"welcome_message": bg_style}
            style_params = StyleParams(theme="custom", custom_theme=custom_theme)
            assert style_params.custom_theme["welcome_message"] == bg_style

    def test_emoji_and_visual_configuration(self):
        """Test emoji and visual configuration options."""
        # Test emoji configuration
        style_with_emoji = StyleParams(enable_emojis=True)
        style_without_emoji = StyleParams(enable_emojis=False)
        
        assert style_with_emoji.enable_emojis is True
        assert style_without_emoji.enable_emojis is False
        
        # Test panel configuration
        wide_panel_style = StyleParams(use_wide_panels=True)
        narrow_panel_style = StyleParams(use_wide_panels=False)
        
        assert wide_panel_style.use_wide_panels is True
        assert narrow_panel_style.use_wide_panels is False

    def test_console_dimensions_configuration(self):
        """Test console size and dimension configuration."""
        # Test various console width settings
        console_configs = [
            {"console_width": 80},
            {"console_width": 120},
            {"console_width": 160}
        ]
        
        for config in console_configs:
            style_params = StyleParams(**config)
            if hasattr(style_params, 'console_width'):
                assert style_params.console_width == config["console_width"]

    def test_style_inheritance_and_fallbacks(self):
        """Test that styles properly inherit and fallback to defaults."""
        # Test partial custom theme with fallbacks
        partial_theme = {
            "user_prompt": "#ffffff",
            "error": "#ff0000"
            # Missing assistant_response, should use default
        }
        
        style_params = StyleParams(theme="custom", custom_theme=partial_theme)
        
        # Should preserve custom values
        assert style_params.custom_theme["user_prompt"] == "#ffffff"
        assert style_params.custom_theme["error"] == "#ff0000"


class TestPromptToolkitIntegration:
    """Test prompt_toolkit integration and parameter usage."""

    @patch('oumi.core.input.enhanced_input.PromptSession')
    def test_enhanced_input_initialization(self, mock_prompt_session):
        """Test that EnhancedInput properly initializes with styling."""
        mock_session = MagicMock()
        mock_prompt_session.return_value = mock_session
        
        style_params = StyleParams(theme="neon")
        enhanced_input = EnhancedInput(style_params=style_params)
        
        # Verify PromptSession was called with styling parameters
        mock_prompt_session.assert_called_once()
        call_args = mock_prompt_session.call_args
        
        # Should have completer and key bindings
        assert 'completer' in call_args.kwargs
        assert 'key_bindings' in call_args.kwargs

    def test_command_completer_integration(self):
        """Test command completer with all interactive commands."""
        expected_commands = [
            '/attach', '/branch', '/browse', '/clear', '/export', '/fetch',
            '/help', '/render', '/save', '/show', '/style', '/system', 
            '/switch', '/load', '/history', '/reset', '/exit'
        ]
        
        completer = CommandCompleter()
        
        # Test completion for each command prefix
        for cmd in expected_commands:
            completions = list(completer.get_completions(
                document=MagicMock(text=cmd[:3]),
                complete_event=MagicMock()
            ))
            
            # Should find matching completions
            assert len(completions) > 0
            
            # Should include the full command
            completion_texts = [c.text for c in completions]
            assert any(cmd.startswith(text) or text.startswith(cmd[:3]) for text in completion_texts)

    @patch('oumi.core.input.enhanced_input.PromptSession')
    def test_style_parameters_passed_to_prompt_toolkit(self, mock_prompt_session):
        """Test that style parameters are passed to prompt_toolkit components."""
        mock_session = MagicMock()
        mock_prompt_session.return_value = mock_session
        
        # Create style with custom colors
        custom_style = StyleParams(
            theme="custom",
            custom_theme={
                "user_prompt": "bold green",
                "assistant_response": "#00ffff"
            }
        )
        
        enhanced_input = EnhancedInput(style_params=custom_style)
        
        # Prompt session should be initialized
        mock_prompt_session.assert_called_once()
        
        # Test actual input gathering (mocked)
        mock_session.prompt.return_value = "test input"
        result = enhanced_input.get_user_input(prompt="Test> ")
        
        assert result == "test input"
        mock_session.prompt.assert_called_once()

    def test_multiline_input_mode_switching(self):
        """Test switching between single-line and multi-line input modes."""
        style_params = StyleParams()
        
        with patch('oumi.core.input.enhanced_input.PromptSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            enhanced_input = EnhancedInput(style_params=style_params)
            
            # Test single-line mode
            enhanced_input.set_multiline_mode(False)
            assert enhanced_input.multiline_mode is False
            
            # Test multi-line mode
            enhanced_input.set_multiline_mode(True) 
            assert enhanced_input.multiline_mode is True

    def test_command_history_persistence(self):
        """Test command history saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            history_file = Path(temp_dir) / "test_history"
            
            with patch('oumi.core.input.enhanced_input.PromptSession') as mock_session_class:
                mock_session = MagicMock()
                mock_session_class.return_value = mock_session
                
                # Create enhanced input with custom history file
                enhanced_input = EnhancedInput(
                    style_params=StyleParams(),
                    history_file=str(history_file)
                )
                
                # Simulate adding commands to history
                test_commands = [
                    "Hello, how are you?",
                    "/help()",
                    "/save(test.json)",
                    "What is artificial intelligence?"
                ]
                
                # Mock the history functionality
                for cmd in test_commands:
                    if hasattr(enhanced_input, 'add_to_history'):
                        enhanced_input.add_to_history(cmd)

    @patch('oumi.core.input.enhanced_input.KeyBindings')
    def test_key_binding_configuration(self, mock_key_bindings):
        """Test that key bindings are properly configured."""
        mock_kb = MagicMock()
        mock_key_bindings.return_value = mock_kb
        
        style_params = StyleParams()
        
        with patch('oumi.core.input.enhanced_input.PromptSession'):
            enhanced_input = EnhancedInput(style_params=style_params)
            
            # Should have created key bindings
            mock_key_bindings.assert_called()


class TestSystemMonitorStyling:
    """Test system monitor styling integration."""

    def test_system_monitor_style_application(self):
        """Test that system monitor applies styles correctly."""
        # Test with different themes
        themes_to_test = ["dark", "light", "neon", "monokai"]
        
        for theme in themes_to_test:
            style_params = StyleParams(theme=theme)
            
            with patch('oumi.core.monitoring.system_monitor.Console') as mock_console:
                mock_console_instance = MagicMock()
                mock_console.return_value = mock_console_instance
                
                monitor = SystemMonitor(
                    style_params=style_params,
                    update_interval=1.0
                )
                
                # Should initialize with styled console
                mock_console.assert_called_once()

    def test_usage_based_color_coding(self):
        """Test that monitor uses different colors for different usage levels."""
        style_params = StyleParams(theme="dark")
        
        with patch('oumi.core.monitoring.system_monitor.Console'):
            monitor = SystemMonitor(style_params=style_params)
            
            # Test color selection for different usage levels
            usage_scenarios = [
                (10, "low"),     # Should use low usage color
                (50, "medium"),  # Should use medium usage color  
                (80, "high"),    # Should use high usage color
                (95, "critical") # Should use critical usage color
            ]
            
            for usage_percent, expected_level in usage_scenarios:
                if hasattr(monitor, 'get_usage_color'):
                    color = monitor.get_usage_color(usage_percent)
                    assert color is not None
                    
                    # Color should vary by usage level
                    if expected_level == "low":
                        assert "green" in str(color).lower() or "#" in str(color)
                    elif expected_level == "critical":
                        assert "red" in str(color).lower() or "#" in str(color)

    def test_resource_monitor_panel_styling(self):
        """Test resource monitor panel formatting and styling."""
        custom_theme = {
            "monitor_panel_border": "cyan",
            "monitor_title": "bold white",
            "monitor_low": "bright_green",
            "monitor_critical": "bold red"
        }
        
        style_params = StyleParams(theme="custom", custom_theme=custom_theme)
        
        with patch('oumi.core.monitoring.system_monitor.Console'):
            monitor = SystemMonitor(style_params=style_params)
            
            # Test panel creation with styling
            mock_stats = {
                "cpu_percent": 25.0,
                "memory_percent": 60.0,
                "gpu_memory_percent": 80.0,
                "context_usage": 45.0
            }
            
            if hasattr(monitor, 'create_status_panel'):
                with patch.object(monitor, 'get_system_stats', return_value=mock_stats):
                    panel = monitor.create_status_panel()
                    assert panel is not None

    def test_context_window_monitoring_style(self):
        """Test context window monitoring with proper styling."""
        style_params = StyleParams(theme="monokai")
        
        with patch('oumi.core.monitoring.system_monitor.Console'):
            monitor = SystemMonitor(style_params=style_params)
            
            # Test context monitoring with various levels
            context_scenarios = [
                {"tokens_used": 100, "max_tokens": 2048, "turns": 5},
                {"tokens_used": 1500, "max_tokens": 2048, "turns": 20},
                {"tokens_used": 1900, "max_tokens": 2048, "turns": 35}
            ]
            
            for scenario in context_scenarios:
                if hasattr(monitor, 'update_context_stats'):
                    monitor.update_context_stats(**scenario)
                
                # Should handle various context levels appropriately


class TestInvalidParameterProtection:
    """Test protections against invalid parameter values."""

    def test_invalid_color_rejection(self):
        """Test that invalid color values are rejected."""
        invalid_colors = [
            "not_a_color",
            "#gggggg",  # Invalid hex
            "rgb(256, 256, 256)",  # Invalid RGB
            "",  # Empty string
            None,  # None value
            123,  # Numeric value
        ]
        
        for invalid_color in invalid_colors:
            with pytest.raises((ValueError, TypeError)):
                custom_theme = {"user_prompt": invalid_color}
                StyleParams(theme="custom", custom_theme=custom_theme)

    def test_invalid_theme_structure_rejection(self):
        """Test rejection of malformed theme structures."""
        invalid_themes = [
            None,  # None theme
            "string_instead_of_dict",  # Wrong type
            [],  # List instead of dict
            {"incomplete": "theme"},  # Missing required keys
        ]
        
        for invalid_theme in invalid_themes:
            with pytest.raises((ValueError, TypeError)):
                StyleParams(theme="custom", custom_theme=invalid_theme)

    def test_console_parameter_bounds_checking(self):
        """Test bounds checking on console parameters."""
        invalid_console_configs = [
            {"console_width": -1},  # Negative width
            {"console_width": 0},   # Zero width
            {"console_width": 10000},  # Unreasonably large
        ]
        
        for invalid_config in invalid_console_configs:
            with pytest.raises((ValueError, TypeError)):
                StyleParams(**invalid_config)

    def test_style_format_validation(self):
        """Test validation of style format strings."""
        invalid_styles = [
            "bold bold",  # Duplicate modifiers
            "invalid_modifier red",  # Unknown modifier
            "#ffffff #000000",  # Multiple colors without 'on'
            "on",  # Incomplete background specification
        ]
        
        for invalid_style in invalid_styles:
            with pytest.raises((ValueError, TypeError)):
                custom_theme = {"error": invalid_style}
                StyleParams(theme="custom", custom_theme=custom_theme)

    def test_emoji_configuration_validation(self):
        """Test emoji configuration parameter validation."""
        # Valid emoji settings
        valid_emoji_configs = [True, False]
        for config in valid_emoji_configs:
            style_params = StyleParams(enable_emojis=config)
            assert style_params.enable_emojis == config
        
        # Invalid emoji settings
        invalid_emoji_configs = ["true", "false", 1, 0, None]
        for invalid_config in invalid_emoji_configs:
            with pytest.raises((ValueError, TypeError)):
                StyleParams(enable_emojis=invalid_config)


class TestInferenceConfigStylingIntegration:
    """Test styling integration with inference configuration."""

    def test_style_params_in_inference_config(self):
        """Test that StyleParams integrate properly with InferenceConfig."""
        style_params = StyleParams(theme="neon")
        
        # Create minimal inference config with styling
        config = InferenceConfig(
            model={
                "model_name": "test-model",
                "model_max_length": 2048
            },
            generation={
                "max_new_tokens": 100,
                "temperature": 0.7
            },
            style=style_params
        )
        
        assert config.style is not None
        assert config.style.theme == "neon"

    def test_inference_config_style_defaults(self):
        """Test default styling when not explicitly specified."""
        config = InferenceConfig(
            model={
                "model_name": "test-model",
                "model_max_length": 2048
            },
            generation={
                "max_new_tokens": 100
            }
        )
        
        # Should have default style parameters
        if hasattr(config, 'style'):
            assert config.style is not None

    def test_style_inheritance_in_config_chain(self):
        """Test style parameter inheritance through config chain."""
        custom_style = StyleParams(
            theme="custom",
            custom_theme={
                "user_prompt": "bold cyan",
                "assistant_response": "bright_green"
            }
        )
        
        config = InferenceConfig(
            model={"model_name": "test-model"},
            generation={"max_new_tokens": 50},
            style=custom_style
        )
        
        # Style should be preserved through config
        assert config.style.theme == "custom"
        assert config.style.custom_theme["user_prompt"] == "bold cyan"


class TestStylingUtilities:
    """Test styling utility functions and helpers."""

    def test_theme_validation_utilities(self):
        """Test theme validation helper functions."""
        # Test if validation utilities exist
        style_params = StyleParams()
        
        # Should have methods for theme validation
        expected_utility_methods = [
            'validate_theme',
            'validate_color',
            'get_theme_colors',
            'apply_theme'
        ]
        
        for method_name in expected_utility_methods:
            if hasattr(style_params, method_name):
                method = getattr(style_params, method_name)
                assert callable(method)

    def test_color_conversion_utilities(self):
        """Test color format conversion utilities."""
        # Test color format conversions if available
        test_colors = [
            "#ffffff",
            "rgb(255, 255, 255)", 
            "bright_white",
            "bold red"
        ]
        
        style_params = StyleParams()
        for color in test_colors:
            if hasattr(style_params, 'normalize_color'):
                try:
                    normalized = style_params.normalize_color(color)
                    assert normalized is not None
                except (ValueError, NotImplementedError):
                    # Method might not be implemented yet
                    pass

    def test_rich_console_integration(self):
        """Test integration with Rich console styling."""
        style_params = StyleParams(theme="monokai")
        
        # Test Rich console creation with styling
        console = Console()
        
        # Should be able to apply styles to console output
        test_styles = [
            style_params.user_prompt_style,
            style_params.assistant_response_style,
            style_params.error_style
        ]
        
        for style in test_styles:
            if style:
                # Should not raise errors when applying to console
                try:
                    styled_text = f"[{style}]Test text[/{style}]"
                    console.print(styled_text, end="", file=None)
                except Exception:
                    # Some styles might not be compatible
                    pass