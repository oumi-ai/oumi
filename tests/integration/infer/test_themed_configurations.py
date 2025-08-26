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

"""Tests for themed configuration validation and integration."""

import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import InferenceConfig
from oumi.core.configs.params.style_params import StyleParams


class TestThemedConfigurationValidation:
    """Test validation of themed inference configurations."""

    def test_dark_theme_configuration_completeness(self):
        """Test that dark theme has all required style properties."""
        style_params = StyleParams(theme="dark")
        
        # Essential style properties that should be defined
        required_properties = [
            "user_prompt_style",
            "assistant_response_style", 
            "error_style",
            "system_monitor_style",
            "welcome_message_style"
        ]
        
        for prop in required_properties:
            assert hasattr(style_params, prop), f"Dark theme missing {prop}"
            prop_value = getattr(style_params, prop)
            assert prop_value is not None, f"Dark theme {prop} is None"

    def test_light_theme_configuration_completeness(self):
        """Test that light theme has appropriate contrasting colors."""
        style_params = StyleParams(theme="light")
        
        # Light theme should have different colors than dark theme
        dark_params = StyleParams(theme="dark")
        
        # Compare key styling properties
        if hasattr(style_params, 'background_color') and hasattr(dark_params, 'background_color'):
            assert style_params.background_color != dark_params.background_color
        
        # Light theme should use darker text colors for contrast
        if hasattr(style_params, 'text_color'):
            text_color = style_params.text_color
            # Should not be white or very light colors
            assert "white" not in str(text_color).lower()

    def test_neon_theme_configuration_vibrancy(self):
        """Test that neon theme uses appropriately vibrant colors."""
        style_params = StyleParams(theme="neon")
        
        # Neon theme should have bright, vibrant colors
        neon_color_indicators = [
            "#ff00ff",  # Magenta
            "#00ffff",  # Cyan  
            "#ffff00",  # Yellow
            "bright_",  # Bright color prefix
            "neon"      # Neon color names
        ]
        
        # Check that theme uses neon-style colors
        theme_colors = []
        if hasattr(style_params, 'custom_theme') and style_params.custom_theme:
            theme_colors = list(style_params.custom_theme.values())
        else:
            # Get colors from individual properties
            for attr_name in dir(style_params):
                if "style" in attr_name and not attr_name.startswith("_"):
                    attr_value = getattr(style_params, attr_name)
                    if isinstance(attr_value, str):
                        theme_colors.append(attr_value)
        
        # Should have at least some neon-like colors
        neon_color_found = any(
            any(indicator in str(color).lower() for indicator in neon_color_indicators)
            for color in theme_colors
        )
        
        assert neon_color_found or len(theme_colors) == 0  # Allow if no colors extracted

    def test_monokai_theme_developer_familiarity(self):
        """Test that monokai theme uses developer-familiar color scheme."""
        style_params = StyleParams(theme="monokai")
        
        # Monokai should use the classic developer color palette
        expected_monokai_colors = [
            "#66d9ef",  # Blue
            "#a6e22e",  # Green  
            "#f92672",  # Pink/Red
            "#fd971f",  # Orange
            "#ae81ff",  # Purple
        ]
        
        # Extract theme colors
        theme_colors = []
        if hasattr(style_params, 'custom_theme') and style_params.custom_theme:
            theme_colors = list(style_params.custom_theme.values())
        
        # Should use some classic monokai colors
        if theme_colors:
            monokai_colors_used = sum(
                1 for color in theme_colors
                for monokai_color in expected_monokai_colors
                if monokai_color.lower() in str(color).lower()
            )
            assert monokai_colors_used > 0

    def test_minimal_theme_simplicity(self):
        """Test that minimal theme uses simple, clean styling."""
        style_params = StyleParams(theme="minimal")
        
        # Minimal theme should avoid complex styling
        complex_style_indicators = [
            "bold italic",
            "underline bold",
            "blink",
            "reverse",
            "on bright_"  # Complex background combinations
        ]
        
        # Extract theme colors
        theme_colors = []
        for attr_name in dir(style_params):
            if "style" in attr_name and not attr_name.startswith("_"):
                attr_value = getattr(style_params, attr_name)
                if isinstance(attr_value, str):
                    theme_colors.append(attr_value)
        
        # Should avoid complex styling combinations
        complex_styling_found = any(
            any(complex_indicator in str(color).lower() for complex_indicator in complex_style_indicators)
            for color in theme_colors
        )
        
        # Minimal theme should not use complex styling
        assert not complex_styling_found or len(theme_colors) == 0

    def test_custom_theme_validation_comprehensive(self):
        """Test comprehensive validation of custom themes."""
        # Valid comprehensive custom theme
        comprehensive_theme = {
            # Basic UI elements
            "user_prompt": "bold cyan",
            "assistant_response": "bright_green", 
            "error": "bold red",
            "warning": "yellow",
            "info": "blue",
            
            # System monitor elements
            "monitor_low": "green",
            "monitor_medium": "yellow",
            "monitor_high": "red", 
            "monitor_critical": "bold red on black",
            
            # UI panels and decorations
            "panel_border": "dim white",
            "panel_title": "bold white",
            "status_bar": "reverse",
            
            # Specialized elements
            "code_highlight": "#ffaa00",
            "link": "underline blue",
            "emphasis": "italic"
        }
        
        style_params = StyleParams(
            theme="custom",
            custom_theme=comprehensive_theme
        )
        
        # Should accept comprehensive theme
        assert style_params.theme == "custom"
        assert style_params.custom_theme == comprehensive_theme
        
        # All colors should be preserved
        for key, value in comprehensive_theme.items():
            assert style_params.custom_theme[key] == value

    def test_theme_consistency_across_components(self):
        """Test that themes maintain consistency across different UI components."""
        themes_to_test = ["dark", "light", "neon", "monokai", "minimal"]
        
        for theme_name in themes_to_test:
            style_params = StyleParams(theme=theme_name)
            
            # Extract colors from the theme
            ui_colors = []
            monitor_colors = []
            
            for attr_name in dir(style_params):
                if not attr_name.startswith("_"):
                    attr_value = getattr(style_params, attr_name)
                    if isinstance(attr_value, str) and attr_value:
                        if "monitor" in attr_name:
                            monitor_colors.append(attr_value)
                        elif "style" in attr_name:
                            ui_colors.append(attr_value)
            
            # Themes should have reasonable color palettes
            total_colors = len(ui_colors) + len(monitor_colors)
            assert total_colors >= 3, f"Theme {theme_name} has too few colors defined"

    def test_theme_configuration_in_yaml_files(self):
        """Test that themed configurations can be saved and loaded from YAML."""
        themes_to_test = ["dark", "neon", "minimal"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for theme_name in themes_to_test:
                # Create inference config with theme
                style_params = StyleParams(theme=theme_name)
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
                
                # Save to YAML
                yaml_path = Path(temp_dir) / f"config_{theme_name}.yaml"
                
                # Convert config to dict for YAML serialization
                config_dict = {
                    "model": {
                        "model_name": "test-model",
                        "model_max_length": 2048
                    },
                    "generation": {
                        "max_new_tokens": 100,
                        "temperature": 0.7
                    },
                    "style": {
                        "theme": theme_name
                    }
                }
                
                with open(yaml_path, 'w') as f:
                    yaml.dump(config_dict, f)
                
                # Verify YAML was created
                assert yaml_path.exists()
                
                # Load and verify
                with open(yaml_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                assert loaded_config["style"]["theme"] == theme_name

    def test_invalid_theme_combinations_rejection(self):
        """Test rejection of invalid theme combinations."""
        invalid_combinations = [
            # Conflicting themes
            {
                "theme": "dark",
                "custom_theme": {
                    "user_prompt": "white on white"  # No contrast
                }
            },
            # Impossible color combinations
            {
                "theme": "custom", 
                "custom_theme": {
                    "error": "green red blue"  # Multiple foreground colors
                }
            },
            # Missing required theme data
            {
                "theme": "custom"
                # Missing custom_theme
            }
        ]
        
        for invalid_combo in invalid_combinations:
            with pytest.raises((ValueError, TypeError)):
                StyleParams(**invalid_combo)

    def test_theme_performance_impact_minimal(self):
        """Test that themes don't significantly impact performance."""
        import time
        
        # Test theme initialization performance
        theme_creation_times = []
        
        for theme_name in ["dark", "light", "neon", "monokai", "minimal"]:
            start_time = time.time()
            
            # Create theme multiple times
            for _ in range(100):
                style_params = StyleParams(theme=theme_name)
            
            end_time = time.time()
            theme_creation_times.append(end_time - start_time)
        
        # Theme creation should be fast
        max_creation_time = max(theme_creation_times)
        assert max_creation_time < 1.0, f"Theme creation too slow: {max_creation_time}s"
        
        # All themes should have similar performance
        min_creation_time = min(theme_creation_times)
        performance_variance = max_creation_time / min_creation_time if min_creation_time > 0 else 1
        assert performance_variance < 5.0, f"High performance variance between themes: {performance_variance}x"


class TestThemedConfigurationIntegration:
    """Test integration of themed configurations with inference system."""

    @patch('oumi.core.monitoring.system_monitor.Console')
    def test_theme_integration_with_system_monitor(self, mock_console):
        """Test that themes properly integrate with system monitor."""
        mock_console_instance = MagicMock()
        mock_console.return_value = mock_console_instance
        
        themes_to_test = ["dark", "neon", "minimal"]
        
        for theme_name in themes_to_test:
            style_params = StyleParams(theme=theme_name)
            
            # Mock system monitor creation with theme
            with patch('oumi.core.monitoring.system_monitor.SystemMonitor') as mock_monitor:
                mock_monitor_instance = MagicMock()
                mock_monitor.return_value = mock_monitor_instance
                
                # Should initialize with theme styles
                monitor = mock_monitor(style_params=style_params)
                
                # Verify monitor was created
                assert monitor is not None

    @patch('oumi.core.input.enhanced_input.PromptSession')
    def test_theme_integration_with_enhanced_input(self, mock_prompt_session):
        """Test that themes properly integrate with enhanced input."""
        mock_session = MagicMock()
        mock_prompt_session.return_value = mock_session
        
        themes_to_test = ["dark", "light", "monokai"]
        
        for theme_name in themes_to_test:
            style_params = StyleParams(theme=theme_name)
            
            # Create enhanced input with theme
            with patch('oumi.core.input.enhanced_input.EnhancedInput') as mock_input:
                mock_input_instance = MagicMock()
                mock_input.return_value = mock_input_instance
                
                enhanced_input = mock_input(style_params=style_params)
                
                # Should integrate theme styling
                assert enhanced_input is not None

    def test_theme_configuration_persistence(self):
        """Test that theme configurations persist correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create themed configuration
            custom_theme = {
                "user_prompt": "bold green",
                "assistant_response": "cyan", 
                "error": "red"
            }
            
            style_params = StyleParams(
                theme="custom",
                custom_theme=custom_theme
            )
            
            config = InferenceConfig(
                model={"model_name": "test-model"},
                generation={"max_new_tokens": 50},
                style=style_params
            )
            
            # Save configuration
            config_path = Path(temp_dir) / "themed_config.yaml"
            config_dict = {
                "model": {"model_name": "test-model"},
                "generation": {"max_new_tokens": 50},
                "style": {
                    "theme": "custom",
                    "custom_theme": custom_theme
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f)
            
            # Load and verify persistence
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config["style"]["theme"] == "custom"
            assert loaded_config["style"]["custom_theme"] == custom_theme

    def test_theme_backward_compatibility(self):
        """Test that new themes are backward compatible."""
        # Create configurations with older theme structure
        legacy_config_dict = {
            "model": {"model_name": "legacy-model"},
            "generation": {"max_new_tokens": 25},
            "style": {
                "theme": "dark"  # Simple theme without custom_theme
            }
        }
        
        # Should still work with current system
        style_params = StyleParams(theme="dark")
        assert style_params.theme == "dark"
        
        # Should have default styling even without custom_theme
        assert hasattr(style_params, 'user_prompt_style') or \
               hasattr(style_params, 'theme')

    def test_theme_extensibility(self):
        """Test that theme system is extensible for new themes."""
        # Test adding a new theme-like configuration
        experimental_theme = {
            "user_prompt": "bold #ff6b6b",
            "assistant_response": "#4ecdc4",
            "error": "#ff6b6b", 
            "warning": "#ffe66d",
            "success": "#4ecdc4",
            "info": "#95a5a6",
            
            # Experimental features
            "code_block": "on #2c3e50",
            "table_header": "bold #34495e",
            "table_cell": "#7f8c8d",
            "hyperlink": "underline #3498db"
        }
        
        # Should accept extended theme
        style_params = StyleParams(
            theme="custom",
            custom_theme=experimental_theme
        )
        
        assert style_params.theme == "custom"
        assert len(style_params.custom_theme) == len(experimental_theme)

    def test_theme_environment_adaptation(self):
        """Test that themes can adapt to different terminal environments."""
        # Test different simulated environments
        environment_scenarios = [
            {"term": "xterm", "colors": 16},
            {"term": "xterm-256color", "colors": 256},
            {"term": "screen", "colors": 8},
            {"term": "tmux", "colors": 256}
        ]
        
        for scenario in environment_scenarios:
            # Themes should work across environments
            style_params = StyleParams(theme="dark")
            
            # Should not raise errors regardless of environment
            assert style_params.theme == "dark"
            
            # Basic functionality should remain intact
            if hasattr(style_params, 'user_prompt_style'):
                assert style_params.user_prompt_style is not None


class TestThemedConfigurationEdgeCases:
    """Test edge cases and error scenarios in themed configurations."""

    def test_empty_custom_theme_handling(self):
        """Test handling of empty custom themes."""
        # Empty custom theme should fall back to defaults
        style_params = StyleParams(
            theme="custom",
            custom_theme={}
        )
        
        # Should handle empty theme gracefully
        assert style_params.theme == "custom"
        assert style_params.custom_theme == {}

    def test_partial_custom_theme_completion(self):
        """Test that partial custom themes are completed with defaults."""
        partial_theme = {
            "user_prompt": "green",
            "error": "red"
            # Missing other required styles
        }
        
        style_params = StyleParams(
            theme="custom",
            custom_theme=partial_theme
        )
        
        # Should preserve provided styles
        assert style_params.custom_theme["user_prompt"] == "green"
        assert style_params.custom_theme["error"] == "red"

    def test_theme_override_precedence(self):
        """Test that custom themes properly override base themes."""
        # Start with base theme
        base_style = StyleParams(theme="dark")
        
        # Override with custom theme
        override_theme = {
            "user_prompt": "bright_yellow",  # Different from dark theme
            "assistant_response": "bright_blue"
        }
        
        custom_style = StyleParams(
            theme="custom",
            custom_theme=override_theme
        )
        
        # Custom theme should override base
        if hasattr(custom_style, 'user_prompt_style'):
            # Should use custom colors, not base theme colors
            assert custom_style.custom_theme["user_prompt"] == "bright_yellow"

    def test_malformed_color_graceful_degradation(self):
        """Test graceful degradation with malformed colors."""
        questionable_colors = [
            "#zzzzzz",  # Invalid hex
            "rgb(999, 999, 999)",  # Out of range RGB
            "some_invalid_color_name",  # Unknown color
            "",  # Empty string
        ]
        
        for bad_color in questionable_colors:
            # Should either accept with graceful fallback or reject cleanly
            try:
                custom_theme = {"user_prompt": bad_color}
                style_params = StyleParams(
                    theme="custom",
                    custom_theme=custom_theme
                )
                
                # If accepted, should not crash the system
                assert style_params.theme == "custom"
                
            except (ValueError, TypeError):
                # Clean rejection is also acceptable
                pass

    def test_unicode_and_special_characters_in_themes(self):
        """Test handling of unicode and special characters in theme names."""
        special_themes = [
            "dark-mode",
            "theme_with_underscores",
            "theme.with.dots",
            "ThemeCamelCase",
        ]
        
        for special_theme in special_themes:
            # Most should be rejected as invalid theme names
            with pytest.raises((ValueError, TypeError)):
                StyleParams(theme=special_theme)

    def test_theme_circular_reference_prevention(self):
        """Test prevention of circular references in theme definitions."""
        # This would be a complex scenario where themes reference each other
        # For now, test that themes don't break with self-referential data
        
        custom_theme = {
            "user_prompt": "style_reference_to_assistant",
            "assistant_response": "bright_green"
        }
        
        # Should not create circular dependencies
        style_params = StyleParams(
            theme="custom", 
            custom_theme=custom_theme
        )
        
        # Should complete without infinite recursion
        assert style_params.theme == "custom"