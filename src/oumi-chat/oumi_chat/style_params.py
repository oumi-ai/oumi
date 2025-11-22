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

from dataclasses import dataclass
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class StyleParams(BaseParams):
    """Simplified styling parameters for Rich console in interactive inference.

    This uses a theme-based approach with sensible defaults. Most users should
    just set theme_name to one of the predefined themes.

    Rich style format examples:
    - Colors: 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white'
    - Bright variants: 'bright_red', 'bright_green', etc.
    - RGB: '#af00ff' or 'rgb(175,0,255)'
    - Attributes: 'bold', 'italic', 'underline', 'dim'
    - Combined: 'bold red', 'dim cyan', etc.

    Examples:
        # Use a predefined theme
        style:
          theme_name: "dark"

        # Disable colors
        style:
          no_color: true

        # Customize key elements
        style:
          user_style: "bold green"
          assistant_style: "cyan"
          error_style: "red"
    """

    # Theme selection
    theme_name: Optional[str] = None
    """Predefined theme: 'dark', 'light', 'monokai', 'minimal', or 'neon'."""

    custom_theme: Optional[dict[str, str]] = None
    """Custom theme dictionary mapping semantic names to Rich styles."""

    # Core UI styles (override theme defaults)
    user_style: str = "bold blue"
    """Style for user prompts and input."""

    assistant_style: str = "cyan"
    """Style for assistant responses (titles and borders)."""

    assistant_text_style: str = "white"
    """Style for assistant response text content."""

    error_style: str = "red"
    """Style for errors (titles and borders)."""

    info_style: str = "green"
    """Style for info messages and status indicators."""

    # Code highlighting
    code_theme: str = "monokai"
    """Syntax highlighting theme: 'monokai', 'github-dark', 'dracula', 'nord'."""

    # Panel configuration
    panel_box_style: str = "rounded"
    """Panel border style: 'rounded', 'square', 'double', 'heavy', 'ascii'."""

    expand_panels: bool = False
    """Whether panels should expand to full terminal width."""

    assistant_padding: tuple[int, int] = (1, 2)
    """Panel padding as (vertical, horizontal) tuple."""

    # Display options
    no_color: bool = False
    """Disable all colors and styling."""

    use_emoji: bool = True
    """Enable emoji in output."""

    width: Optional[int] = None
    """Console width override (default: auto-detect)."""

    def get_style(self, element: str) -> str:
        """Get the resolved style for a UI element.

        This method resolves styles in priority order:
        1. Custom theme (if provided)
        2. Predefined theme (if theme_name is set)
        3. Explicit style parameters
        4. Built-in defaults

        Args:
            element: Style element name (e.g., 'user', 'assistant', 'error').

        Returns:
            Rich style string for the element.
        """
        # Check custom theme first
        if self.custom_theme and element in self.custom_theme:
            return self.custom_theme[element]

        # Check predefined theme
        theme = self.get_predefined_theme()
        if theme and element in theme:
            return theme[element]

        # Fall back to explicit parameters based on element name
        style_map = {
            "user": self.user_style,
            "user_title": f"bold {self.user_style}",
            "assistant": self.assistant_style,
            "assistant_title": f"bold {self.assistant_style}",
            "assistant_text": self.assistant_text_style,
            "error": self.error_style,
            "error_title": f"bold {self.error_style}",
            "info": self.info_style,
            "info_title": f"bold {self.info_style}",
            "warning": "yellow",
            "warning_title": "bold yellow",
            "success": self.info_style,
            # Derived styles for compatibility
            "status": self.info_style,
            "welcome": "magenta",
            "monitor": self.info_style,
            # Thinking/reasoning styles
            "analysis": "dim cyan",
            "analysis_title": "bold yellow",
            "response_title": f"bold {self.info_style}",
        }

        return style_map.get(element, "white")

    def get_predefined_theme(self) -> Optional[dict[str, str]]:
        """Get a predefined theme by name.

        Returns:
            Dictionary mapping element names to Rich style strings, or None.
        """
        if self.theme_name is None:
            return None

        themes = {
            "dark": {
                "user": "bold bright_blue",
                "assistant": "bright_cyan",
                "assistant_text": "bright_white",
                "error": "bright_red",
                "info": "bright_green",
            },
            "light": {
                "user": "bold blue",
                "assistant": "dark_cyan",
                "assistant_text": "black",
                "error": "red",
                "info": "green",
            },
            "monokai": {
                "user": "bold #66d9ef",
                "assistant": "#a6e22e",
                "assistant_text": "#f8f8f2",
                "error": "#f92672",
                "info": "#a6e22e",
            },
            "minimal": {
                "user": "bold",
                "assistant": "white",
                "assistant_text": "white",
                "error": "red",
                "info": "green",
            },
            "neon": {
                "user": "bold #00ffff",
                "assistant": "#ff00ff",
                "assistant_text": "#ffffff",
                "error": "#ff0066",
                "info": "#00ff00",
            },
        }

        return themes.get(self.theme_name)

    def __post_init__(self):
        """Validate style parameters."""
        # Validate panel box style
        valid_box_styles = {"rounded", "square", "double", "heavy", "minimal", "ascii"}
        if self.panel_box_style not in valid_box_styles:
            raise ValueError(
                f"Invalid panel_box_style '{self.panel_box_style}'. "
                f"Must be one of: {', '.join(valid_box_styles)}"
            )

        # Validate code theme
        valid_code_themes = {
            "monokai",
            "github-dark",
            "dracula",
            "nord",
            "solarized-dark",
            "solarized-light",
            "material",
            "native",
        }
        if self.code_theme not in valid_code_themes:
            raise ValueError(
                f"Invalid code_theme '{self.code_theme}'. "
                f"Must be one of: {', '.join(valid_code_themes)}"
            )

        # Validate theme name
        if self.theme_name:
            valid_themes = {"dark", "light", "monokai", "minimal", "neon"}
            if self.theme_name not in valid_themes:
                raise ValueError(
                    f"Invalid theme_name '{self.theme_name}'. "
                    f"Must be one of: {', '.join(valid_themes)}"
                )

        # Validate padding tuple
        if len(self.assistant_padding) != 2:
            raise ValueError(
                "assistant_padding must be a tuple of (vertical, horizontal)"
            )
