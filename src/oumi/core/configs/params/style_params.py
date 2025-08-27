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
    """Parameters for customizing Rich console styling in interactive inference.

    Rich supports a wide variety of text styling options:
    - Colors: 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'black'
    - Bright variants: 'bright_red', 'bright_green', etc.
    - RGB colors: 'rgb(175,0,255)' or '#af00ff'
    - Background colors: 'on white', 'on black', etc.
    - Text attributes: 'bold', 'italic', 'underline', 'strike', 'dim', 'reverse'
    - Combinations: 'bold red underline on white'

    Examples:
        # Minimal styling
        style:
          no_color: true

        # Dark theme with green accent
        style:
          user_prompt_style: "bold green"
          assistant_title_style: "bold bright_green"
          assistant_border_style: "green"

        # Custom theme
        style:
          custom_theme:
            info: "dim cyan"
            warning: "yellow"
            error: "bold red"
            success: "bold green"
    """

    # Theme configuration
    theme_name: Optional[str] = None
    """Name of the predefined theme to use.

    Available themes:
    - 'dark': Dark background optimized theme
    - 'light': Light background optimized theme
    - 'monokai': Monokai-inspired colorful theme
    - 'minimal': Minimal styling with few colors
    - 'neon': Bright neon colors on dark background
    """

    custom_theme: Optional[dict[str, str]] = None
    """Custom theme mapping style names to Rich style strings.

    This allows defining a complete custom color scheme that can be
    referenced throughout the application.
    """

    # Welcome and UI styles
    welcome_style: str = "bold magenta"
    """Style for welcome message and title."""

    welcome_border_style: str = "magenta"
    """Style for welcome message border."""

    status_style: str = "bold green"
    """Style for status messages (e.g., 'Thinking...')."""

    status_title_style: Optional[str] = None
    """Style for status panel titles (defaults to status_style)."""

    status_border_style: Optional[str] = None
    """Style for status panel borders (defaults to status_style without bold)."""

    # System monitor color scheme
    monitor_low_usage_color: str = "green"
    """Color for low usage percentages (0-49%)."""

    monitor_medium_usage_color: str = "cyan"
    """Color for medium usage percentages (50-69%)."""

    monitor_high_usage_color: str = "yellow"
    """Color for high usage percentages (70-89%)."""

    monitor_critical_usage_color: str = "red"
    """Color for critical usage percentages (90-100%)."""

    monitor_label_style: str = "bold cyan"
    """Style for system monitor labels (CPU:, RAM:, etc.)."""

    monitor_value_style: str = "white"
    """Style for system monitor values when not colored by usage."""

    # User interaction styles
    user_prompt_style: str = "bold blue"
    """Style for user input prompts."""

    user_input_style: Optional[str] = None
    """Style for user input text (if different from prompt)."""

    # Assistant response styles
    assistant_title_style: str = "bold cyan"
    """Style for assistant response titles."""

    assistant_border_style: str = "cyan"
    """Style for assistant response borders."""

    assistant_text_style: str = "white"
    """Style for assistant response text."""

    assistant_padding: tuple[int, int] = (1, 2)
    """Padding (vertical, horizontal) for assistant response panels."""

    # Error styles
    error_style: str = "red"
    """Style for error message text."""

    error_title_style: str = "bold red"
    """Style for error message titles."""

    error_border_style: str = "red"
    """Style for error message borders."""

    # GPT-OSS reasoning tag styles
    analysis_text_style: str = "dim cyan"
    """Style for GPT-OSS analysis/reasoning sections."""

    analysis_title_style: str = "bold yellow"
    """Style for GPT-OSS analysis section titles."""

    analysis_border_style: str = "yellow"
    """Style for GPT-OSS analysis borders."""

    response_text_style: str = "bright_white"
    """Style for GPT-OSS final response sections."""

    response_title_style: str = "bold green"
    """Style for GPT-OSS response section titles."""

    response_border_style: str = "green"
    """Style for GPT-OSS response borders."""

    # Special content styles
    code_theme: str = "monokai"
    """Syntax highlighting theme for code blocks.

    Available themes: 'monokai', 'github-dark', 'dracula', 'nord',
    'solarized-dark', 'solarized-light', 'material', 'native'
    """

    markdown_code_theme: Optional[str] = None
    """Override theme for markdown code blocks (defaults to code_theme)."""

    # Console configuration
    force_terminal: Optional[bool] = None
    """Force terminal mode even if not in a terminal."""

    force_jupyter: Optional[bool] = None
    """Force Jupyter notebook mode."""

    no_color: bool = False
    """Disable all colors and styling."""

    width: Optional[int] = None
    """Console width override (default: auto-detect)."""

    height: Optional[int] = None
    """Console height override (default: auto-detect)."""

    legacy_windows: bool = False
    """Use legacy Windows mode for older terminals."""

    # Emoji configuration
    use_emoji: bool = True
    """Whether to use emoji in output (🤖, 🧠, 💬, etc.)."""

    emoji_variant: Optional[str] = None
    """Emoji variant ('emoji' or 'text')."""

    # Panel configuration
    expand_panels: bool = False
    """Whether panels should expand to full terminal width."""

    panel_box_style: str = "rounded"
    """Box drawing style for panels.

    Options: 'rounded', 'square', 'double', 'heavy', 'minimal', 'ascii'
    """

    def get_predefined_theme(self) -> Optional[dict[str, str]]:
        """Get a predefined theme by name.

        Returns:
            Dictionary mapping style names to Rich style strings, or None.
        """
        themes = {
            "dark": {
                "user_prompt": "bold bright_blue",
                "assistant_title": "bold bright_cyan",
                "assistant_text": "bright_white",
                "error": "bright_red",
                "warning": "bright_yellow",
                "success": "bright_green",
                "info": "bright_blue",
                "monitor_low": "bright_green",
                "monitor_medium": "bright_cyan",
                "monitor_high": "bright_yellow",
                "monitor_critical": "bright_red",
                "monitor_labels": "bold bright_cyan",
                "monitor_values": "bright_white",
            },
            "light": {
                "user_prompt": "bold blue",
                "assistant_title": "bold dark_cyan",
                "assistant_text": "black",
                "error": "red",
                "warning": "dark_orange",
                "success": "green",
                "info": "blue",
                "monitor_low": "green",
                "monitor_medium": "blue",
                "monitor_high": "dark_orange",
                "monitor_critical": "red",
                "monitor_labels": "bold blue",
                "monitor_values": "black",
            },
            "monokai": {
                "user_prompt": "bold #66d9ef",  # Monokai blue
                "assistant_title": "bold #a6e22e",  # Monokai green
                "assistant_text": "#f8f8f2",  # Monokai foreground
                "error": "#f92672",  # Monokai red
                "warning": "#fd971f",  # Monokai orange
                "success": "#a6e22e",  # Monokai green
                "info": "#ae81ff",  # Monokai purple
                "monitor_low": "#a6e22e",  # Monokai green
                "monitor_medium": "#66d9ef",  # Monokai blue
                "monitor_high": "#fd971f",  # Monokai orange
                "monitor_critical": "#f92672",  # Monokai red
                "monitor_labels": "bold #ae81ff",  # Monokai purple
                "monitor_values": "#f8f8f2",  # Monokai foreground
            },
            "minimal": {
                "user_prompt": "bold",
                "assistant_title": "bold",
                "assistant_text": "white",
                "error": "red",
                "warning": "yellow",
                "success": "green",
                "info": "blue",
                "monitor_low": "green",
                "monitor_medium": "blue",
                "monitor_high": "yellow",
                "monitor_critical": "red",
                "monitor_labels": "bold",
                "monitor_values": "white",
            },
            "neon": {
                "user_prompt": "bold #00ffff",  # Cyan neon
                "assistant_title": "bold #ff00ff",  # Magenta neon
                "assistant_text": "#ffffff",
                "error": "#ff0066",  # Hot pink
                "warning": "#ffff00",  # Yellow neon
                "success": "#00ff00",  # Green neon
                "info": "#0099ff",  # Blue neon
                "monitor_low": "#00ff00",  # Green neon
                "monitor_medium": "#00ffff",  # Cyan neon
                "monitor_high": "#ffff00",  # Yellow neon
                "monitor_critical": "#ff0066",  # Hot pink
                "monitor_labels": "bold #ff00ff",  # Magenta neon
                "monitor_values": "#ffffff",
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

        # Validate code themes
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

        if (
            self.markdown_code_theme
            and self.markdown_code_theme not in valid_code_themes
        ):
            raise ValueError(
                f"Invalid markdown_code_theme '{self.markdown_code_theme}'. "
                f"Must be one of: {', '.join(valid_code_themes)}"
            )

        # Validate predefined theme name
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

        # Validate emoji variant
        if self.emoji_variant and self.emoji_variant not in {"emoji", "text"}:
            raise ValueError("emoji_variant must be 'emoji' or 'text'")
