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

import os
import sys
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.text import Text
from rich.theme import Theme

# Default Oumi theme colors
DEFAULT_THEME = Theme(
    {
        "info": "bright_blue",
        "warning": "bright_yellow",
        "error": "bright_red",
        "success": "bright_green",
        "primary": "bright_magenta",
        "secondary": "cyan",
        "accent": "purple",
        "normal": "white",
        "muted": "dim white",
        "header": "bold bright_blue",
        "subheader": "bold bright_cyan",
    }
)

# Environment variable to control styling
OUMI_NO_STYLE = "OUMI_NO_STYLE"
OUMI_STYLE_LEVEL = "OUMI_STYLE_LEVEL"


class StyleLevel(str, Enum):
    """Control the level of styling in the CLI."""

    NONE = "none"  # No styling at all
    FULL = "full"  # Full styling with animations and colors


@lru_cache(maxsize=1)
def get_style_level() -> StyleLevel:
    """Get the current style level from environment or defaults.

    Returns:
        StyleLevel: The determined style level
    """
    # If OUMI_NO_STYLE is set, return NONE regardless of OUMI_STYLE_LEVEL
    if os.environ.get(OUMI_NO_STYLE, "").lower() in ("1", "true", "yes"):
        return StyleLevel.NONE

    # Check OUMI_STYLE_LEVEL environment variable
    level = os.environ.get(OUMI_STYLE_LEVEL, "").lower()
    if level == StyleLevel.NONE.value:
        return StyleLevel.NONE

    # Default to NONE for non-interactive terminals
    if not sys.stdout.isatty():
        return StyleLevel.NONE

    # Default to FULL in all other cases
    return StyleLevel.FULL


def create_styled_console(force_terminal: bool = False) -> Console:
    """Create a console with the appropriate styling based on environment.

    Args:
        force_terminal: Force the console to use terminal styling, even if
                        not connected to a terminal.

    Returns:
        Console: A properly configured Rich console
    """
    style_level = get_style_level()

    if style_level == StyleLevel.NONE:
        # Plain console with no colors or styling
        return Console(highlight=False, color_system=None, theme=None)

    # For FULL, enable colors and styling
    return Console(
        highlight=True,
        color_system="auto",
        theme=DEFAULT_THEME,
        force_terminal=force_terminal,
    )


def styled_header(
    title: str,
    console: Optional[Console] = None,
    style: str = "header",
) -> None:
    """Print a styled section header with the given title.

    Args:
        title: The title text to display in the header
        console: The Console object to use for printing
        style: The style name to use for the divider
    """
    if console is None:
        console = create_styled_console()

    # Simple plain text header or rich styled header based on style level
    if get_style_level() == StyleLevel.NONE:
        console.print(f"\n{title}\n")
    else:
        console.print(f"\n[{style}]{'━' * console.width}[/{style}]")
        console.print(f"[primary]   {title}[/primary]")
        console.print(f"[{style}]{'━' * console.width}[/{style}]\n")


def styled_panel(
    content: Union[str, Text],
    title: Optional[str] = None,
    console: Optional[Console] = None,
    style: str = "primary",
) -> None:
    """Print content in a styled panel.

    Args:
        content: The content to display in the panel
        title: Optional title for the panel
        console: The Console object to use for printing
        style: The style name to use for the panel border
    """
    if console is None:
        console = create_styled_console()

    # Choose between plain text and styled panel based on style level
    if get_style_level() == StyleLevel.NONE:
        if title:
            console.print(f"{title}:")
        console.print(content)
    else:
        panel = Panel(
            content,
            title=title,
            border_style=style,
            expand=False,
        )
        console.print(panel)


def with_spinner(
    message: str,
    console: Optional[Console] = None,
) -> Callable:
    """Create a context manager that shows a spinner during an operation.

    Args:
        message: The message to display alongside the spinner
        console: The Console object to use for the spinner

    Returns:
        A context manager that shows a spinner
    """
    if console is None:
        console = create_styled_console()

    # Simple context manager for plain text mode
    class TextSpinner:
        def __enter__(self):
            console.print(f"{message}...")
            return self

        def __exit__(self, *args: Any):
            pass

    # Fancy spinner for full styling mode
    class RichSpinner:
        def __enter__(self):
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[primary]{task.description}[/primary]"),
                console=console,
                transient=True,
            )
            self.task_id = self.progress.add_task(message)
            self.progress.start()
            return self

        def __exit__(self, *args: Any):
            self.progress.stop()

    return TextSpinner() if get_style_level() == StyleLevel.NONE else RichSpinner()


def print_result(
    message: str,
    success: bool = True,
    console: Optional[Console] = None,
) -> None:
    """Print a result message with appropriate styling.

    Args:
        message: The message to display
        success: Whether the operation was successful
        console: The Console object to use for printing
    """
    if console is None:
        console = create_styled_console()

    # Plain text for NONE style, rich formatting for FULL
    if get_style_level() == StyleLevel.NONE:
        console.print(message)
    else:
        style = "success" if success else "error"
        prefix = "✓ " if success else "✗ "
        console.print(f"[{style}]{prefix}{message}[/{style}]")
