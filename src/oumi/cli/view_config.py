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

"""Interactive TUI for browsing Oumi configuration files."""

from pathlib import Path

import typer

import oumi.cli.cli_utils as cli_utils


def view_config(file_path: Path) -> None:
    """Display an Oumi config file with an interactive TUI browser.

    Args:
        file_path: Path to the YAML config file.
    """
    try:
        from oumi.cli.view_config_app import ConfigViewerApp
    except ImportError as e:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] The 'textual' package is required for viewing configs."
        )
        cli_utils.CONSOLE.print("Install it with: [cyan]pip install textual[/cyan]")
        raise typer.Exit(code=1) from e

    app = ConfigViewerApp(file_path=file_path)
    app.run()
