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

"""Textual TUI for viewing JSONL conversation files."""

import sys
import tempfile
from pathlib import Path
from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils


def view(
    file_path: Annotated[
        str,
        typer.Argument(
            help="Path to the JSONL file containing conversations. Use '-' to read from stdin."
        ),
    ] = "-",
    start_index: Annotated[
        int,
        typer.Option(
            "--start", "-s", help="Start viewing from this conversation index."
        ),
    ] = 0,
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Browse conversations in a JSONL file using an interactive TUI.

    The viewer displays conversations in a chat-like format with:
    - Navigation between conversations using arrow keys or j/k
    - Scrollable message content
    - Color-coded roles (system, user, assistant, tool)
    - Search functionality with /
    - Metadata display for each conversation

    Examples:
        oumi view data/conversations.jsonl
        oumi view data/conversations.jsonl --start 10
        cat data/conversations.jsonl | oumi view
        oumi view -  # Read from stdin
    """
    try:
        from oumi.cli.view_app import ConversationViewerApp
    except ImportError as e:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] The 'textual' package is required for the view command."
        )
        cli_utils.CONSOLE.print("Install it with: [cyan]pip install textual[/cyan]")
        raise typer.Exit(code=1) from e

    from_stdin = False
    actual_path = file_path

    # Handle stdin input
    if file_path == "-" or (file_path == "-" and not sys.stdin.isatty()):
        # Check if there's actually data on stdin
        if sys.stdin.isatty():
            cli_utils.CONSOLE.print(
                "[red]Error:[/red] No input provided. "
                "Either specify a file path or pipe data to stdin."
            )
            cli_utils.CONSOLE.print(
                "\nUsage:\n"
                "  oumi view data/conversations.jsonl\n"
                "  cat data/conversations.jsonl | oumi view\n"
                "  oumi view -  # with piped input"
            )
            raise typer.Exit(code=1)

        # Read all stdin content to a temp file (textual needs a file path)
        from_stdin = True
        stdin_content = sys.stdin.read()

        if not stdin_content.strip():
            cli_utils.CONSOLE.print("[red]Error:[/red] No data received from stdin.")
            raise typer.Exit(code=1)

        # Write to a temp file that persists for the app duration
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        )
        temp_file.write(stdin_content)
        temp_file.close()
        actual_path = temp_file.name
    else:
        path = Path(file_path)
        if not path.exists():
            cli_utils.CONSOLE.print(f"[red]Error:[/red] File not found: {file_path}")
            raise typer.Exit(code=1)

        if path.suffix.lower() not in (".jsonl", ".json"):
            cli_utils.CONSOLE.print(
                f"[yellow]Warning:[/yellow] File does not have .jsonl extension: {file_path}"
            )

    app = ConversationViewerApp(
        file_path=actual_path, start_index=start_index, from_stdin=from_stdin
    )
    app.run()

    # Clean up temp file if we created one
    if from_stdin:
        try:
            Path(actual_path).unlink()
        except Exception:
            pass
