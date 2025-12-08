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

"""Interactive TUI viewer for JSONL conversations, YAML configs, and training outputs."""

import os
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
            help=(
                "Path to file or folder to view. Supports JSONL conversations, "
                "YAML configs, and training output folders. "
                "Use '-' to read from stdin (JSONL only)."
            )
        ),
    ] = "-",
    start_index: Annotated[
        int,
        typer.Option(
            "--start", "-s", help="Start viewing from this conversation index (JSONL only)."
        ),
    ] = 0,
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Browse JSONL conversations, YAML configs, or training outputs with an interactive TUI.

    For JSONL files, displays conversations in an interactive TUI with:
    - Navigation between conversations using arrow keys or j/k
    - Scrollable message content
    - Color-coded roles (system, user, assistant, tool)
    - Search functionality with /
    - Metadata display for each conversation

    Supported JSONL formats:
    - Oumi format: {"messages": [{"role": "...", "content": "..."}]}
    - Alpaca format: {"instruction": "...", "input": "...", "output": "..."}
    - DPO format: {"prompt": "...", "chosen": [...], "rejected": [...]}
    - Pre-training format: {"text": "..."}

    For YAML config files, displays an interactive browser with:
    - Config type detection (TrainingConfig, EvaluationConfig, etc.)
    - Hierarchical tree navigation of all settings
    - Field docstrings from Oumi config classes
    - Raw YAML toggle with 'r' key

    For training output folders (containing trainer_state.json), displays a dashboard with:
    - Training summary with status and key metrics
    - Timeline of training events
    - Metrics charts (loss, learning rate, accuracy)
    - Searchable training logs
    - Configuration tree view
    - Checkpoint and output file listing

    Examples:
        oumi view data/conversations.jsonl
        oumi view data/alpaca_format.jsonl  # Alpaca format also supported
        oumi view data/conversations.jsonl --start 10
        oumi view configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml
        oumi view output/my_training_run/  # Training folder
        cat data/conversations.jsonl | oumi view
        oumi view -  # Read from stdin (JSONL)
    """
    # Determine file type and route to appropriate viewer
    is_stdin = file_path == "-"

    if is_stdin:
        # Stdin always goes to conversation viewer
        _view_conversations(file_path, start_index)
        return

    path = Path(file_path)
    if not path.exists():
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Path not found: {file_path}")
        raise typer.Exit(code=1)

    # Check if it's a training output folder
    if path.is_dir():
        if _is_training_folder(path):
            _view_training(path)
        else:
            cli_utils.CONSOLE.print(
                f"[red]Error:[/red] Directory '{file_path}' is not a training output folder. "
                "Training folders must contain a trainer_state.json file."
            )
            raise typer.Exit(code=1)
        return

    # Route based on file extension
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        _view_config(path)
    elif suffix in (".jsonl", ".json"):
        _view_conversations(file_path, start_index)
    else:
        # Try to detect from content
        try:
            with open(path) as f:
                first_line = f.readline().strip()
            if first_line.startswith("{"):
                _view_conversations(file_path, start_index)
            else:
                # Default to config viewer for unknown files
                cli_utils.CONSOLE.print(
                    f"[yellow]Warning:[/yellow] Unknown file type '{suffix}', "
                    "attempting to view as config."
                )
                _view_config(path)
        except Exception:
            cli_utils.CONSOLE.print(
                f"[red]Error:[/red] Could not determine file type for: {file_path}"
            )
            raise typer.Exit(code=1)


def _is_training_folder(path: Path) -> bool:
    """Check if the path is a training output folder.

    A training output folder is identified by the presence of trainer_state.json.
    """
    return (path / "trainer_state.json").exists()


def _view_training(folder_path: Path) -> None:
    """View a training output folder."""
    try:
        from oumi.cli.view_training_app import TrainingViewerApp
    except ImportError as e:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] The 'textual' package is required for viewing training outputs."
        )
        cli_utils.CONSOLE.print("Install it with: [cyan]pip install textual[/cyan]")
        raise typer.Exit(code=1) from e

    app = TrainingViewerApp(folder_path=folder_path)
    app.run()


def _view_config(file_path: Path) -> None:
    """View a YAML config file."""
    from oumi.cli.view_config import view_config

    view_config(file_path=file_path)


def _view_conversations(file_path: str, start_index: int) -> None:
    """View a JSONL conversations file."""
    try:
        from oumi.cli.view_app import ConversationViewerApp
    except ImportError as e:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] The 'textual' package is required for viewing conversations."
        )
        cli_utils.CONSOLE.print("Install it with: [cyan]pip install textual[/cyan]")
        raise typer.Exit(code=1) from e

    from_stdin = False
    actual_path = file_path

    # Handle stdin input
    if file_path == "-":
        # Check if there's actually data on stdin
        if sys.stdin.isatty():
            cli_utils.CONSOLE.print(
                "[red]Error:[/red] No input provided. "
                "Either specify a file path or pipe data to stdin."
            )
            cli_utils.CONSOLE.print(
                "\nUsage:\n"
                "  oumi view data/conversations.jsonl\n"
                "  oumi view config.yaml\n"
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

        # Reconnect stdin to the terminal so Textual can receive keyboard input
        # This is necessary because stdin was consumed by the piped data
        try:
            tty_fd = os.open("/dev/tty", os.O_RDONLY)
            os.dup2(tty_fd, 0)
            os.close(tty_fd)
            sys.stdin = open(0, "r", closefd=False)
        except OSError:
            cli_utils.CONSOLE.print(
                "[yellow]Warning:[/yellow] Cannot reconnect to terminal. "
                "Interactive controls may not work."
            )
    else:
        path = Path(file_path)
        if path.suffix.lower() not in (".jsonl", ".json"):
            cli_utils.CONSOLE.print(
                f"[yellow]Warning:[/yellow] File does not have .jsonl extension: {file_path}"
            )

    app = ConversationViewerApp(
        file_path=actual_path,
        start_index=start_index,
        from_stdin=from_stdin,
    )
    app.run()

    # Clean up temp file if we created one
    if from_stdin:
        try:
            Path(actual_path).unlink()
        except Exception:
            pass
