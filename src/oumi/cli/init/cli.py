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

"""CLI command for oumi init."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax

from oumi.cli.init.generator import InitGenerator
from oumi.cli.init.schemas import OutputFormat

console = Console()

app = typer.Typer(
    help="Initialize oumi configs from task description",
    pretty_exceptions_enable=False,
)


@app.callback(invoke_without_command=True)
def init_callback(
    ctx: typer.Context,
    task: Annotated[
        str | None,
        typer.Option(
            "--task",
            "-t",
            help="Natural language description of your data generation task",
        ),
    ] = None,
    task_file: Annotated[
        str | None,
        typer.Option(
            "--task-file",
            "-T",
            help="Path to file containing task description",
        ),
    ] = None,
    source: Annotated[
        list[str] | None,
        typer.Option(
            "--source",
            "-s",
            help="Source file(s) - documents or datasets",
        ),
    ] = None,
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to save generated configs",
        ),
    ] = "./configs/",
    output_format: Annotated[
        str,
        typer.Option(
            "--output-format",
            "-f",
            help="Output format: conversation, instruction, or raw",
        ),
    ] = "conversation",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview configs without saving",
        ),
    ] = False,
    new_session: Annotated[
        bool,
        typer.Option(
            "--new",
            "-n",
            help="Start a new session (ignore existing session)",
        ),
    ] = False,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            "-N",
            help="Run without user interaction (for automation/evaluation)",
        ),
    ] = False,
):
    r"""Generate oumi synth and judge configs from a task description.

    Automatically resumes from existing session if one exists in the output directory.

    Examples:
        oumi init --task "Generate QA pairs about world history"

        oumi init --task-file task.txt  # Load task from file

        oumi init --task "Create variations of these support tickets" \\
            --source tickets.jsonl

        oumi init --task "Generate medical QA from guidelines" \\
            --source guidelines.pdf --output-dir ./medical/

        oumi init --new --task "Start fresh"  # Ignore existing session

        oumi init --non-interactive --task "Generate QA pairs" \\
            --output-dir ./output/  # No user interaction (for automation)
    """
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            Panel(
                "[red]ANTHROPIC_API_KEY environment variable not set.[/red]\n\n"
                "Set it with:\n"
                "  export ANTHROPIC_API_KEY='your-key'\n\n"
                "Get a key at: https://console.anthropic.com/",
                title="[bold red]Error[/bold red]",
            )
        )
        raise typer.Exit(1)

    # Validate task input: either --task or --task-file, not both
    if task is not None and task_file is not None:
        console.print("[red]Error: Cannot specify both --task and --task-file[/red]")
        raise typer.Exit(1)

    # Load task from file if --task-file is provided
    if task_file is not None:
        task_path = Path(task_file)
        if not task_path.exists():
            console.print(f"[red]Task file not found: {task_file}[/red]")
            raise typer.Exit(1)
        try:
            task = task_path.read_text().strip()
            if not task:
                console.print(f"[red]Task file is empty: {task_file}[/red]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Failed to read task file: {e}[/red]")
            raise typer.Exit(1)

    generator = InitGenerator()

    # Check for existing session (auto-resume unless --new or --non-interactive)
    existing_session = generator.load_session(output_dir)
    if existing_session and not new_session and not non_interactive:
        console.print(
            Panel(
                f"[bold]Found existing session[/bold]\n\n"
                f"Task: {existing_session.task}\n"
                f"Phase: {existing_session.phase}\n"
                f"Sources: {len(existing_session.sources)} file(s)\n\n"
                "[dim]Use --new to start a fresh session[/dim]",
                title="[cyan]Resuming Session[/cyan]",
            )
        )
        try:
            synth_yaml, judge_yaml = generator.resume(output_dir)
            state = generator.load_session(output_dir)
            understanding = state.understanding if state else {}
        except Exception as e:
            console.print(f"[red]Resume failed: {e}[/red]")
            raise typer.Exit(1)
    else:
        # Start new session
        if new_session:
            generator.clear_session(output_dir)

        # If no task provided, show help
        if task is None:
            console.print(ctx.get_help())
            raise typer.Exit(0)

        source = source or []

        # Validate sources exist
        for s in source:
            if not Path(s).exists():
                console.print(f"[red]Source file not found: {s}[/red]")
                raise typer.Exit(1)

        # Parse output format
        try:
            fmt = OutputFormat(output_format)
        except ValueError:
            console.print(f"[red]Invalid output format: {output_format}[/red]")
            console.print("Valid options: conversation, instruction, raw")
            raise typer.Exit(1)

        # Display header
        console.print(
            Panel(
                f"[bold]Task:[/bold] {task}\n"
                f"[bold]Sources:[/bold] {len(source)} file(s)\n"
                f"[bold]Output format:[/bold] {output_format}",
                title="[bold cyan]oumi init[/bold cyan]",
            )
        )

        # Run generator
        try:
            synth_yaml, judge_yaml = generator.run(
                task=task,
                sources=source,
                output_format=fmt,
                output_dir=output_dir,
                non_interactive=non_interactive,
            )
            # Get understanding from session for edit loop
            state = generator.load_session(output_dir)
            understanding = state.understanding if state else {}
        except Exception as e:
            console.print(f"[red]Generation failed: {e}[/red]")
            raise typer.Exit(1)

    # Show preview (unless non-interactive with quiet output preferred)
    if not non_interactive:
        console.print("\n")
        console.print(
            Panel(
                Syntax(synth_yaml, "yaml", theme="monokai"),
                title="[bold]synth_config.yaml[/bold]",
            )
        )
        console.print(
            Panel(
                Syntax(judge_yaml, "yaml", theme="monokai"),
                title="[bold]judge_config.yaml[/bold]",
            )
        )

    # Confirm save with edit option
    if dry_run:
        console.print("[yellow]Dry run - configs not saved[/yellow]")
        # In non-interactive mode, output the YAML to stdout for capture
        if non_interactive:
            print("---SYNTH_CONFIG---")
            print(synth_yaml)
            print("---JUDGE_CONFIG---")
            print(judge_yaml)
        return

    # Non-interactive mode: skip the interactive loop and auto-save
    if non_interactive:
        pass  # Fall through to save
    else:
        # Interactive loop: Save / Edit / Preview / Cancel
        while True:
            choice = Prompt.ask(
                "\n[bold]What would you like to do?[/bold]",
                choices=["save", "edit", "preview", "cancel"],
                default="save",
            )

            if choice == "cancel":
                console.print("[yellow]Cancelled[/yellow]")
                return

            if choice == "preview":
                generator._run_preview(synth_yaml, judge_yaml)
                continue

            if choice == "edit":
                synth_yaml, judge_yaml = generator.edit_configs(
                    synth_yaml, judge_yaml, understanding
                )
                # Update session with edited configs
                state = generator.load_session(output_dir)
                if state:
                    state.synth_yaml = synth_yaml
                    state.judge_yaml = judge_yaml
                    generator.save_session(state)
                continue

            if choice == "save":
                break

    # Save configs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filenames from task
    state = generator.load_session(output_dir)
    task_for_filename = state.task if state else (task or "config")
    base_name = _task_to_filename(task_for_filename)

    synth_path = output_path / f"{base_name}_synth.yaml"
    judge_path = output_path / f"{base_name}_judge.yaml"

    synth_path.write_text(synth_yaml)
    judge_path.write_text(judge_yaml)

    console.print(f"\n[green]Saved: {synth_path}[/green]")
    console.print(f"[green]Saved: {judge_path}[/green]")

    # Show next steps
    console.print(
        Panel(
            f"[bold]Next steps:[/bold]\n\n"
            f"1. Review the generated configs\n"
            f"2. Generate synthetic data:\n"
            f"   [cyan]oumi synth -c {synth_path}[/cyan]\n"
            f"3. Evaluate quality:\n"
            f"   [cyan]oumi judge dataset -c {judge_path} "
            f"--input {base_name}_dataset.jsonl[/cyan]",
            title="[bold green]Done![/bold green]",
        )
    )


def _task_to_filename(task: str) -> str:
    """Convert task description to valid filename."""
    # Take first few words, lowercase, replace spaces with underscores
    words = task.lower().split()[:4]
    name = "_".join(words)
    # Remove non-alphanumeric
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name or "config"
