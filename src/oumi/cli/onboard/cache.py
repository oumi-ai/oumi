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

"""Caching utilities for the onboard wizard."""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Optional

import yaml
from rich.panel import Panel
from rich.prompt import Prompt

import oumi.cli.cli_utils as cli_utils

from .dataclasses import INPUT_FORMATS, WIZARD_CACHE_FILE, WizardState


def compute_file_hash(file_path: Path) -> str:
    """Compute a hash of the file content for cache invalidation.

    Args:
        file_path: Path to the file.

    Returns:
        SHA256 hash of the file content as hex string.
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


def get_cache_path(output_dir: Path) -> Path:
    """Get the path to the wizard cache file."""
    return output_dir / WIZARD_CACHE_FILE


def save_wizard_cache(state: WizardState, output_dir: Path, step_name: str) -> None:
    """Save wizard state to cache file after completing a step.

    Args:
        state: Current wizard state.
        output_dir: Output directory for cache file.
        step_name: Name of the step just completed.
    """
    if step_name not in state.completed_steps:
        state.completed_steps.append(step_name)

    cache_path = get_cache_path(output_dir)
    cache_data = state.to_dict()

    cache_data["_metadata"] = {
        "cache_version": "1.0",
        "description": "Oumi wizard state cache. You can edit this file to modify wizard settings.",
    }

    with open(cache_path, "w") as f:
        yaml.dump(cache_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    cli_utils.CONSOLE.print(f"[dim]Cache saved: {cache_path}[/dim]")


def load_wizard_cache(output_dir: Path) -> Optional[WizardState]:
    """Load wizard state from cache file if it exists.

    Args:
        output_dir: Output directory containing cache file.

    Returns:
        WizardState if cache exists and is valid, None otherwise.
    """
    cache_path = get_cache_path(output_dir)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path) as f:
            cache_data = yaml.safe_load(f)

        if not cache_data or not isinstance(cache_data, dict):
            return None

        cache_data.pop("_metadata", None)
        return WizardState.from_dict(cache_data)
    except Exception as e:
        cli_utils.CONSOLE.print(f"[yellow]Warning: Could not load cache: {e}[/yellow]")
        return None


def display_cache_summary(state: WizardState, cache_path: Path) -> None:
    """Display a summary of the cached wizard state.

    Args:
        state: Loaded wizard state.
        cache_path: Path to the cache file.
    """
    completed = state.completed_steps

    step_status = []
    all_steps = [
        ("task", "Task"),
        ("inputs", "Inputs"),
        ("outputs", "Outputs"),
        ("generate", "Generate"),
    ]

    for step_id, step_name in all_steps:
        if step_id in completed:
            step_status.append(f"  [green]v[/green] {step_name}")
        else:
            step_status.append(f"  [dim]o[/dim] {step_name}")

    task_preview = state.task.description[:50] + "..." if state.task.description else "[not set]"
    prompt_preview = state.task.system_prompt[:50] + "..." if state.task.system_prompt else "[not set]"
    input_preview = INPUT_FORMATS.get(state.inputs.format, state.inputs.format) if state.inputs.format else "[not set]"
    criteria_preview = ", ".join(state.outputs.criteria[:3]) if state.outputs.criteria else "[not set]"

    summary_parts = [
        f"[bold]Task:[/bold] {task_preview}",
        f"[bold]System Prompt:[/bold] {prompt_preview}",
        f"[bold]Input Format:[/bold] {input_preview}",
        f"[bold]Quality Criteria:[/bold] {criteria_preview}",
        "",
        "[bold]Steps Completed:[/bold]",
        *step_status,
    ]

    cli_utils.CONSOLE.print(
        Panel(
            "\n".join(summary_parts),
            title=f"[cyan]Cached State: {cache_path}[/cyan]",
            border_style="cyan",
        )
    )


def prompt_cache_action(cache_path: Path) -> str:
    """Prompt user for what to do with existing cache.

    Args:
        cache_path: Path to the cache file.

    Returns:
        One of: "resume", "edit", "restart"
    """
    cli_utils.CONSOLE.print(
        "\n[bold]Options:[/bold]\n"
        "  [cyan][1][/cyan] Resume from where you left off\n"
        "  [cyan][2][/cyan] Edit the cache file first (opens in editor)\n"
        "  [cyan][3][/cyan] Start fresh (delete cache)\n"
    )

    choice = Prompt.ask("Your choice", choices=["1", "2", "3"], default="1")

    if choice == "1":
        return "resume"
    elif choice == "2":
        return "edit"
    else:
        return "restart"


def open_cache_for_editing(cache_path: Path) -> None:
    """Open the cache file in the user's default editor.

    Args:
        cache_path: Path to the cache file.
    """
    editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "nano"))

    cli_utils.CONSOLE.print(f"\n[dim]Opening {cache_path} in {editor}...[/dim]")
    cli_utils.CONSOLE.print("[dim]Save and close the editor when done.[/dim]\n")

    try:
        subprocess.run([editor, str(cache_path)], check=True)
        cli_utils.CONSOLE.print("[green]Cache file updated.[/green]")
    except FileNotFoundError:
        cli_utils.CONSOLE.print(
            f"[yellow]Could not open editor '{editor}'.[/yellow]\n"
            f"[dim]Please edit the file manually: {cache_path}[/dim]\n"
        )
        Prompt.ask("Press Enter when done editing")
    except subprocess.CalledProcessError:
        cli_utils.CONSOLE.print("[yellow]Editor closed without saving changes.[/yellow]")
