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

import fnmatch
import shutil
from typing import Annotated, Optional

import typer
from huggingface_hub import model_info, snapshot_download
from rich.console import Console
from rich.table import Table

from oumi.utils.hf_model_cache_utils import format_size, list_cached_models

console = Console()


def ls(
    filter_pattern: Annotated[
        Optional[str],
        typer.Option(
            "--filter",
            "-f",
            help="Filter models by pattern (supports wildcards like '*llama*')",
        ),
    ] = None,
    sort_by: Annotated[
        str, typer.Option("--sort", help="Sort by: size, name (default: size)")
    ] = "size",
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show detailed information")
    ] = False,
):
    """List locally cached Hugging Face models."""
    try:
        # Get cached models
        cached_models = list_cached_models()

        if not cached_models:
            console.print("[yellow]No cached models found.[/yellow]")
            return

        # Apply filtering if specified
        if filter_pattern:
            cached_models = [
                model
                for model in cached_models
                if fnmatch.fnmatch(model.repo_id.lower(), filter_pattern.lower())
            ]

            if not cached_models:
                console.print(
                    f"[yellow]No models found matching pattern "
                    f"'{filter_pattern}'[/yellow]"
                )
                return

        # Sort models
        if sort_by == "size":
            cached_models.sort(key=lambda x: x.size_bytes, reverse=True)
        elif sort_by == "name":
            cached_models.sort(key=lambda x: x.repo_id)
        else:
            console.print(
                f"[red]Invalid sort option: {sort_by}. Use 'size' or 'name'.[/red]"
            )
            return

        # Create and populate table
        table = Table(title="Cached Hugging Face Models")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Size", style="green", justify="right")
        table.add_column("Repo Type", style="dim", justify="right")
        if verbose:
            table.add_column("Last Modified", style="dim", justify="right")
            table.add_column("Last Accessed", style="dim", justify="right")
            table.add_column("Number of Files", style="dim", justify="right")
            table.add_column("Repo Path", style="dim", justify="right")

        # Add rows to table
        for model in cached_models:
            row = [model.repo_id, model.size, model.repo_type]
            if verbose:
                row.extend(
                    [
                        str(model.last_modified),
                        str(model.last_accessed),
                        str(model.nb_files),
                        str(model.repo_path),
                    ]
                )
            table.add_row(*row)
        # Print the table
        console.print(table)
        # Summary
        total_size_bytes = sum(model.size_bytes for model in cached_models)
        total_size_str = format_size(total_size_bytes)
        console.print(f"\nTotal: {len(cached_models)} models, {total_size_str}")

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        raise typer.Exit(1)


def rm(
    model_id: Annotated[str, typer.Argument(help="Model ID to remove from cache")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force removal without confirmation")
    ] = False,
):
    """Remove a cached Hugging Face model."""
    try:
        cached_models = list_cached_models()

        # Find the model to remove
        model_to_remove = None
        for model in cached_models:
            if model.repo_id == model_id:
                model_to_remove = model
                break

        if not model_to_remove:
            console.print(f"[red]Model '{model_id}' not found in cache.[/red]")
            raise typer.Exit(1)

        # Confirm removal unless force flag is used
        if not force:
            confirm = typer.confirm(
                f"Remove model '{model_id}' ({model_to_remove.size}) from cache?"
            )
            if not confirm:
                console.print("[yellow]Removal cancelled.[/yellow]")
                return

        # Remove the model directory
        shutil.rmtree(model_to_remove.repo_path)
        console.print(
            f"""[green]Successfully removed model '{model_id}' ({model_to_remove.size}).
            [/green]"""
        )

    except Exception as e:
        console.print(f"[red]Error removing model: {e}[/red]")
        raise typer.Exit(1)


def get(
    model_id: Annotated[str, typer.Argument(help="Model ID to download/cache")],
    revision: Annotated[
        Optional[str], typer.Option("--revision", help="Model revision to download")
    ] = None,
):
    """Download and cache a Hugging Face model."""
    try:
        # Check if model is already cached
        cached_models = list_cached_models()
        for model in cached_models:
            if model.repo_id == model_id:
                console.print(
                    f"""[green]Model '{model_id}' is already cached ({model.size}).
                    [/green]"""
                )
                return

        console.print(f"[blue]Downloading model '{model_id}'...[/blue]")
        if revision:
            console.print(f"[dim]Revision: {revision}[/dim]")

        # Download the model
        snapshot_download(repo_id=model_id, revision=revision)
        console.print(
            f"[green]Successfully downloaded and cached model '{model_id}'.[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        raise typer.Exit(1)


def card(
    model_id: Annotated[str, typer.Argument(help="Model ID to show information for")],
):
    """Show model card information for a Hugging Face model."""
    try:
        # Check if model is cached locally first
        cached_models = list_cached_models()
        cached_model = None
        for model in cached_models:
            if model.repo_id == model_id:
                cached_model = model
                break

        # Create info table
        table = Table(title=f"Model Information: {model_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        if cached_model:
            table.add_row("Status", "[green]Cached locally[/green]")
            table.add_row("Size", cached_model.size)
            table.add_row("Last Modified", cached_model.last_modified)
            table.add_row("Last Accessed", cached_model.last_accessed)
            table.add_row("Repo Type", cached_model.repo_type)
            table.add_row("Number of Files", str(cached_model.nb_files))
        else:
            table.add_row("Status", "[yellow]Not cached locally[/yellow]")

        # Fetch model info from HF Hub
        try:
            info = model_info(model_id)
            table.add_row(
                "Model Type", str(info.pipeline_tag) if info.pipeline_tag else "Unknown"
            )
            table.add_row(
                "Downloads", str(info.downloads) if info.downloads else "Unknown"
            )
            table.add_row("Likes", str(info.likes) if info.likes else "0")
            table.add_row(
                "Library", str(info.library_name) if info.library_name else "Unknown"
            )
        except Exception:
            table.add_row(
                "Hub Info", "[dim]Unable to fetch from Hugging Face Hub[/dim]"
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error showing model card: {e}[/red]")
        raise typer.Exit(1)
