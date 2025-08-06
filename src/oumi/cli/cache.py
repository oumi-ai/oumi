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

from oumi.utils.hf_cache_utils import format_size, list_hf_cache

console = Console()


def ls(
    filter_pattern: Annotated[
        Optional[str],
        typer.Option(
            "--filter",
            "-f",
            help="Filter cached items by pattern (supports wildcards like '*llama*')",
        ),
    ] = None,
    sort_by: Annotated[
        str, typer.Option("--sort", help="Sort by: size, name (default: size)")
    ] = "size",
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show detailed information")
    ] = False,
):
    """List locally cached Hugging Face items."""
    try:
        # Get cached items
        cached_items = list_hf_cache()

        if not cached_items:
            console.print("[yellow]No cached items found.[/yellow]")
            return

        # Apply filtering if specified
        if filter_pattern:
            cached_items = [
                item
                for item in cached_items
                if fnmatch.fnmatch(item.repo_id.lower(), filter_pattern.lower())
            ]

            if not cached_items:
                console.print(
                    f"[yellow]No items found matching pattern "
                    f"'{filter_pattern}'[/yellow]"
                )
                return

        # Sort items
        if sort_by == "size":
            cached_items.sort(key=lambda x: x.size_bytes, reverse=True)
        elif sort_by == "name":
            cached_items.sort(key=lambda x: x.repo_id)
        else:
            console.print(
                f"[red]Invalid sort option: {sort_by}. Use 'size' or 'name'.[/red]"
            )
            return

        # Create and populate table
        table = Table(title="Cached Hugging Face Items")
        table.add_column("Repository ID", style="cyan", no_wrap=True)
        table.add_column("Size", style="green", justify="right")
        table.add_column("Repo Type", style="dim", justify="right")
        if verbose:
            table.add_column("Last Modified", style="dim", justify="right")
            table.add_column("Last Accessed", style="dim", justify="right")
            table.add_column("Number of Files", style="dim", justify="right")
            table.add_column("Repo Path", style="dim", justify="right")

        # Add rows to table
        for item in cached_items:
            row = [item.repo_id, item.size, item.repo_type]
            if verbose:
                row.extend(
                    [
                        str(item.last_modified),
                        str(item.last_accessed),
                        str(item.nb_files),
                        str(item.repo_path),
                    ]
                )
            table.add_row(*row)
        # Print the table
        console.print(table)
        # Summary
        total_size_bytes = sum(item.size_bytes for item in cached_items)
        total_size_str = format_size(total_size_bytes)
        console.print(f"\nTotal: {len(cached_items)} items, {total_size_str}")

    except Exception as e:
        console.print(f"[red]Error listing cached items: {e}[/red]")
        raise typer.Exit(1)


def rm(
    repo_id: Annotated[str, typer.Argument(help="Repository ID to remove from cache")],
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force removal without confirmation")
    ] = False,
):
    """Remove a cached Hugging Face item."""
    try:
        cached_items = list_hf_cache()

        # Find the item to remove
        item_to_remove = None
        for item in cached_items:
            if item.repo_id == repo_id:
                item_to_remove = item
                break

        if not item_to_remove:
            console.print(f"[red]Repository '{repo_id}' not found in cache.[/red]")
            raise typer.Exit(1)

        # Confirm removal unless force flag is used
        if not force:
            confirm = typer.confirm(
                f"Remove '{repo_id}' ({item_to_remove.size}) from cache?"
            )
            if not confirm:
                console.print("[yellow]Removal cancelled.[/yellow]")
                return

        # Remove the repository directory
        shutil.rmtree(item_to_remove.repo_path)
        console.print(
            f"""[green]Successfully removed '{repo_id}' ({item_to_remove.size}).
            [/green]"""
        )

    except Exception as e:
        console.print(f"[red]Error removing cached item: {e}[/red]")
        raise typer.Exit(1)


def get(
    repo_id: Annotated[str, typer.Argument(help="Repository ID to download/cache")],
    revision: Annotated[
        Optional[str],
        typer.Option("--revision", help="Repository revision to download"),
    ] = None,
):
    """Download and cache a Hugging Face repository."""
    try:
        # Check if repository is already cached
        cached_items = list_hf_cache()
        for item in cached_items:
            if item.repo_id == repo_id:
                console.print(
                    f"""[green]Repository '{repo_id}' is already cached ({item.size}).
                    [/green]"""
                )
                return

        console.print(f"[blue]Downloading repository '{repo_id}'...[/blue]")
        if revision:
            console.print(f"[dim]Revision: {revision}[/dim]")

        # Download the repository
        snapshot_download(repo_id=repo_id, revision=revision)
        console.print(
            f"[green]Successfully downloaded and cached repository '{repo_id}'.[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error downloading repository: {e}[/red]")
        raise typer.Exit(1)


def card(
    repo_id: Annotated[
        str, typer.Argument(help="Repository ID to show information for")
    ],
):
    """Show repository information for a Hugging Face repository."""
    try:
        # Check if repository is cached locally first
        cached_items = list_hf_cache()
        cached_item = None
        for item in cached_items:
            if item.repo_id == repo_id:
                cached_item = item
                break

        # Create info table
        table = Table(title=f"Repository Information: {repo_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        if cached_item:
            table.add_row("Status", "[green]Cached locally[/green]")
            table.add_row("Size", cached_item.size)
            table.add_row("Last Modified", cached_item.last_modified)
            table.add_row("Last Accessed", cached_item.last_accessed)
            table.add_row("Repo Type", cached_item.repo_type)
            table.add_row("Number of Files", str(cached_item.nb_files))
        else:
            table.add_row("Status", "[yellow]Not cached locally[/yellow]")

        # Fetch repository info from HF Hub
        try:
            info = model_info(repo_id)
            table.add_row(
                "Pipeline Type",
                str(info.pipeline_tag) if info.pipeline_tag else "Unknown",
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
        console.print(f"[red]Error showing repository information: {e}[/red]")
        raise typer.Exit(1)
