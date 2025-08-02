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
from typing import Annotated, Optional

import typer
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
        if verbose:
            table.add_column("Last Modified", style="dim", justify="right")
            table.add_column("Last Accessed", style="dim", justify="right")
            table.add_column("Repo Type", style="dim", justify="right")
            table.add_column("Number of Files", style="dim", justify="right")
            table.add_column("Repo Path", style="dim", justify="right")

        # Add rows to table
        for model in cached_models:
            row = [model.repo_id, model.size]
            if verbose:
                row.extend(
                    [
                        str(model.last_modified),
                        str(model.last_accessed),
                        str(model.repo_type),
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
