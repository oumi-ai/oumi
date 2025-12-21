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

"""Config management commands for oumi CLI."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from oumi.cli.alias import AliasType, _ALIASES
from oumi.cli.config_index import (
    filter_configs,
    generate_config_index_from_aliases,
    get_config_metadata,
    load_config_index,
    parse_filter_expression,
    save_config_index,
)
from oumi.core.configs.metadata import get_vram_tier

CONSOLE = Console()


def index(
    regenerate: bool = typer.Option(
        False,
        "--regenerate",
        "-r",
        help="Force regenerate the index even if it exists.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress.",
    ),
) -> None:
    """Generate or update the config metadata index.

    The index is used for fast config lookup and filtering.
    """
    CONSOLE.print("[cyan]Generating config index...[/cyan]")

    index_data = generate_config_index_from_aliases(verbose=verbose)
    save_config_index(index_data)

    num_configs = len(index_data.get("configs", {}))
    CONSOLE.print(f"[green]Index generated with {num_configs} configs.[/green]")


def show(
    alias: str = typer.Argument(..., help="Config alias to show metadata for."),
) -> None:
    """Show detailed metadata for a config.

    Example:
        oumi configs show llama3.1-8b-lora
    """
    metadata = get_config_metadata(alias)

    if metadata is None:
        CONSOLE.print(f"[red]Config not found: {alias}[/red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Metadata for {alias}", show_header=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Config Type", metadata.config_type.value if metadata.config_type else "-")
    table.add_row("Model Family", metadata.model_family or "-")
    table.add_row("Model Size", f"{metadata.model_size_billions}B" if metadata.model_size_billions else "-")
    table.add_row("Training Method", metadata.training_method.value if metadata.training_method else "-")
    table.add_row("Finetuning Type", metadata.finetuning_type.value if metadata.finetuning_type else "-")
    table.add_row("Min VRAM", f"~{metadata.min_vram_gb:.1f} GB" if metadata.min_vram_gb else "-")
    table.add_row("Recommended GPUs", str(metadata.recommended_gpus) if metadata.recommended_gpus else "-")
    table.add_row("Vision Model", "Yes" if metadata.is_vision_model else "No")
    table.add_row("Tags", ", ".join(metadata.tags) if metadata.tags else "-")
    table.add_row("Description", metadata.description or "-")

    if metadata.min_vram_gb:
        table.add_row("Hardware Tier", get_vram_tier(metadata.min_vram_gb))

    CONSOLE.print()
    CONSOLE.print(table)
    CONSOLE.print()


def search(
    filter_expr: str = typer.Argument(
        ...,
        help="Filter expression (e.g., 'vram<24,family=llama,type=qlora').",
    ),
    config_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by config type (training, inference, evaluation, job).",
    ),
) -> None:
    """Search configs by metadata criteria.

    Filter expressions support:
    - vram<N or vram<=N: Max VRAM requirement in GB
    - family=X: Model family (llama, qwen, gemma, etc.)
    - type=X: Finetuning type (full, lora, qlora)
    - method=X: Training method (sft, dpo, grpo, kto)
    - tag=X: Config tag
    - config=X: Config type (training, inference, evaluation)

    Multiple filters can be combined with commas.

    Examples:
        oumi configs search "vram<24"
        oumi configs search "family=llama,type=qlora"
        oumi configs search "vram<48,method=dpo"
    """
    from oumi.core.configs.metadata import ConfigType

    index = load_config_index()

    # Parse filter expression
    criteria = parse_filter_expression(filter_expr)

    # Add config type filter if specified
    if config_type:
        try:
            criteria["config_type"] = ConfigType(config_type.lower())
        except ValueError:
            CONSOLE.print(f"[red]Invalid config type: {config_type}[/red]")
            raise typer.Exit(code=1)

    # Apply filters
    filtered = filter_configs(index, **criteria)

    if not filtered:
        CONSOLE.print(f"[yellow]No configs found matching: {filter_expr}[/yellow]")
        raise typer.Exit(code=0)

    # Build results table
    table = Table(
        title=f"Configs matching: {filter_expr}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Config", style="green")
    table.add_column("Type", style="blue")
    table.add_column("Model", style="cyan")
    table.add_column("Method", style="yellow")
    table.add_column("VRAM", style="magenta")

    for key, meta in sorted(filtered.items()):
        cfg_type = meta.get("config_type", "-")

        model_size = meta.get("model_size_billions")
        model_str = f"{model_size}B" if model_size else "-"

        method = meta.get("training_method", "")
        ftype = meta.get("finetuning_type", "")
        method_str = f"{method}/{ftype}" if method and ftype else (method or ftype or "-")

        vram = meta.get("min_vram_gb")
        vram_str = f"~{int(vram)} GB" if vram else "-"

        table.add_row(key, cfg_type, model_str, method_str, vram_str)

    CONSOLE.print()
    CONSOLE.print(table)
    CONSOLE.print()
    CONSOLE.print(f"[dim]Found {len(filtered)} matching configs[/dim]")
    CONSOLE.print()


def list_all(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show metadata columns.",
    ),
    config_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by config type (training, inference, evaluation, job).",
    ),
) -> None:
    """List all available configs.

    Example:
        oumi configs list
        oumi configs list --verbose
        oumi configs list --type training
    """
    from oumi.core.configs.metadata import ConfigType

    index = load_config_index()
    configs = index.get("configs", {})

    # Filter by config type if specified
    if config_type:
        try:
            ct = ConfigType(config_type.lower())
            configs = {k: v for k, v in configs.items() if v.get("config_type") == ct.value}
        except ValueError:
            CONSOLE.print(f"[red]Invalid config type: {config_type}[/red]")
            raise typer.Exit(code=1)

    if not configs:
        CONSOLE.print("[yellow]No configs found.[/yellow]")
        raise typer.Exit(code=0)

    # Build table
    title = "All Available Configs"
    if config_type:
        title = f"Available {config_type.title()} Configs"

    table = Table(
        title=title,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Config", style="green")

    if verbose:
        table.add_column("Type", style="blue")
        table.add_column("Model", style="cyan")
        table.add_column("Method", style="yellow")
        table.add_column("VRAM", style="magenta")
    else:
        table.add_column("Type", style="blue")
        table.add_column("Path", style="dim", max_width=60)

    for key in sorted(configs.keys()):
        meta = configs[key]

        if verbose:
            cfg_type = meta.get("config_type", "-")

            model_size = meta.get("model_size_billions")
            model_str = f"{model_size}B" if model_size else "-"

            method = meta.get("training_method", "")
            ftype = meta.get("finetuning_type", "")
            method_str = f"{method}/{ftype}" if method and ftype else (method or ftype or "-")

            vram = meta.get("min_vram_gb")
            vram_str = f"~{int(vram)} GB" if vram else "-"

            table.add_row(key, cfg_type, model_str, method_str, vram_str)
        else:
            cfg_type = meta.get("config_type", "-")
            path = meta.get("path", "-")
            table.add_row(key, cfg_type, path)

    CONSOLE.print()
    CONSOLE.print(table)
    CONSOLE.print()
    CONSOLE.print(f"[dim]Total: {len(configs)} configs[/dim]")
    CONSOLE.print()


# Create the configs subcommand app
configs_app = typer.Typer(
    name="configs",
    help="Manage and explore config metadata.",
    pretty_exceptions_enable=False,
)

configs_app.command(name="index", help="Generate or update the config metadata index.")(index)
configs_app.command(name="show", help="Show detailed metadata for a config.")(show)
configs_app.command(name="search", help="Search configs by metadata criteria.")(search)
configs_app.command(name="list", help="List all available configs.")(list_all)
