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

"""CLI commands for listing available models, datasets, and configs."""

import re
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.tree import Tree

from oumi.cli.cli_utils import CONSOLE, section_header


def _apply_filter(items: dict, pattern: Optional[str]) -> dict:
    """Apply regex filter to dictionary items.

    Args:
        items: Dictionary of items to filter.
        pattern: Regex pattern to match against keys.

    Returns:
        Filtered dictionary.
    """
    if not pattern:
        return items

    try:
        regex = re.compile(pattern, re.IGNORECASE)
        return {k: v for k, v in items.items() if regex.search(k)}
    except re.error as e:
        CONSOLE.print(f"[red]Invalid regex pattern: {e}[/red]")
        raise typer.Exit(1)


def _get_dataset_type(dataset_class) -> str:
    """Infer dataset type from class or module path.

    Args:
        dataset_class: The dataset class.

    Returns:
        Dataset type string.
    """
    if not hasattr(dataset_class, "__module__"):
        return "unknown"

    module = dataset_class.__module__
    if "sft" in module:
        return "sft"
    elif "pretraining" in module:
        return "pretraining"
    elif "vision" in module or "vision_language" in module:
        return "vision"
    elif "preference" in module or "dpo" in module or "kto" in module:
        return "preference"
    elif "grpo" in module:
        return "grpo"
    elif "evaluation" in module:
        return "evaluation"
    else:
        return "other"


def list_datasets(
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Regex pattern to filter dataset names"
    ),
    type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by dataset type "
        "(sft, pretraining, vision, preference, grpo, evaluation)",
    ),
):
    """List all registered datasets."""
    # Delayed imports to avoid slow startup
    from oumi.core.registry import REGISTRY, RegistryType

    section_header("Available datasets:")

    datasets = REGISTRY.get_all(RegistryType.DATASET)

    if not datasets:
        CONSOLE.print("[yellow]No datasets found in registry.[/yellow]")
        return

    # Apply filter
    if filter:
        datasets = _apply_filter(datasets, filter)

    # Apply type filter
    if type:
        type_lower = type.lower()
        datasets = {
            k: v for k, v in datasets.items() if _get_dataset_type(v) == type_lower
        }

    if not datasets:
        CONSOLE.print(
            f"[yellow]No datasets found matching criteria "
            f"(filter={filter}, type={type}).[/yellow]"
        )
        return

    # Create table
    table = Table(show_header=True, show_lines=False)
    table.add_column("Dataset Name", justify="left", style="cyan")
    table.add_column("Type", justify="left", style="green")
    table.add_column("Class", justify="left", style="dim")

    # Sort datasets by name
    for name in sorted(datasets.keys()):
        dataset_class = datasets[name]
        dataset_type = _get_dataset_type(dataset_class)
        class_name = (
            dataset_class.__name__ if hasattr(dataset_class, "__name__") else "unknown"
        )
        table.add_row(name, dataset_type, class_name)

    CONSOLE.print(table)
    CONSOLE.print(f"\n[dim]Total: {len(datasets)} dataset(s)[/dim]")


def list_models(
    filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="Regex pattern to filter model types"
    ),
    tested_only: bool = typer.Option(
        False, "--tested-only", help="Show only tested/verified models"
    ),
):
    """List all supported model types."""
    # Delayed imports
    from oumi.core.configs.internal.supported_models import get_all_models_map

    section_header("Supported models:")

    models_map = get_all_models_map()

    if not models_map:
        CONSOLE.print("[yellow]No models found.[/yellow]")
        return

    # Convert to dict for filtering
    models_dict = {k: v for k, v in models_map.items()}

    # Apply filter
    if filter:
        models_dict = _apply_filter(models_dict, filter)

    # Apply tested_only filter
    if tested_only:
        models_dict = {k: v for k, v in models_dict.items() if v.tested}

    if not models_dict:
        CONSOLE.print(
            f"[yellow]No models found matching criteria "
            f"(filter={filter}, tested_only={tested_only}).[/yellow]"
        )
        return

    # Create table
    table = Table(show_header=True, show_lines=False)
    table.add_column("Model Type", justify="left", style="cyan")
    table.add_column("Tested", justify="center", style="green")
    table.add_column("Model Class", justify="left", style="dim")

    # Sort models by type
    for model_type in sorted(models_dict.keys()):
        model_info = models_dict[model_type]
        tested = "✓" if model_info.tested else "✗"
        model_class = (
            model_info.model_class.__name__
            if hasattr(model_info.model_class, "__name__")
            else str(model_info.model_class)
        )
        table.add_row(model_type, tested, model_class)

    CONSOLE.print(table)
    CONSOLE.print(f"\n[dim]Total: {len(models_dict)} model(s)[/dim]")


def list_configs(
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by category (recipes, apis, projects, examples)",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Filter by model name (e.g., llama4, phi3)"
    ),
):
    """List available configuration files."""
    # Find configs directory relative to this file
    configs_dir = Path(__file__).parent.parent.parent.parent / "configs"

    if not configs_dir.exists():
        CONSOLE.print(f"[red]Configs directory not found: {configs_dir}[/red]")
        raise typer.Exit(1)

    section_header("Available configuration files:")

    # Determine search path based on filters
    if category:
        search_dir = configs_dir / category
        if not search_dir.exists():
            CONSOLE.print(
                f"[yellow]Category '{category}' "
                "not found in configs directory.[/yellow]"
            )
            return
    else:
        search_dir = configs_dir

    # Find all YAML config files
    config_files = list(search_dir.rglob("*.yaml"))

    # Apply model filter
    if model:
        config_files = [f for f in config_files if model.lower() in str(f).lower()]

    if not config_files:
        CONSOLE.print(
            f"[yellow]No config files found matching criteria "
            f"(category={category}, model={model}).[/yellow]"
        )
        return

    # Organize configs by category
    by_category = {}
    for config_file in config_files:
        rel_path = config_file.relative_to(configs_dir)
        parts = rel_path.parts
        if len(parts) > 0:
            cat = parts[0]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(str(rel_path))

    # Create tree structure
    tree = Tree("[bold cyan]configs/[/bold cyan]")

    for cat in sorted(by_category.keys()):
        cat_node = tree.add(f"[green]{cat}/[/green]")
        for config_path in sorted(by_category[cat]):
            # Remove category from path for display
            display_path = "/".join(Path(config_path).parts[1:])
            cat_node.add(f"[dim]{display_path}[/dim]")

    CONSOLE.print(tree)
    CONSOLE.print(f"\n[dim]Total: {len(config_files)} config file(s)[/dim]")


def list_registry(
    type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by registry type (CLOUD, DATASET, MODEL, METRICS_FUNCTION, etc.)",
    ),
):
    """List all items in the global registry."""
    # Delayed imports
    from oumi.core.registry import REGISTRY, RegistryType

    section_header("Registry contents:")

    # Determine which registry types to show
    if type:
        try:
            registry_type = RegistryType[type.upper()]
            registry_types = [registry_type]
        except KeyError:
            valid_types = [t.name for t in RegistryType]
            CONSOLE.print(
                f"[red]Invalid registry type: {type}[/red]\n"
                f"Valid types: {', '.join(valid_types)}"
            )
            raise typer.Exit(1)
    else:
        registry_types = list(RegistryType)

    # Collect all items
    all_items = {}
    for reg_type in registry_types:
        items = REGISTRY.get_all(reg_type)
        if items:
            all_items[reg_type.name] = items

    if not all_items:
        CONSOLE.print("[yellow]No items found in registry.[/yellow]")
        return

    # Create table for each registry type
    for reg_type_name in sorted(all_items.keys()):
        items = all_items[reg_type_name]

        CONSOLE.print(f"\n[bold green]{reg_type_name}:[/bold green]")
        table = Table(show_header=True, show_lines=False, box=None)
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Value", justify="left", style="dim")

        for name in sorted(items.keys()):
            value = items[name]
            value_str = (
                value.__name__
                if hasattr(value, "__name__")
                else str(type(value).__name__)  # type: ignore
            )
            table.add_row(name, value_str)

        CONSOLE.print(table)
        CONSOLE.print(f"[dim]  {len(items)} item(s)[/dim]")

    total_items = sum(len(items) for items in all_items.values())
    CONSOLE.print(
        f"\n[dim]Total: {total_items} item(s) across all registry types[/dim]"
    )
