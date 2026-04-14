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

from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_analyze_config
from oumi.utils.logging import logger

# Valid output formats for analysis results
_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.ANALYZE, "Available Analysis Configs", "analyze"
)

# Fields that only exist in the old AnalyzeConfig (v1) format.
_OLD_CONFIG_FIELDS = {
    "dataset_source",
    "dataset_format",
    "processor_name",
    "processor_kwargs",
    "is_multimodal",
    "trust_remote_code",
}


def _check_old_config_format(config_path: str) -> None:
    """Check if a config file uses the old AnalyzeConfig (v1) format.

    If old-format fields are detected, prints a helpful migration message
    and exits.

    Args:
        config_path: Path to the YAML config file.
    """
    import yaml

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except Exception:
        return  # Let the normal loader handle errors

    if not isinstance(data, dict):
        return

    old_fields_found = _OLD_CONFIG_FIELDS & set(data.keys())
    if old_fields_found:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] This config file uses the old AnalyzeConfig (v1) "
            "format.\n\n"
            f"Detected v1 fields: {', '.join(sorted(old_fields_found))}\n\n"
            "The analyze CLI now uses TypedAnalyzeConfig (v2). Key changes:\n"
            "  - 'analyzers' entries use 'type' and 'display_name' "
            "(not 'id'/'instance_id')\n"
            "  - Metrics are accessed as '{display_name}.{field}' "
            "(e.g. 'Length.total_tokens')\n"
            "  - 'dataset_source', 'processor_name', 'is_multimodal' "
            "are no longer needed\n\n"
            "See: docs/user_guides/analyze/analyze_config.md for the new format.\n"
            "Example: configs/examples/analyze/analyze.yaml"
        )
        raise typer.Exit(code=1)


def _run_typed_analysis_cli(
    config: str,
    output: str | None,
    output_format: str,
    list_metrics: bool,
    verbose: bool,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    sample_count: int | None = None,
) -> None:
    """Run analysis using the typed analyzer system.

    Args:
        config: Path to the configuration file.
        output: Output directory override.
        output_format: Output format (csv, json, parquet).
        list_metrics: Whether to just list available metrics.
        verbose: Enable verbose output.
        dataset_name: Override dataset name from CLI.
        dataset_path: Override dataset path from CLI.
        sample_count: Override sample count from CLI.
    """
    from oumi.analyze import (
        TypedAnalyzeConfig,
        print_summary,
        run_typed_analysis,
        save_results,
    )
    from oumi.analyze import list_metrics as list_metrics_func

    try:
        # Handle --list-metrics
        if list_metrics:
            cli_utils.CONSOLE.print("\n[bold cyan]Available Metrics[/bold cyan]\n")
            list_metrics_func()

            # Also show custom metrics from config if provided
            try:
                typed_config = TypedAnalyzeConfig.from_yaml(config)
                if typed_config.custom_metrics:
                    from rich.table import Table

                    for cm in typed_config.custom_metrics:
                        cli_utils.CONSOLE.print(
                            f"\n[bold]{cm.id}[/bold] "
                            f"[green]({cm.scope} scope)[/green] "
                            f"[dim](custom)[/dim]"
                        )
                        if cm.description:
                            cli_utils.CONSOLE.print(f"[dim]{cm.description}[/dim]\n")

                        table = Table(
                            show_header=True,
                            header_style="bold",
                            box=None,
                            padding=(0, 2),
                        )
                        table.add_column("Metric Path", style="cyan")
                        table.add_column("Type", style="yellow", width=15)
                        table.add_column("Description", style="white")

                        if cm.output_schema:
                            for f in cm.output_schema:
                                table.add_row(
                                    f"{cm.id}.{f.name}",
                                    f.type,
                                    f.description,
                                )
                        else:
                            table.add_row(
                                f"{cm.id}.<field>",
                                "any",
                                "Add output_schema to config for field details",
                            )

                        cli_utils.CONSOLE.print(table)
                        cli_utils.CONSOLE.print()
            except Exception:
                pass  # Config may not exist yet, that's OK
            return

        # Detect old config format before loading
        _check_old_config_format(config)

        # Load config
        with cli_utils.CONSOLE.status(
            "[green]Loading configuration...[/green]", spinner="dots"
        ):
            typed_config = TypedAnalyzeConfig.from_yaml(config)

        # Apply CLI overrides
        if output:
            typed_config.output_path = output
        if dataset_name is not None:
            typed_config.dataset_name = dataset_name
        if dataset_path is not None:
            typed_config.dataset_path = dataset_path
        if sample_count is not None:
            typed_config.sample_count = sample_count

        if verbose:
            cli_utils.CONSOLE.print(f"[dim]Config loaded from: {config}[/dim]")
            dataset = typed_config.dataset_name or typed_config.dataset_path
            cli_utils.CONSOLE.print(f"[dim]Dataset: {dataset}[/dim]")
            analyzer_ids = [a.display_name for a in typed_config.analyzers]
            cli_utils.CONSOLE.print(f"[dim]Analyzers: {analyzer_ids}[/dim]")

        # Run analysis
        with cli_utils.CONSOLE.status(
            "[green]Running analysis...[/green]", spinner="dots"
        ):
            results = run_typed_analysis(typed_config)

        # Print summary
        print_summary(results)

        # Save results
        if typed_config.output_path:
            save_results(typed_config.output_path, results, output_format)
            cli_utils.CONSOLE.print(
                f"\n[green]Results saved to:[/green] {typed_config.output_path}"
            )

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Configuration file not found: {e}")
        raise typer.Exit(code=1)

    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Invalid configuration: {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        cli_utils.CONSOLE.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def analyze(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str | None,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name for analysis.",
            rich_help_panel="Options",
            autocompletion=complete_analyze_config,
        ),
    ] = None,
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available analysis configs.",
            callback=_list_configs_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    list_metrics: Annotated[
        bool,
        typer.Option(
            "--list-metrics",
            help="List all available metrics for test configurations.",
            rich_help_panel="Options",
        ),
    ] = False,
    level: Annotated[
        cli_utils.LogLevel | None,
        typer.Option(
            "--log-level",
            "-log",
            help="Logging level.",
            show_default=False,
            show_choices=True,
            case_sensitive=False,
            callback=cli_utils.set_log_level,
            rich_help_panel="Options",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output.",
            rich_help_panel="Options",
        ),
    ] = False,
    # Data overrides
    dataset_name: Annotated[
        str | None,
        typer.Option(
            "--dataset_name",
            help="Dataset name to analyze (overrides config).",
            rich_help_panel="Data",
        ),
    ] = None,
    dataset_path: Annotated[
        str | None,
        typer.Option(
            "--dataset_path",
            help="Path to dataset file in JSONL format (overrides config).",
            rich_help_panel="Data",
        ),
    ] = None,
    sample_count: Annotated[
        int | None,
        typer.Option(
            "--sample_count",
            help="Number of samples to analyze (overrides config).",
            rich_help_panel="Data",
        ),
    ] = None,
    # Output options
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for analysis results (overrides config).",
            rich_help_panel="Output",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format for results: csv, json, or parquet.",
            rich_help_panel="Output",
        ),
    ] = "csv",
):
    """Analyze a dataset to compute metrics and statistics.

    Metrics are accessed via paths like 'Length.total_tokens'.

    Args:
        ctx: The Typer context object.
        config: Path or config name for analysis.
        list_configs: List all available analysis configs.
        list_metrics: List available metrics without running analysis.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        dataset_name: Dataset name override.
        dataset_path: Dataset path override.
        sample_count: Sample count override.
        output: Output directory override.
        output_format: Output format (csv, json, parquet).
    """
    # If a subcommand is being invoked, don't run the main analyze logic
    if ctx.invoked_subcommand is not None:
        return

    # --list-metrics doesn't require a config
    if list_metrics:
        _run_typed_analysis_cli(
            config=config or "",
            output=output,
            output_format="csv",
            list_metrics=True,
            verbose=verbose,
        )
        return

    # Resolve config aliases
    if config is not None:
        config = str(
            cli_utils.resolve_and_fetch_config(
                try_get_config_name_for_alias(config, AliasType.ANALYZE),
            )
        )

    # Config is required when running analyze directly (not as a subcommand)
    if config is None:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] Missing option '--config' / '-c'.\n"
            "Run 'oumi analyze --help' for usage."
        )
        raise typer.Exit(code=1)

    # Validate output format early before any expensive operations
    output_format = output_format.lower()
    if output_format not in _VALID_OUTPUT_FORMATS:
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Invalid output format '{output_format}'. "
            f"Supported formats: {', '.join(_VALID_OUTPUT_FORMATS)}"
        )
        raise typer.Exit(code=1)

    # Run analysis
    _run_typed_analysis_cli(
        config=config,
        output=output,
        output_format=output_format,
        list_metrics=False,
        verbose=verbose,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        sample_count=sample_count,
    )
