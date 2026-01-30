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

from pathlib import Path
from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.utils.logging import logger

# Valid output formats for analysis results
_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")


def _run_typed_analysis_cli(
    config: str,
    output: str | None,
    output_format: str,
    list_metrics: bool,
    verbose: bool,
) -> None:
    """Run analysis using the new typed analyzer system.

    This is the handler for the --typed flag in the CLI.

    Args:
        config: Path to the configuration file.
        output: Output directory override.
        output_format: Output format (csv, json, parquet).
        list_metrics: Whether to just list available metrics.
        verbose: Enable verbose output.
    """
    from oumi.analyze import (
        TypedAnalyzeConfig,
        print_summary,
        run_typed_analysis,
        save_results,
    )
    from oumi.analyze import list_metrics as list_metrics_func

    try:
        # Handle --list-metrics for typed system
        if list_metrics:
            cli_utils.CONSOLE.print(
                "\n[bold cyan]Typed Analyzer System - Available Metrics[/bold cyan]\n"
            )
            # Show built-in analyzers
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

                        # Create table matching built-in format
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

        # Load typed config
        with cli_utils.CONSOLE.status(
            "[green]Loading typed configuration...[/green]", spinner="dots"
        ):
            typed_config = TypedAnalyzeConfig.from_yaml(config)

        # Override output path if provided
        if output:
            typed_config.output_path = output

        if verbose:
            cli_utils.CONSOLE.print(f"[dim]Config loaded from: {config}[/dim]")
            dataset = typed_config.dataset_name or typed_config.dataset_path
            cli_utils.CONSOLE.print(f"[dim]Dataset: {dataset}[/dim]")
            analyzer_ids = [a.id for a in typed_config.analyzers]
            cli_utils.CONSOLE.print(f"[dim]Analyzers: {analyzer_ids}[/dim]")

        # Run analysis
        with cli_utils.CONSOLE.status(
            "[green]Running typed analysis...[/green]", spinner="dots"
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

        # Save to eval storage for web viewer
        try:
            from dataclasses import asdict

            from oumi.analyze.storage import AnalyzeStorage

            storage = AnalyzeStorage()
            # Use eval_name from config if provided, otherwise use config filename
            eval_name = typed_config.eval_name or Path(config).stem

            # Convert config to dict (it's a dataclass)
            config_dict = asdict(typed_config)

            # Handle test_summary - may be a Pydantic model or dict
            test_summary = results.get("test_summary")
            if test_summary is not None:
                if hasattr(test_summary, "model_dump"):
                    test_summary = test_summary.model_dump()
                elif not isinstance(test_summary, dict):
                    test_summary = {}

            # Serialize conversations for storage
            conversations_data = []
            raw_conversations = results.get("conversations", [])
            for conv in raw_conversations:
                if hasattr(conv, "to_dict"):
                    conversations_data.append(conv.to_dict())
                elif hasattr(conv, "model_dump"):
                    conversations_data.append(conv.model_dump())
                elif isinstance(conv, dict):
                    conversations_data.append(conv)

            eval_id = storage.save_eval(
                name=eval_name,
                config=config_dict,
                analysis_results=results.get("results", {}),
                test_results=test_summary or {},
                config_path=config,
                dataset_path=typed_config.dataset_path,
                conversations=conversations_data,
            )
            cli_utils.CONSOLE.print(
                f"[dim]Eval saved (ID: {eval_id}). View with: oumi analyze view[/dim]"
            )
        except Exception as e:
            logger.debug(f"Could not save to eval storage: {e}")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Configuration file not found: {e}")
        raise typer.Exit(code=1)

    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Invalid configuration: {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Typed analysis failed: {e}", exc_info=True)
        cli_utils.CONSOLE.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


def analyze(
    ctx: typer.Context,
    config: Annotated[
        str | None,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for analysis.",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for analysis results. Overrides config output_path.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format for results: csv, json, or parquet (case-insensitive).",
        ),
    ] = "csv",
    list_metrics: Annotated[
        bool,
        typer.Option(
            "--list-metrics",
            help="List all available metrics for test configurations.",
        ),
    ] = False,
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Analyze a dataset to compute metrics and statistics.

    Uses the typed analyzer system with Pydantic-based results.
    Metrics are accessed via paths like 'LengthAnalyzer.total_words'.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for analysis.
        output: Output directory for results. Overrides config output_path.
        output_format: Output format (csv, json, parquet). Case-insensitive.
        list_metrics: List available metrics without running analysis.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
    """
    # If a subcommand is being invoked, don't run the main analyze logic
    if ctx.invoked_subcommand is not None:
        return

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

    # Run typed analysis
    _run_typed_analysis_cli(
        config=config,
        output=output,
        output_format=output_format,
        list_metrics=list_metrics,
        verbose=verbose,
    )


def analyze_view(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the web viewer on.",
        ),
    ] = 8765,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Host address to bind to.",
        ),
    ] = "localhost",
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Don't automatically open the browser.",
        ),
    ] = False,
):
    """Launch the web viewer for analysis results.

    The web viewer provides:
    - Browse and compare past analysis runs
    - Interactive results table with filters
    - Charts and visualizations
    - Export options

    Args:
        port: Port to run the web viewer on.
        host: Host address to bind to.
        no_browser: Don't automatically open the browser.
    """
    from oumi.analyze.serve import serve_ui

    try:
        serve_ui(port=port, host=host, open_browser=not no_browser)
    except KeyboardInterrupt:
        cli_utils.CONSOLE.print("\n[dim]Web viewer stopped.[/dim]")
