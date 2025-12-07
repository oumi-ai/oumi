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

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Optional

import pandas as pd
import typer
from rich.box import ROUNDED
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger

# Valid output formats for analysis results
_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")

if TYPE_CHECKING:
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer


def analyze(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for analysis.",
        ),
    ],
    output: Annotated[
        Optional[str],
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
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            "-r",
            help="Generate an interactive HTML report with charts. Requires plotly.",
        ),
    ] = False,
    report_title: Annotated[
        Optional[str],
        typer.Option(
            "--report-title",
            help="Custom title for the HTML report.",
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Analyze a dataset to compute metrics and statistics.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for analysis.
        output: Output directory for results. Overrides config output_path.
        output_format: Output format (csv, json, parquet). Case-insensitive.
        report: Whether to generate an interactive HTML report with charts.
        report_title: Custom title for the HTML report.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
    """
    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer

    # Validate output format early before any expensive operations
    output_format = output_format.lower()
    if output_format not in _VALID_OUTPUT_FORMATS:
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Invalid output format '{output_format}'. "
            f"Supported formats: {', '.join(_VALID_OUTPUT_FORMATS)}"
        )
        raise typer.Exit(code=1)

    try:
        extra_args = cli_utils.parse_extra_cli_args(ctx)

        config = str(
            cli_utils.resolve_and_fetch_config(
                try_get_config_name_for_alias(config, AliasType.ANALYZE),
            )
        )

        with cli_utils.CONSOLE.status(
            "[green]Loading configuration...[/green]", spinner="dots"
        ):
            # Delayed imports
            from oumi.core.configs import AnalyzeConfig

        # Load configuration
        parsed_config: AnalyzeConfig = AnalyzeConfig.from_yaml_and_arg_list(
            config, extra_args, logger=logger
        )

        # Override output path if provided via CLI
        if output:
            parsed_config.output_path = output

        # Validate configuration
        parsed_config.finalize_and_validate()

        if verbose:
            parsed_config.print_config(logger)

        # Create analyzer
        with cli_utils.CONSOLE.status(
            "[green]Loading dataset...[/green]", spinner="dots"
        ):
            analyzer = DatasetAnalyzer(parsed_config)

        # Run analysis
        with cli_utils.CONSOLE.status(
            "[green]Running analysis...[/green]", spinner="dots"
        ):
            analyzer.analyze_dataset()

        # Display summary
        _display_analysis_summary(analyzer, verbose=verbose)

        # Export results
        if parsed_config.output_path:
            _export_results(analyzer, parsed_config.output_path, output_format)

        # Generate HTML report if requested via CLI flag or config
        should_generate_report = report or getattr(
            parsed_config, "generate_report", False
        )
        if should_generate_report:
            # CLI flag takes precedence over config for title
            effective_title = report_title or getattr(
                parsed_config, "report_title", None
            )
            _generate_html_report(analyzer, parsed_config.output_path, effective_title)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Configuration file not found: {e}")
        raise typer.Exit(code=1)

    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Invalid configuration: {e}")
        raise typer.Exit(code=1)

    except RuntimeError as e:
        logger.error(f"Analysis failed: {e}")
        cli_utils.CONSOLE.print(f"[red]Error:[/red] Analysis failed: {e}")
        raise typer.Exit(code=1)

    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        cli_utils.CONSOLE.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(code=1)


def _clean_metric_name(name: str, analyzer_id: str = "") -> str:
    """Clean up metric name for display.

    Args:
        name: The raw metric name (e.g., 'text_content_length_char_count')
        analyzer_id: The analyzer ID to strip from the name

    Returns:
        Cleaned metric name in Title Case (e.g., 'Char Count')
    """
    # Remove common prefixes
    for prefix in ["text_content_", "content_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]

    # Remove analyzer ID prefix if present (e.g., 'length_', 'format_')
    if analyzer_id and name.startswith(f"{analyzer_id}_"):
        name = name[len(analyzer_id) + 1 :]

    # Convert snake_case to Title Case
    return name.replace("_", " ").title()


def _get_analyzer_id_from_group(group_name: str) -> str:
    """Extract the base analyzer ID from a group name.

    Group names may be like 'length', 'format_code', 'format_format'.
    We want to extract 'length', 'format', 'format'.
    """
    # Known analyzer IDs
    known_analyzers = ["length", "format", "diversity", "embedding"]
    for analyzer in known_analyzers:
        if group_name == analyzer or group_name.startswith(f"{analyzer}_"):
            return analyzer
    return group_name


def _display_analysis_summary(
    analyzer: "DatasetAnalyzer", verbose: bool = False
) -> None:
    """Display analysis summary in a single consolidated table to the console.

    Args:
        analyzer: The DatasetAnalyzer with completed analysis
        verbose: If True, show all stats (std, min, max, median).
                 If False, show only mean and count.
    """
    from rich.panel import Panel
    from rich.text import Text

    summary = analyzer.analysis_summary

    # Dataset overview - compact header panel
    overview = summary.get("dataset_overview", {})
    if overview:
        conversations = overview.get("total_conversations", 0)
        coverage = overview.get("dataset_coverage_percentage", 0)
        messages = overview.get("total_messages", 0)
        analyzers_list = overview.get("analyzers_used", [])
        dataset_name = overview.get("dataset_name", "Unknown")

        overview_text = Text()
        overview_text.append(f"Dataset: ", style="dim")
        overview_text.append(f"{dataset_name}\n", style="cyan bold")
        overview_text.append(f"Conversations: ", style="dim")
        overview_text.append(f"{conversations}", style="green")
        overview_text.append(f" ({coverage:.0f}% coverage)  ", style="dim")
        overview_text.append(f"Messages: ", style="dim")
        overview_text.append(f"{messages}\n", style="green")
        overview_text.append(f"Analyzers: ", style="dim")
        overview_text.append(", ".join(analyzers_list) or "None", style="yellow")

        cli_utils.CONSOLE.print(
            Panel(overview_text, title="Dataset Overview", border_style="magenta")
        )
        cli_utils.CONSOLE.print()

    # Build a single consolidated metrics table
    msg_summary = summary.get("message_level_summary", {})
    turns_summary = summary.get("conversation_turns", {})

    if msg_summary or turns_summary:
        # Create consolidated table
        table = Table(
            title="Analysis Metrics",
            title_style="bold blue",
            box=ROUNDED,
            show_header=True,
            header_style="bold",
        )

        table.add_column("Analyzer", style="magenta", width=12)
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green", justify="right", width=10)
        table.add_column("Count", style="dim", justify="right", width=8)
        if verbose:
            table.add_column("Std", style="yellow", justify="right", width=10)
            table.add_column("Min", style="dim", justify="right", width=10)
            table.add_column("Max", style="dim", justify="right", width=10)
            table.add_column("Median", style="dim", justify="right", width=10)

        # Group metrics by base analyzer ID
        analyzer_metrics: dict[str, dict[str, dict]] = {}
        for group_name, metrics in msg_summary.items():
            base_analyzer = _get_analyzer_id_from_group(group_name)
            if base_analyzer not in analyzer_metrics:
                analyzer_metrics[base_analyzer] = {}
            analyzer_metrics[base_analyzer].update(metrics)

        # Sort analyzers for consistent display order
        analyzer_order = ["length", "diversity", "format", "quality", "embedding"]
        sorted_analyzers = sorted(
            analyzer_metrics.keys(),
            key=lambda x: analyzer_order.index(x) if x in analyzer_order else 999,
        )

        # Add rows grouped by analyzer
        for idx, analyzer_id in enumerate(sorted_analyzers):
            metrics = analyzer_metrics[analyzer_id]

            # Add section separator (except for first)
            if idx > 0:
                if verbose:
                    table.add_row("", "", "", "", "", "", "", "", style="dim")
                else:
                    table.add_row("", "", "", "", style="dim")

            first_metric = True
            for metric_name, stats in sorted(metrics.items()):
                if isinstance(stats, dict):
                    clean_name = _clean_metric_name(metric_name, analyzer_id)
                    mean_val = stats.get("mean", 0)
                    count_val = stats.get("count", 0)

                    # Format values
                    mean_str = _format_number(mean_val)
                    count_str = str(count_val)

                    # Show analyzer name only on first row of each group
                    analyzer_label = analyzer_id.title() if first_metric else ""

                    if verbose:
                        std_val = stats.get("std", 0)
                        min_val = stats.get("min", 0)
                        max_val = stats.get("max", 0)
                        median_val = stats.get("median", 0)
                        table.add_row(
                            analyzer_label,
                            clean_name,
                            mean_str,
                            count_str,
                            _format_number(std_val),
                            _format_number(min_val),
                            _format_number(max_val),
                            _format_number(median_val),
                        )
                    else:
                        table.add_row(analyzer_label, clean_name, mean_str, count_str)

                    first_metric = False

        # Add conversation turns as a separate section
        if turns_summary and isinstance(turns_summary, dict) and turns_summary.get(
            "count"
        ):
            # Add separator
            if sorted_analyzers:
                if verbose:
                    table.add_row("", "", "", "", "", "", "", "", style="dim")
                else:
                    table.add_row("", "", "", "", style="dim")

            mean_val = turns_summary.get("mean", 0)
            count_val = turns_summary.get("count", 0)

            if verbose:
                std_val = turns_summary.get("std", 0)
                min_val = turns_summary.get("min", 0)
                max_val = turns_summary.get("max", 0)
                median_val = turns_summary.get("median", 0)
                table.add_row(
                    "Turns",
                    "Messages Per Conversation",
                    _format_number(mean_val),
                    str(count_val),
                    _format_number(std_val),
                    _format_number(min_val),
                    _format_number(max_val),
                    _format_number(median_val),
                )
            else:
                table.add_row(
                    "Turns",
                    "Messages Per Conversation",
                    _format_number(mean_val),
                    str(count_val),
                )

        cli_utils.CONSOLE.print(table)
        cli_utils.CONSOLE.print()

    # Display recommendations
    _display_recommendations(summary)


def _format_number(value: Any) -> str:
    """Format a number for display.

    Args:
        value: The value to format.

    Returns:
        Formatted string representation.
    """
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"
    if isinstance(value, int):
        if abs(value) >= 1000:
            return f"{value:,}"
        return str(value)
    return str(value)


def _display_recommendations(summary: dict) -> None:
    """Display recommendations with color-coded severity.

    Args:
        summary: The analysis summary containing recommendations.
    """
    recommendations = summary.get("recommendations", [])
    if not recommendations:
        return

    # Severity icons and colors
    severity_styles = {
        "high": ("[red]●[/red]", "red"),
        "medium": ("[yellow]●[/yellow]", "yellow"),
        "low": ("[dim]●[/dim]", "dim"),
    }

    table = Table(
        title=f"Recommendations ({len(recommendations)})",
        title_style="bold cyan",
        box=ROUNDED,
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("", style="white", overflow="fold")

    for rec in recommendations[:5]:  # Show top 5 recommendations
        severity = rec.get("severity", "low")
        icon, style = severity_styles.get(severity, severity_styles["low"])
        title = rec.get("title", "")
        description = rec.get("description", "")

        # Format: ● SEVERITY: Title
        severity_label = severity.upper()
        table.add_row(f"{icon} [bold]{severity_label}:[/bold] {title}")
        if description:
            table.add_row(f"   [{style}]{description}[/{style}]")

    cli_utils.CONSOLE.print(table)

    if len(recommendations) > 5:
        cli_utils.CONSOLE.print(
            f"[dim]  ... and {len(recommendations) - 5} more recommendations "
            "(see analysis_summary.json)[/dim]"
        )
    cli_utils.CONSOLE.print()


def _export_results(
    analyzer: "DatasetAnalyzer",
    output_path: str,
    output_format: str,
) -> None:
    """Export analysis results to files."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export message-level results
    if analyzer.message_df is not None and not analyzer.message_df.empty:
        msg_path = output_dir / f"message_analysis.{output_format}"
        _save_dataframe(analyzer.message_df, msg_path, output_format)
        cli_utils.CONSOLE.print(f"[green]Saved message analysis to:[/green] {msg_path}")

    # Export conversation-level results
    if analyzer.conversation_df is not None and not analyzer.conversation_df.empty:
        conv_path = output_dir / f"conversation_analysis.{output_format}"
        _save_dataframe(analyzer.conversation_df, conv_path, output_format)
        cli_utils.CONSOLE.print(
            f"[green]Saved conversation analysis to:[/green] {conv_path}"
        )

    # Export summary as JSON
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(analyzer.analysis_summary, f, indent=2, default=str)
    cli_utils.CONSOLE.print(f"[green]Saved analysis summary to:[/green] {summary_path}")


def _save_dataframe(df: pd.DataFrame, path: Path, output_format: str) -> None:
    """Save a DataFrame to the specified format."""
    if output_format == "csv":
        df.to_csv(path, index=False)
    elif output_format == "json":
        df.to_json(path, orient="records", indent=2)
    elif output_format == "parquet":
        df.to_parquet(path, index=False)


def _generate_html_report(
    analyzer: "DatasetAnalyzer",
    output_path: Optional[str],
    title: Optional[str],
) -> None:
    """Generate an interactive HTML report with charts.

    Args:
        analyzer: The DatasetAnalyzer instance with completed analysis.
        output_path: Output directory for the report.
        title: Optional custom title for the report.
    """
    try:
        from oumi.core.analyze.report_generator import HTMLReportGenerator
    except ImportError:
        cli_utils.CONSOLE.print(
            "[yellow]Warning:[/yellow] HTML report generation requires additional "
            "dependencies. Install with: pip install 'oumi[analyze_advanced]'"
        )
        return

    try:
        generator = HTMLReportGenerator()

        # Determine output path
        if output_path:
            report_path = Path(output_path)
        else:
            report_path = Path()

        # Generate report
        with cli_utils.CONSOLE.status(
            "[green]Generating HTML report...[/green]", spinner="dots"
        ):
            output_file = generator.generate_report(
                analyzer=analyzer,
                output_path=report_path,
                title=title,
            )

        cli_utils.CONSOLE.print(
            f"[green]Generated HTML report:[/green] {output_file}"
        )

    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
        cli_utils.CONSOLE.print(
            f"[yellow]Warning:[/yellow] Failed to generate HTML report: {e}"
        )
