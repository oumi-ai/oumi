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
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_analyze_config
from oumi.utils.logging import logger

# Valid output formats for analysis results
_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.ANALYZE, "Available Analysis Configs", "analyze"
)

if TYPE_CHECKING:
    import pandas as pd

    from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer


def analyze(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name for analysis.",
            rich_help_panel="Options",
            autocompletion=complete_analyze_config,
        ),
    ],
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
            help="Dataset name to analyze.",
            rich_help_panel="Data",
        ),
    ] = None,
    dataset_path: Annotated[
        str | None,
        typer.Option(
            "--dataset_path",
            help="Path to custom dataset file (JSON or JSONL).",
            rich_help_panel="Data",
        ),
    ] = None,
    sample_count: Annotated[
        int | None,
        typer.Option(
            "--sample_count",
            help="Number of examples to sample from the dataset.",
            rich_help_panel="Data",
        ),
    ] = None,
    # Output options
    output: Annotated[
        str | None,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for analysis results.",
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

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for analysis.
        list_configs: List all available analysis configs.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        dataset_name: Dataset name to analyze.
        dataset_path: Path to custom dataset file.
        sample_count: Number of examples to sample.
        output: Output directory for results.
        output_format: Output format (csv, json, parquet).
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
        # Auto-collect overrides from dot-notation options
        option_overrides = cli_utils.collect_config_overrides(ctx)
        # Parse any additional extra args from command line
        extra_args = cli_utils.parse_extra_cli_args(ctx)
        # Combine: explicit options take precedence (added last)
        all_overrides = extra_args + option_overrides

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
            config, all_overrides, logger=logger
        )

        # Apply non-dot-notation overrides
        if dataset_name is not None:
            parsed_config.dataset_name = dataset_name
        if dataset_path is not None:
            parsed_config.dataset_path = dataset_path
        if sample_count is not None:
            parsed_config.sample_count = sample_count
        if output is not None:
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
        _display_analysis_summary(analyzer)

        # Export results
        if parsed_config.output_path:
            _export_results(analyzer, parsed_config.output_path, output_format)

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


def _display_quality_summary(quality_metrics: dict) -> None:
    """Display quality metrics summary in a formatted table."""
    table = Table(
        title="Quality Metrics Summary",
        title_style="bold red",
        show_lines=True,
    )
    table.add_column("Category", style="cyan", width=25)
    table.add_column("Metric", style="yellow", width=30)
    table.add_column("Value", style="green")

    # Key metrics to display for each analyzer
    key_metrics = {
        "duplicate_detection": [
            ("text_content_is_duplicate", "Duplicates Found", "true_percentage"),
        ],
        "empty_content": [
            ("text_content_has_content", "Has Valid Content", "true_percentage"),
        ],
        "format_validation": [
            ("format_is_valid", "Valid Format", "true_percentage"),
        ],
        "encoding": [
            ("text_content_has_encoding_issues", "Encoding Issues", "true_percentage"),
        ],
        "ngram_analysis": [
            ("text_content_unique_ngram_ratio", "Unique N-grams", "mean"),
            ("text_content_contains_overrepresented", "Overrepresented N-grams", "true_percentage"),
        ],
        "repetition": [
            ("text_content_word_repetition_ratio", "Word Repetition", "mean"),
            ("text_content_has_excessive_repetition", "Excessive Repetition", "true_percentage"),
        ],
        "vocabulary": [
            ("text_content_vocabulary_size", "Avg Vocabulary Size", "mean"),
            ("text_content_type_token_ratio", "Type-Token Ratio", "mean"),
        ],
        "request_type": [
            ("request_type", "Distribution", "distribution"),
            ("request_type_is_unknown", "Unknown Types", "true_percentage"),
        ],
        "readability": [
            ("text_content_flesch_reading_ease", "Flesch Reading Ease", "mean"),
            ("text_content_flesch_kincaid_grade", "Flesch-Kincaid Grade", "mean"),
        ],
    }

    category_names = {
        "duplicate_detection": "🔍 Duplicates",
        "empty_content": "📝 Content",
        "format_validation": "✅ Format",
        "encoding": "🔤 Encoding",
        "ngram_analysis": "📊 N-grams",
        "repetition": "🔁 Repetition",
        "vocabulary": "📚 Vocabulary",
        "request_type": "🏷️  Request Type",
        "readability": "📖 Readability",
    }

    for category_key, metrics in key_metrics.items():
        if category_key not in quality_metrics:
            continue

        category_data = quality_metrics[category_key]
        category_name = category_names.get(category_key, category_key)

        for metric_key, display_name, value_key in metrics:
            if metric_key not in category_data:
                continue

            metric_data = category_data[metric_key]

            # Format value based on type
            if value_key == "true_percentage":
                value = f"{metric_data.get('true_percentage', 0):.1f}%"
            elif value_key == "mean":
                mean_val = metric_data.get("mean", 0)
                # Format as percentage for ratios
                if "ratio" in metric_key or "percentage" in metric_key:
                    value = f"{mean_val * 100:.1f}%" if mean_val <= 1 else f"{mean_val:.2f}"
                else:
                    value = f"{mean_val:.2f}"
            elif value_key == "distribution":
                # Show top 3 categories for distributions
                dist = metric_data.get("distribution", {})
                if dist:
                    top_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    value = ", ".join([f"{k}({v})" for k, v in top_items])
                else:
                    value = "N/A"
            else:
                value = str(metric_data.get(value_key, "N/A"))

            table.add_row(category_name, display_name, value)
            # Only show category name once
            category_name = ""

    cli_utils.CONSOLE.print(table)


def _display_analysis_summary(analyzer: "DatasetAnalyzer") -> None:
    """Display analysis summary in formatted tables to the console."""
    summary = analyzer.analysis_summary

    # Dataset overview table
    overview = summary.get("dataset_overview", {})
    if overview:
        table = Table(
            title="Dataset Overview",
            title_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Dataset Name", str(overview.get("dataset_name", "N/A")))
        table.add_row(
            "Total Conversations", str(overview.get("total_conversations", "N/A"))
        )
        table.add_row(
            "Conversations Analyzed", str(overview.get("conversations_analyzed", "N/A"))
        )
        table.add_row(
            "Coverage",
            f"{overview.get('dataset_coverage_percentage', 0):.1f}%",
        )
        table.add_row("Total Messages", str(overview.get("total_messages", "N/A")))
        table.add_row(
            "Analyzers Used",
            ", ".join(overview.get("analyzers_used", [])) or "None",
        )
        cli_utils.CONSOLE.print(table)

    # Message-level summary
    msg_summary = summary.get("message_level_summary", {})
    if msg_summary:
        table = Table(
            title="Message-Level Metrics",
            title_style="bold blue",
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")
        table.add_column("Min", style="dim")
        table.add_column("Max", style="dim")
        table.add_column("Median", style="dim")

        for metric_name, stats in msg_summary.items():
            if isinstance(stats, dict):
                table.add_row(
                    metric_name,
                    f"{stats.get('mean', 'N/A'):.2f}"
                    if isinstance(stats.get("mean"), int | float)
                    else "N/A",
                    f"{stats.get('std', 'N/A'):.2f}"
                    if isinstance(stats.get("std"), int | float)
                    else "N/A",
                    str(stats.get("min", "N/A")),
                    str(stats.get("max", "N/A")),
                    f"{stats.get('median', 'N/A'):.2f}"
                    if isinstance(stats.get("median"), int | float)
                    else "N/A",
                )
        cli_utils.CONSOLE.print(table)

    # Conversation-level summary
    conv_summary = summary.get("conversation_level_summary", {})
    if conv_summary:
        table = Table(
            title="Conversation-Level Metrics",
            title_style="bold green",
            show_lines=True,
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green")
        table.add_column("Std", style="yellow")
        table.add_column("Min", style="dim")
        table.add_column("Max", style="dim")
        table.add_column("Median", style="dim")

        for metric_name, stats in conv_summary.items():
            if isinstance(stats, dict):
                table.add_row(
                    metric_name,
                    f"{stats.get('mean', 'N/A'):.2f}"
                    if isinstance(stats.get("mean"), int | float)
                    else "N/A",
                    f"{stats.get('std', 'N/A'):.2f}"
                    if isinstance(stats.get("std"), int | float)
                    else "N/A",
                    str(stats.get("min", "N/A")),
                    str(stats.get("max", "N/A")),
                    f"{stats.get('median', 'N/A'):.2f}"
                    if isinstance(stats.get("median"), int | float)
                    else "N/A",
                )
        cli_utils.CONSOLE.print(table)

    # Conversation turns summary
    turns_summary = summary.get("conversation_turns", {})
    if turns_summary and isinstance(turns_summary, dict) and turns_summary.get("count"):
        table = Table(
            title="Conversation Turns",
            title_style="bold yellow",
            show_lines=True,
        )
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Count", str(turns_summary.get("count", "N/A")))
        table.add_row(
            "Mean",
            f"{turns_summary.get('mean', 0):.2f}"
            if isinstance(turns_summary.get("mean"), int | float)
            else "N/A",
        )
        table.add_row(
            "Std",
            f"{turns_summary.get('std', 0):.2f}"
            if isinstance(turns_summary.get("std"), int | float)
            else "N/A",
        )
        table.add_row("Min", str(turns_summary.get("min", "N/A")))
        table.add_row("Max", str(turns_summary.get("max", "N/A")))
        table.add_row(
            "Median",
            f"{turns_summary.get('median', 0):.2f}"
            if isinstance(turns_summary.get("median"), int | float)
            else "N/A",
        )
        cli_utils.CONSOLE.print(table)

    # Quality metrics summary
    quality_metrics = summary.get("quality_metrics", {})
    if quality_metrics:
        _display_quality_summary(quality_metrics)


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


def _save_dataframe(df: "pd.DataFrame", path: Path, output_format: str) -> None:
    """Save a DataFrame to the specified format."""
    if output_format == "csv":
        df.to_csv(path, index=False)
    elif output_format == "json":
        df.to_json(path, orient="records", indent=2)
    elif output_format == "parquet":
        df.to_parquet(path, index=False)
