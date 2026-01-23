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
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.box import ROUNDED
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger

# Valid output formats for analysis results
_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")

# Metric descriptions with range and interpretation guidance
# Format: (description, range, better_direction)
# better_direction: "higher", "lower", "context", or None (for informational metrics)
_METRIC_DESCRIPTIONS: dict[str, tuple[str, str, str | None]] = {
    # Length metrics
    "token_count": (
        "Number of tokens after tokenization",
        "0 to ∞",
        "context",
    ),
    # Diversity metrics
    "unique_words_ratio": (
        "Ratio of unique words to total words (vocabulary diversity)",
        "0.0 to 1.0",
        "higher",
    ),
    "words_ratio": (  # Alternate key for unique_words_ratio
        "Ratio of unique words to total words (vocabulary diversity)",
        "0.0 to 1.0",
        "higher",
    ),
    # Format metrics
    "has_markdown": (
        "Whether text contains markdown formatting (headers, lists, etc.)",
        "0 or 1",
        "context",
    ),
    "has_json": (
        "Whether text contains JSON content",
        "0 or 1",
        "context",
    ),
    "has_code_blocks": (
        "Whether text contains fenced code blocks (```)",
        "0 or 1",
        "context",
    ),
    "code_block_count": (
        "Number of fenced code blocks in text",
        "0 to ∞",
        "context",
    ),
    "block_count": (  # Alternate key for code_block_count
        "Number of fenced code blocks in text",
        "0 to ∞",
        "context",
    ),
    "has_urls": (
        "Whether text contains URLs",
        "0 or 1",
        "context",
    ),
    "has_emails": (
        "Whether text contains email addresses",
        "0 or 1",
        "context",
    ),
    "format_complexity_score": (
        "Weighted score of formatting elements (markdown, code, JSON, URLs)",
        "0.0 to 1.0",
        "context",
    ),
    "complexity_score": (  # Alternate key for format_complexity_score
        "Weighted score of formatting elements (markdown, code, JSON, URLs)",
        "0.0 to 1.0",
        "context",
    ),
    # Quality metrics
    "has_pii": (
        "Whether text contains personally identifiable information",
        "0 or 1",
        "lower",
    ),
    "pii_count": (
        "Count of PII instances detected (emails, phones, SSNs, etc.)",
        "0 to ∞",
        "lower",
    ),
    "has_encoding_issues": (
        "Whether text has encoding problems (mojibake, invalid UTF-8)",
        "0 or 1",
        "lower",
    ),
    "repetition_ratio": (
        "Ratio of repeated n-grams to total n-grams",
        "0.0 to 1.0",
        "lower",
    ),
    "has_high_repetition": (
        "Whether text has high repetitive content",
        "0 or 1",
        "lower",
    ),
    "language_confidence": (
        "Confidence score for detected language",
        "0.0 to 1.0",
        "higher",
    ),
    # Embedding metrics
    "has_semantic_duplicate": (
        "Whether sample has semantically similar duplicates",
        "0 or 1",
        "lower",
    ),
    "has_fuzzy_duplicate": (
        "Whether sample has near-duplicates (MinHash LSH)",
        "0 or 1",
        "lower",
    ),
    "fuzzy_jaccard_score": (
        "Jaccard similarity to nearest fuzzy duplicate",
        "0.0 to 1.0",
        "lower",
    ),
    "cluster": (
        "Cluster assignment ID for semantic grouping",
        "integer",
        None,
    ),
    "duplicate_group": (
        "Group ID for samples that are duplicates of each other",
        "integer",
        None,
    ),
    # Conversation metrics
    "messages_per_conversation": (
        "Number of messages (turns) in each conversation",
        "1 to ∞",
        "context",
    ),
}

if TYPE_CHECKING:
    import pandas as pd

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
    report: Annotated[
        bool,
        typer.Option(
            "--report",
            "-r",
            help="Generate an interactive HTML report with charts. Requires plotly.",
        ),
    ] = False,
    report_title: Annotated[
        str | None,
        typer.Option(
            "--report-title",
            help="Custom title for the HTML report.",
        ),
    ] = None,
    skip_llm: Annotated[
        bool,
        typer.Option(
            "--skip-llm",
            help="Skip analyzers that require LLM inference (e.g., llm_judge, evol_*).",
        ),
    ] = False,
    skip_remote_llm: Annotated[
        bool,
        typer.Option(
            "--skip-remote-llm",
            help="Skip analyzers that require remote LLM APIs (e.g., llm_judge, evol_*). "
            "Local model analyzers like IFD are still allowed.",
        ),
    ] = False,
    reanalyze: Annotated[
        bool,
        typer.Option(
            "--reanalyze",
            help="Force re-running the full analysis even if artifacts exist. "
            "By default, if artifacts exist, only tests are re-run.",
        ),
    ] = False,
    list_metrics: Annotated[
        bool,
        typer.Option(
            "--list-metrics",
            help="List all available metrics for test configurations. "
            "If artifacts exist, shows actual metrics. Otherwise, shows a preview "
            "based on configured analyzers (useful for writing tests before running analysis).",
        ),
    ] = False,
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Analyze a dataset to compute metrics and statistics.

    By default, if analysis artifacts already exist at the output path, only tests
    will be re-run (fast). Use --reanalyze to force a fresh analysis.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for analysis.
        output: Output directory for results. Overrides config output_path.
        output_format: Output format (csv, json, parquet). Case-insensitive.
        report: Whether to generate an interactive HTML report with charts.
        report_title: Custom title for the HTML report.
        skip_llm: Skip analyzers that require LLM inference.
        skip_remote_llm: Skip analyzers that require remote LLM APIs.
        reanalyze: Force re-running the full analysis even if artifacts exist.
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

        # Check if artifacts exist and can be reused
        output_dir = (
            Path(parsed_config.output_path) if parsed_config.output_path else None
        )
        artifacts_exist = False
        loaded_artifacts = None

        # Handle --list-metrics flag
        if list_metrics:
            # Try to load from artifacts first (more accurate)
            artifacts_have_schemas = False
            if output_dir and output_dir.exists():
                try:
                    from oumi.utils.analysis_utils import load_analyzer_artifacts

                    artifacts = load_analyzer_artifacts(output_dir, output_format)
                    # Check if artifacts have meaningful schema data
                    schemas = artifacts.get("schemas", {})
                    message_schema = schemas.get("message_schema", {})
                    conversation_schema = schemas.get("conversation_schema", {})

                    if message_schema or conversation_schema:
                        # Artifacts have schema data, use them
                        _display_available_metrics(output_dir, output_format)
                        artifacts_have_schemas = True
                except FileNotFoundError:
                    pass

            if not artifacts_have_schemas:
                # No artifacts with schemas - show preview based on configured analyzers
                if parsed_config.analyzers:
                    _display_metrics_from_config(parsed_config)
                else:
                    cli_utils.CONSOLE.print(
                        "[red]Error:[/red] No analyzers configured and no artifacts found."
                    )
                    cli_utils.CONSOLE.print(
                        "[yellow]Hint:[/yellow] Add analyzers to your config "
                        "or run analysis first to generate artifacts."
                    )
                    raise typer.Exit(code=1)
            return

        if output_dir and not reanalyze:
            from oumi.utils.analysis_utils import load_analyzer_artifacts

            # Check if artifacts exist
            artifacts_file = output_dir / "analysis_summary.json"
            if artifacts_file.exists():
                try:
                    with cli_utils.CONSOLE.status(
                        "[green]Loading existing artifacts...[/green]", spinner="dots"
                    ):
                        loaded_artifacts = load_analyzer_artifacts(
                            output_dir, output_format
                        )
                        artifacts_exist = True
                        cli_utils.CONSOLE.print(
                            "[cyan]Found existing analysis artifacts. "
                            "Re-running tests only (use --reanalyze to force full analysis).[/cyan]"
                        )
                except Exception as e:
                    logger.warning(f"Could not load existing artifacts: {e}")
                    artifacts_exist = False

        if artifacts_exist and loaded_artifacts is not None:
            # Use loaded artifacts - only run tests
            analyzer = _create_analyzer_from_artifacts(loaded_artifacts, parsed_config)

            # Run tests if configured
            if parsed_config.tests and analyzer.message_df is not None:
                with cli_utils.CONSOLE.status(
                    "[green]Running tests...[/green]", spinner="dots"
                ):
                    from oumi.core.analyze.test_engine import TestEngine

                    test_engine = TestEngine(parsed_config.tests)
                    # Provide empty DataFrame for conversation_df if None
                    import pandas as pd

                    conv_df = analyzer.conversation_df
                    if conv_df is None:
                        conv_df = pd.DataFrame()

                    test_summary = test_engine.run_tests(
                        message_df=analyzer.message_df,
                        conversation_df=conv_df,
                        summary=analyzer.analysis_summary,
                    )
                    # Update the analysis summary with new test results
                    if analyzer._analysis_summary is not None:
                        analyzer._analysis_summary["test_summary"] = (
                            test_summary.to_dict()
                        )
        else:
            # Run full analysis
            # Create analyzer
            with cli_utils.CONSOLE.status(
                "[green]Loading dataset...[/green]", spinner="dots"
            ):
                analyzer = DatasetAnalyzer(
                    parsed_config,
                    skip_llm_analyzers=skip_llm,
                    skip_remote_llm_analyzers=skip_remote_llm,
                )

            # Run analysis
            with cli_utils.CONSOLE.status(
                "[green]Running analysis...[/green]", spinner="dots"
            ):
                analyzer.analyze_dataset()

        # Display summary
        _display_analysis_summary(analyzer, verbose=verbose)

        # Export results (only if we ran full analysis, not just tests)
        if parsed_config.output_path and not artifacts_exist:
            from oumi.utils.analysis_utils import save_analyzer_artifacts

            save_analyzer_artifacts(analyzer, parsed_config.output_path, output_format)
            cli_utils.CONSOLE.print(
                f"[green]Analysis results saved to:[/green] {parsed_config.output_path}"
            )

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


def _create_analyzer_from_artifacts(
    artifacts: dict[str, Any],
    config: Any,
) -> "DatasetAnalyzer":
    """Create an analyzer-like object from loaded artifacts.

    This creates a minimal analyzer object that can be used for test execution
    and report generation without running the full analysis.

    Args:
        artifacts: Loaded artifacts from load_analyzer_artifacts.
        config: The AnalyzeConfig object.

    Returns:
        A DatasetAnalyzer-like object with loaded data.
    """
    import pandas as pd

    class ArtifactAnalyzer:
        """Minimal analyzer that wraps loaded artifacts."""

        def __init__(self, artifacts: dict[str, Any], config: Any):
            self._artifacts = artifacts
            self._config = config
            # Initialize internal state from artifacts
            self._message_df = artifacts.get("messages_df")
            self._conversation_df = artifacts.get("conversations_df")
            self._merged_df = artifacts.get("merged_df")
            self._analysis_summary = artifacts.get("analysis_summary", {})
            schemas = artifacts.get("schemas", {})
            self._merged_schema = schemas.get("merged_schema")
            self._message_schema = schemas.get("message_schema")
            self._conversation_schema = schemas.get("conversation_schema")

        @property
        def message_df(self) -> pd.DataFrame | None:
            return self._message_df

        @property
        def conversation_df(self) -> pd.DataFrame | None:
            return self._conversation_df

        @property
        def analysis_df(self) -> pd.DataFrame | None:
            return self._merged_df

        @property
        def analysis_summary(self) -> dict[str, Any]:
            return self._analysis_summary

        @property
        def config(self) -> Any:
            return self._config

    return ArtifactAnalyzer(artifacts, config)  # type: ignore


def _display_metrics_from_config(config: Any) -> None:
    """Display available metrics based on configured analyzers (preview mode).

    This allows users to see what metrics will be available before running
    the actual analysis. Uses analyzers' get_output_schema() method.

    Args:
        config: The AnalyzeConfig object with analyzer configurations.
    """
    from oumi.core.registry import REGISTRY

    cli_utils.CONSOLE.print(
        "\n[bold cyan]Available Metrics Preview[/bold cyan] "
        "[dim](based on configured analyzers)[/dim]\n"
    )
    cli_utils.CONSOLE.print(
        "[yellow]Note:[/yellow] This is a preview based on your config. "
        "Actual metrics may vary depending on dataset structure.\n"
    )

    # Collect schemas from all configured analyzers
    message_schema: dict = {}
    conversation_schema: dict = {}

    # Get the configured analyzers
    analyzers_config = config.analyzers if hasattr(config, "analyzers") else []

    if not analyzers_config:
        cli_utils.CONSOLE.print(
            "[yellow]No analyzers configured.[/yellow] "
            "Add analyzers to your config to see available metrics."
        )
        return

    for analyzer_config in analyzers_config:
        # Get analyzer ID
        analyzer_id = (
            analyzer_config.get("id")
            if isinstance(analyzer_config, dict)
            else getattr(analyzer_config, "id", None)
        )

        if not analyzer_id:
            continue

        # Try to get the analyzer class from registry
        try:
            analyzer_cls = REGISTRY.get_sample_analyzer(analyzer_id)
            if analyzer_cls is None:
                continue

            # Get params if any
            params = {}
            if isinstance(analyzer_config, dict):
                params = analyzer_config.get("params", {})
            elif hasattr(analyzer_config, "params"):
                params = analyzer_config.params or {}

            # Instantiate analyzer with params
            try:
                analyzer = analyzer_cls(**params)
                # Set analyzer_id attribute
                analyzer.analyzer_id = analyzer_id
            except Exception:
                # If instantiation fails, try without params
                try:
                    analyzer = analyzer_cls()
                    analyzer.analyzer_id = analyzer_id
                except Exception:
                    continue

            # Get output schema (preview mode: df=None, schema=None uses defaults)
            if hasattr(analyzer, "get_output_schema"):
                schema = analyzer.get_output_schema(
                    df=None,
                    schema=None,
                    analyzer_id=analyzer_id,
                )

                # Categorize metrics by scope
                for col_name, col_info in schema.items():
                    # Check if this is a conversation-level metric
                    if col_name.startswith("conversation__") or col_name.startswith(
                        "conversation_text_content"
                    ):
                        conversation_schema[col_name] = col_info
                    else:
                        message_schema[col_name] = col_info

        except Exception as e:
            logger.debug(f"Could not get schema for analyzer {analyzer_id}: {e}")
            continue

    # Display message-level metrics
    if message_schema:
        cli_utils.CONSOLE.print("[bold green]Message-Level Metrics[/bold green]")
        cli_utils.CONSOLE.print("(Use with scope: message or omit scope)\n")

        table = Table(box=ROUNDED, show_header=True, header_style="bold", expand=True)
        table.add_column(
            "Metric Name", style="cyan", no_wrap=True, min_width=50, overflow="fold"
        )
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Description", style="white", ratio=1, overflow="fold")

        for col in sorted(message_schema.keys()):
            info = message_schema[col]
            col_type = info.get("type", "unknown")
            # Handle ColumnType enum
            if hasattr(col_type, "value"):
                col_type = col_type.value
            description = info.get("description", "")
            table.add_row(col, str(col_type), description)

        cli_utils.CONSOLE.print(table)
        cli_utils.CONSOLE.print()

    # Display conversation-level metrics
    if conversation_schema:
        cli_utils.CONSOLE.print("[bold green]Conversation-Level Metrics[/bold green]")
        cli_utils.CONSOLE.print("(Use with scope: conversation)\n")

        table = Table(box=ROUNDED, show_header=True, header_style="bold", expand=True)
        table.add_column(
            "Metric Name", style="cyan", no_wrap=True, min_width=50, overflow="fold"
        )
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Description", style="white", ratio=1, overflow="fold")

        for col in sorted(conversation_schema.keys()):
            info = conversation_schema[col]
            col_type = info.get("type", "unknown")
            # Handle ColumnType enum
            if hasattr(col_type, "value"):
                col_type = col_type.value
            description = info.get("description", "")
            table.add_row(col, str(col_type), description)

        cli_utils.CONSOLE.print(table)
        cli_utils.CONSOLE.print()

    if not message_schema and not conversation_schema:
        cli_utils.CONSOLE.print(
            "[yellow]No metrics available.[/yellow] "
            "The configured analyzers may not have implemented get_output_schema() yet."
        )
        cli_utils.CONSOLE.print(
            "[dim]Run the analysis to see actual metrics from artifacts.[/dim]"
        )


def _display_available_metrics(output_dir: Path, output_format: str) -> None:
    """Display available metrics from saved artifacts.

    Reads the schema from saved artifacts and displays all available columns
    with their descriptions, organized by scope (message vs conversation).

    Args:
        output_dir: Directory containing saved artifacts.
        output_format: Format of the saved artifacts.
    """
    from oumi.utils.analysis_utils import load_analyzer_artifacts

    # Check if artifacts exist
    if not output_dir.exists():
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Output directory does not exist: {output_dir}"
        )
        cli_utils.CONSOLE.print(
            "[yellow]Hint:[/yellow] Run analysis first to generate artifacts."
        )
        raise typer.Exit(code=1)

    try:
        artifacts = load_analyzer_artifacts(output_dir, output_format)
    except FileNotFoundError:
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] No artifacts found at: {output_dir}"
        )
        cli_utils.CONSOLE.print(
            "[yellow]Hint:[/yellow] Run analysis first to generate artifacts."
        )
        raise typer.Exit(code=1)

    schemas = artifacts.get("schemas", {})
    message_schema = schemas.get("message_schema", {})
    conversation_schema = schemas.get("conversation_schema", {})

    cli_utils.CONSOLE.print("\n[bold cyan]Available Metrics for Tests[/bold cyan]\n")
    cli_utils.CONSOLE.print("Use these column names in your test configurations.\n")

    # Prefixes/names that indicate conversation-level metrics (even if in message schema)
    conversation_prefixes = (
        "conversation__",
        "conversation_text_content",  # Both the column itself and its derived metrics
    )
    # Exact column names that are conversation-level
    conversation_exact_cols = {"conversation_id", "conversation_index", "num_messages"}

    # Message-level metadata columns (should not appear in conversation metrics)
    message_metadata_cols = {
        "message_id",
        "message_index",
        "role",
        "text_content",
    }

    def is_conversation_level(col: str) -> bool:
        """Check if a column is conversation-level based on name."""
        if col in conversation_exact_cols:
            return True
        if col.startswith(conversation_prefixes):
            return True
        return False

    # Separate message-level metrics from conversation-level metrics
    # Some conversation-level metrics may be stored in message_schema
    true_message_metrics = {}
    conversation_metrics_from_message = {}

    for col, info in message_schema.items():
        if is_conversation_level(col):
            conversation_metrics_from_message[col] = info
        else:
            true_message_metrics[col] = info

    # Filter conversation schema to exclude message-level metadata
    filtered_conversation_schema = {
        col: info
        for col, info in conversation_schema.items()
        if col not in message_metadata_cols
    }

    # Merge conversation schemas
    all_conversation_metrics = {
        **filtered_conversation_schema,
        **conversation_metrics_from_message,
    }

    # Display message-level metrics
    if true_message_metrics:
        cli_utils.CONSOLE.print("[bold green]Message-Level Metrics[/bold green]")
        cli_utils.CONSOLE.print("(Use with scope: message or omit scope)\n")

        table = Table(box=ROUNDED, show_header=True, header_style="bold", expand=True)
        table.add_column(
            "Metric Name", style="cyan", no_wrap=True, min_width=50, overflow="fold"
        )
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Description", style="white", ratio=1, overflow="fold")

        # Sort by analyzer/category
        sorted_cols = sorted(true_message_metrics.keys())
        for col in sorted_cols:
            info = true_message_metrics[col]
            col_type = info.get("type", "unknown")
            description = info.get("description", "")
            table.add_row(col, str(col_type), description)

        cli_utils.CONSOLE.print(table)
        cli_utils.CONSOLE.print()

    # Display conversation-level metrics
    if all_conversation_metrics:
        cli_utils.CONSOLE.print("[bold green]Conversation-Level Metrics[/bold green]")
        cli_utils.CONSOLE.print("(Use with scope: conversation)\n")

        table = Table(box=ROUNDED, show_header=True, header_style="bold", expand=True)
        table.add_column(
            "Metric Name", style="cyan", no_wrap=True, min_width=50, overflow="fold"
        )
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Description", style="white", ratio=1, overflow="fold")

        # Sort columns: regular columns first (alphabetically), then conversation__ prefixed
        def sort_key(col: str) -> tuple[int, str]:
            # conversation__ prefixed columns go last (sort key 1), others first (sort key 0)
            if col.startswith("conversation__"):
                return (1, col)
            return (0, col)

        sorted_cols = sorted(all_conversation_metrics.keys(), key=sort_key)
        for col in sorted_cols:
            info = all_conversation_metrics[col]
            col_type = info.get("type", "unknown")
            description = info.get("description", "")
            table.add_row(col, str(col_type), description)

        cli_utils.CONSOLE.print(table)
        cli_utils.CONSOLE.print()

    # Show example test configurations
    cli_utils.CONSOLE.print(
        "[bold magenta]Example Test Configurations[/bold magenta]\n"
    )

    example_yaml = """# Threshold test (check if values exceed a limit)
tests:
  - id: max_tokens
    type: threshold
    metric: "text_content__length__token_count"
    operator: ">"
    value: 4096
    max_percentage: 5.0
    severity: medium
    title: "Messages exceeding token limit"

# Percentage test (check condition frequency)
  - id: no_pii
    type: percentage
    metric: "text_content__quality__has_pii"
    condition: "== True"
    max_percentage: 2.0
    severity: high
    title: "PII detected in messages"

# Conversation-level test
  - id: low_helpfulness
    type: threshold
    metric: "conversation_text_content__helpfulness__score"
    operator: "<="
    value: 4
    max_percentage: 0.0
    scope: conversation
    severity: high
    title: "Low helpfulness conversations"
"""
    cli_utils.CONSOLE.print(example_yaml)

    # Show related columns hint
    cli_utils.CONSOLE.print("[bold yellow]Tip: Related Columns[/bold yellow]")
    cli_utils.CONSOLE.print(
        "LLM judge metrics often have related columns. "
        "For example, if you test on:\n"
        "  - [cyan]helpfulness__score[/cyan]\n"
        "You can also access:\n"
        "  - [cyan]helpfulness__label[/cyan] (category)\n"
        "  - [cyan]helpfulness__reasoning[/cyan] (explanation)\n"
    )


def _get_metric_key(metric_name: str) -> str:
    """Extract the base metric key from a full metric name.

    Args:
        metric_name: Full metric name (e.g., 'text_content_length_token_count')

    Returns:
        Base metric key for lookup in _METRIC_DESCRIPTIONS
    """
    # Remove common prefixes
    key = metric_name
    for prefix in ["text_content_", "content_"]:
        if key.startswith(prefix):
            key = key[len(prefix) :]

    # Remove analyzer prefixes (e.g., 'length_', 'format_', 'diversity_')
    known_analyzers = ["length", "format", "diversity", "embedding", "quality"]
    for analyzer in known_analyzers:
        if key.startswith(f"{analyzer}_"):
            key = key[len(analyzer) + 1 :]
            break

    return key


def _get_metric_description(
    metric_name: str,
) -> tuple[str, str, str | None] | None:
    """Get the description, range, and better direction for a metric.

    Args:
        metric_name: The metric name to look up.

    Returns:
        Tuple of (description, range, better_direction) or None if not found.
    """
    metric_key = _get_metric_key(metric_name)
    return _METRIC_DESCRIPTIONS.get(metric_key)


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


def _display_metric_legend(metrics_shown: list[str]) -> None:
    """Display a legend explaining the metrics in the table.

    Args:
        metrics_shown: List of metric names that were displayed in the table.
    """
    from rich.panel import Panel
    from rich.text import Text

    # Collect unique metric descriptions for displayed metrics
    seen_keys: set[str] = set()
    legend_items: list[tuple[str, str, str, str | None]] = []

    for metric_name in metrics_shown:
        metric_key = _get_metric_key(metric_name)
        if metric_key in seen_keys:
            continue

        desc_info = _METRIC_DESCRIPTIONS.get(metric_key)
        if desc_info:
            seen_keys.add(metric_key)
            description, value_range, better = desc_info
            # Convert metric key to display name
            display_name = metric_key.replace("_", " ").title()
            legend_items.append((display_name, description, value_range, better))

    if not legend_items:
        return

    # Build the legend text
    legend_text = Text()

    for i, (name, description, value_range, better) in enumerate(legend_items):
        if i > 0:
            legend_text.append("\n")

        # Metric name
        legend_text.append(f"{name}: ", style="cyan bold")
        # Description
        legend_text.append(f"{description} ", style="white")
        # Range
        legend_text.append(f"[{value_range}]", style="dim")

        # Better direction indicator
        if better == "higher":
            legend_text.append(" (", style="dim")
            legend_text.append("higher is better", style="green")
            legend_text.append(")", style="dim")
        elif better == "lower":
            legend_text.append(" (", style="dim")
            legend_text.append("lower is better", style="yellow")
            legend_text.append(")", style="dim")
        elif better == "context":
            legend_text.append(" (", style="dim")
            legend_text.append("depends on use case", style="blue")
            legend_text.append(")", style="dim")

    cli_utils.CONSOLE.print(
        Panel(legend_text, title="Metric Descriptions", border_style="dim")
    )
    cli_utils.CONSOLE.print()


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
    metrics_shown: list[str] = []  # Track metrics for legend

    # Dataset overview - compact header panel
    overview = summary.get("dataset_overview", {})
    if overview:
        conversations = overview.get("total_conversations", 0)
        coverage = overview.get("dataset_coverage_percentage", 0)
        messages = overview.get("total_messages", 0)
        analyzers_list = overview.get("analyzers_used", [])
        dataset_name = overview.get("dataset_name", "Unknown")

        overview_text = Text()
        overview_text.append("Dataset: ", style="dim")
        overview_text.append(f"{dataset_name}\n", style="cyan bold")
        overview_text.append("Conversations: ", style="dim")
        overview_text.append(f"{conversations}", style="green")
        overview_text.append(f" ({coverage:.0f}% coverage)  ", style="dim")
        overview_text.append("Messages: ", style="dim")
        overview_text.append(f"{messages}\n", style="green")
        overview_text.append("Analyzers: ", style="dim")
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

                    # Track metric for legend display
                    metrics_shown.append(metric_name)

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
        if (
            turns_summary
            and isinstance(turns_summary, dict)
            and turns_summary.get("count")
        ):
            # Add separator
            if sorted_analyzers:
                if verbose:
                    table.add_row("", "", "", "", "", "", "", "", style="dim")
                else:
                    table.add_row("", "", "", "", style="dim")

            mean_val = turns_summary.get("mean", 0)
            count_val = turns_summary.get("count", 0)

            # Track metric for legend display
            metrics_shown.append("messages_per_conversation")

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

        # Display metric legend/descriptions
        _display_metric_legend(metrics_shown)

    # Display observations (key findings)
    _display_observations(summary)

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


def _display_observations(summary: dict) -> None:
    """Display observations (key findings) from the analysis.

    Args:
        summary: The analysis summary containing observations.
    """
    observations = summary.get("observations", [])
    if not observations:
        return

    # Category icons and colors
    category_styles = {
        "distribution": ("[blue]◆[/blue]", "blue"),
        "composition": ("[green]◆[/green]", "green"),
        "content": ("[cyan]◆[/cyan]", "cyan"),
        "quality": ("[yellow]◆[/yellow]", "yellow"),
        "structure": ("[magenta]◆[/magenta]", "magenta"),
    }

    table = Table(
        title=f"Observations ({len(observations)})",
        title_style="bold green",
        box=ROUNDED,
        show_header=False,
        padding=(0, 1),
    )
    table.add_column("", style="white", overflow="fold")

    for obs in observations[:8]:  # Show top 8 observations
        category = obs.get("category", "content")
        icon, style = category_styles.get(category, category_styles["content"])
        title = obs.get("title", "")
        description = obs.get("description", "")

        # Format: ◆ Title
        table.add_row(f"{icon} [bold]{title}[/bold]")
        if description:
            table.add_row(f"   [{style}]{description}[/{style}]")

    cli_utils.CONSOLE.print(table)

    if len(observations) > 8:
        cli_utils.CONSOLE.print(
            f"[dim]  ... and {len(observations) - 8} more observations "
            "(see analysis_summary.json)[/dim]"
        )
    cli_utils.CONSOLE.print()


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


def _save_dataframe(df: "pd.DataFrame", path: Path, output_format: str) -> None:
    """Save a DataFrame to the specified format."""
    if output_format == "csv":
        df.to_csv(path, index=False)
    elif output_format == "json":
        df.to_json(path, orient="records", indent=2)
    elif output_format == "parquet":
        df.to_parquet(path, index=False)


def _generate_html_report(
    analyzer: "DatasetAnalyzer",
    output_path: str | None,
    title: str | None,
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

        cli_utils.CONSOLE.print(f"[green]Generated HTML report:[/green] {output_file}")

    except Exception as e:
        logger.warning(f"Failed to generate HTML report: {e}")
        cli_utils.CONSOLE.print(
            f"[yellow]Warning:[/yellow] Failed to generate HTML report: {e}"
        )
