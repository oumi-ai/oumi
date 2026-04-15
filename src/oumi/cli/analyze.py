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

"""CLI for the typed analyzer framework.

This module provides the CLI entry point and orchestration logic for running
the typed analyzer pipeline, including YAML configuration, data loading,
analysis execution, and result output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

if TYPE_CHECKING:
    from oumi.analyze.config import TypedAnalyzeConfig
    from oumi.core.types.conversation import Conversation

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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_conversations_from_path(
    path: str | Path,
    sample_count: int | None = None,
) -> list[Conversation]:
    """Load conversations from a JSONL file.

    Args:
        path: Path to JSONL file.
        sample_count: Optional limit on number of conversations.

    Returns:
        List of Conversation objects.
    """
    from oumi.core.types.conversation import Conversation  # noqa: F811
    from oumi.utils.io_utils import load_jsonlines

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = load_jsonlines(str(path))
    conversations: list[Conversation] = []

    for i, item in enumerate(data):
        if sample_count is not None and i >= sample_count:
            break

        try:
            conv = Conversation.from_dict(item)
            conversations.append(conv)
        except Exception as e:
            logger.warning(f"Failed to parse conversation at index {i}: {e}")

    return conversations


def load_conversations_from_dataset(
    dataset_name: str,
    split: str = "train",
    subset: str | None = None,
    sample_count: int | None = None,
) -> list[Conversation]:
    """Load conversations from a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name (e.g., "OpenAssistant/oasst2").
        split: Dataset split to use (default: "train").
        subset: Optional dataset subset/config name.
        sample_count: Optional limit on number of conversations.

    Returns:
        List of Conversation objects.
    """
    from datasets import load_dataset

    logger.info(f"Loading dataset: {dataset_name} (split={split}, subset={subset})")

    # Load dataset directly from HuggingFace
    dataset = load_dataset(dataset_name, subset, split=split)

    # Convert to conversations
    conversations: list[Conversation] = []
    total = len(dataset)
    limit = sample_count if sample_count else total

    for i in range(min(limit, total)):
        try:
            item = dataset[i]
            # Handle different dataset formats
            conv = _item_to_conversation(item, i)
            if conv is not None:
                conversations.append(conv)
        except Exception as e:
            logger.warning(f"Failed to convert item at index {i}: {e}")

    logger.info(f"Loaded {len(conversations)} conversations from {dataset_name}")
    return conversations


def _item_to_conversation(item: Any, index: int) -> Any:
    """Convert a dataset item to a Conversation object.

    Handles multiple dataset formats:
    - Oumi native format (has 'messages' key)
    - Direct Conversation objects
    - Dictionary with conversation data

    Args:
        item: Dataset item.
        index: Item index for ID generation.

    Returns:
        Conversation object or None if conversion fails.
    """
    from oumi.core.types.conversation import Conversation, Message, Role

    # Already a Conversation
    if isinstance(item, Conversation):
        return item

    # Dictionary format
    if isinstance(item, dict):
        # Check for 'messages' key (Oumi format)
        if "messages" in item:
            try:
                return Conversation.from_dict(item)
            except Exception:
                pass

        # Check for 'conversation' key
        if "conversation" in item:
            conv_data = item["conversation"]
            if isinstance(conv_data, list):
                messages = []
                for msg in conv_data:
                    if isinstance(msg, dict):
                        role_str = msg.get("role", msg.get("from", "user"))
                        content = msg.get(
                            "content", msg.get("text", msg.get("value", ""))
                        )
                        try:
                            role = Role(role_str.lower())
                        except ValueError:
                            role = (
                                Role.USER
                                if role_str.lower() in ("human", "user")
                                else Role.ASSISTANT
                            )
                        messages.append(Message(role=role, content=content))
                return Conversation(messages=messages, metadata={"source_index": index})

        # Check for prompt/response format (multiple common variations)
        prompt = None
        response = None
        context = None

        # Try different field names for prompt/instruction
        for key in [
            "prompt",
            "instruction",
            "original-instruction",
            "question",
            "input",
        ]:
            if key in item and item[key]:
                prompt = item[key]
                break

        # Try different field names for response/output
        for key in [
            "response",
            "output",
            "completion",
            "original-response",
            "answer",
            "target",
        ]:
            if key in item and item[key]:
                response = item[key]
                break

        # Try different field names for context (optional)
        for key in ["context", "original-context", "input_context"]:
            if key in item and item[key]:
                context = item[key]
                break

        if prompt:
            # Combine context with prompt if available
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            else:
                full_prompt = str(prompt)

            messages = [
                Message(role=Role.USER, content=full_prompt),
            ]
            if response:
                messages.append(Message(role=Role.ASSISTANT, content=str(response)))
            return Conversation(messages=messages, metadata={"source_index": index})

        # Try direct conversion
        try:
            return Conversation.from_dict(item)
        except Exception:
            pass

    logger.debug(f"Could not convert item at index {index} to Conversation")
    return None


# ---------------------------------------------------------------------------
# Analyzer Creation
# ---------------------------------------------------------------------------


def get_analyzer_class(name: str) -> Any:
    """Get an analyzer class by name from the core registry.

    Args:
        name: Name of the analyzer (e.g., "length", "quality").

    Returns:
        The analyzer class or None if not found.
    """
    from oumi.core.registry import REGISTRY

    return REGISTRY.get_sample_analyzer(name)


def create_analyzer_from_config(
    analyzer_id: str,
    params: dict[str, Any],
) -> Any:
    """Create an analyzer instance from configuration.

    Args:
        analyzer_id: Analyzer type identifier.
        params: Analyzer-specific parameters.

    Returns:
        Analyzer instance or None if not found.
    """
    analyzer_class = get_analyzer_class(analyzer_id)
    if analyzer_class is None:
        logger.warning(f"Unknown analyzer: {analyzer_id}")
        return None

    try:
        # Prefer from_config() if available for better config handling
        if hasattr(analyzer_class, "from_config") and callable(
            getattr(analyzer_class, "from_config")
        ):
            return analyzer_class.from_config(params)
        else:
            return analyzer_class(**params)
    except Exception as e:
        logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Analysis Execution
# ---------------------------------------------------------------------------


def run_typed_analysis(
    config: TypedAnalyzeConfig,  # noqa: F821
    conversations: list | None = None,
) -> dict[str, Any]:
    """Run the typed analysis pipeline.

    Args:
        config: Analysis configuration.
        conversations: Optional pre-loaded conversations. If None, loads from config.

    Returns:
        Dictionary containing:
        - results: Analyzer results
        - test_summary: Test results (if tests configured)
        - dataframe: Results as DataFrame
    """
    from oumi.analyze.pipeline import AnalysisPipeline
    from oumi.analyze.testing.engine import TestEngine

    # Load conversations if not provided
    if conversations is None:
        if config.dataset_path:
            conversations = load_conversations_from_path(
                config.dataset_path,
                config.sample_count,
            )
        elif config.dataset_name:
            conversations = load_conversations_from_dataset(
                config.dataset_name,
                config.split,
                config.subset,
                config.sample_count,
            )
        else:
            raise ValueError(
                "Either conversations, dataset_path, or dataset_name must be provided"
            )

    logger.info(f"Loaded {len(conversations)} conversations for analysis")

    # Create analyzers
    analyzers = []
    for analyzer_config in config.analyzers:
        analyzer = create_analyzer_from_config(
            analyzer_config.type,
            analyzer_config.params,
        )
        if analyzer is not None:
            # Set analyzer_id to display_name so pipeline results and metric
            # paths match (e.g. "Length.total_tokens").
            analyzer.analyzer_id = analyzer_config.display_name
            analyzers.append(analyzer)

    if not analyzers:
        raise ValueError("No valid analyzers configured")

    # Create and run pipeline
    pipeline = AnalysisPipeline(
        analyzers=analyzers,
        cache_dir=config.output_path if config.output_path != "." else None,
    )

    results = pipeline.run(conversations)

    # Run tests if configured
    test_summary = None
    if config.tests:
        test_configs = config.get_test_configs()
        test_engine = TestEngine(test_configs)
        test_summary = test_engine.run(results)

    # Convert to DataFrame
    df = pipeline.to_dataframe()

    return {
        "results": results,
        "test_summary": test_summary,
        "dataframe": df,
        "conversations": conversations,
    }


# ---------------------------------------------------------------------------
# Result Output
# ---------------------------------------------------------------------------


def save_results(
    output_path: str | Path,
    results: dict[str, Any],
    output_format: str = "parquet",
) -> None:
    """Save analysis results to disk.

    Args:
        output_path: Output directory path.
        results: Analysis results from run_typed_analysis.
        output_format: Format for DataFrame ("csv", "json", "parquet").
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    df = results["dataframe"]
    if output_format == "csv":
        df.to_csv(output_dir / "analysis.csv", index=False)
    elif output_format == "json":
        df.to_json(output_dir / "analysis.json", orient="records", indent=2)
    elif output_format == "parquet":
        df.to_parquet(output_dir / "analysis.parquet", index=False)

    logger.info(f"Saved analysis DataFrame to {output_dir}/analysis.{output_format}")

    # Save test results if available
    test_summary = results.get("test_summary")
    if test_summary:
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_summary.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved test results to {output_dir}/test_results.json")

    # Save summary
    summary: dict[str, Any] = {
        "total_conversations": len(results["conversations"]),
        "analyzers_run": list(results["results"].keys()),
    }
    if test_summary:
        summary["tests"] = {
            "total": test_summary.total_tests,
            "passed": test_summary.passed_tests,
            "failed": test_summary.failed_tests,
            "pass_rate": test_summary.pass_rate,
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved analysis summary to {output_dir}/summary.json")


def print_summary(results: dict[str, Any]) -> None:
    """Print a summary of analysis results to console.

    Args:
        results: Analysis results from run_typed_analysis.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Print overview
    console.print("\n[bold cyan]Analysis Summary[/bold cyan]\n")
    console.print(f"Conversations analyzed: {len(results['conversations'])}")
    console.print(f"Analyzers run: {', '.join(results['results'].keys())}")

    # Print test results if available
    test_summary = results.get("test_summary")
    if test_summary:
        console.print("\n[bold]Test Results:[/bold]")
        console.print(
            f"  Passed: {test_summary.passed_tests}/{test_summary.total_tests} "
            f"({test_summary.pass_rate}%)"
        )
        if test_summary.high_severity_failures > 0:
            console.print(
                "  [red]High severity failures: "
                f"{test_summary.high_severity_failures}[/red]"
            )

        # Show failed tests
        failed = test_summary.get_failed_results()
        if failed:
            console.print("\n[bold red]Failed Tests:[/bold red]")
            table = Table(show_header=True)
            table.add_column("Test ID")
            table.add_column("Severity")
            table.add_column("Affected")
            table.add_column("Title")

            for result in failed[:10]:  # Show first 10
                table.add_row(
                    result.test_id,
                    result.severity.value,
                    f"{result.affected_percentage}%",
                    result.title,
                )
            console.print(table)

    # Print sample metrics
    df = results["dataframe"]
    console.print(
        f"\n[bold]DataFrame Shape:[/bold] {df.shape[0]} rows x {df.shape[1]} columns"
    )

    # Show first few columns
    metric_cols = [c for c in df.columns if "__" in c][:5]
    if metric_cols:
        console.print("\n[bold]Sample Metrics:[/bold]")
        for col in metric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                # Skip list-type columns (e.g., message_token_counts)
                if values.apply(lambda x: isinstance(x, list)).any():
                    continue
                if values.dtype in ["int64", "float64"]:
                    console.print(
                        f"  {col}: mean={values.mean():.2f}, "
                        f"min={values.min()}, max={values.max()}"
                    )
                else:
                    try:
                        console.print(
                            f"  {col}: {values.value_counts().head(3).to_dict()}"
                        )
                    except TypeError:
                        # Skip unhashable types
                        continue


# ---------------------------------------------------------------------------
# Discovery / Helpers
# ---------------------------------------------------------------------------


def run_from_config_file(
    config_path: str | Path,
    output_path: str | None = None,
    output_format: str = "parquet",
) -> dict[str, Any]:
    """Run analysis from a YAML configuration file.

    This is a convenience entry point for programmatic usage.

    Args:
        config_path: Path to YAML configuration file.
        output_path: Optional output path override.
        output_format: Output format for DataFrame.

    Returns:
        Analysis results.
    """
    from oumi.analyze.config import TypedAnalyzeConfig

    # Load config
    config = TypedAnalyzeConfig.from_yaml(config_path)

    # Override output path if provided
    if output_path:
        config.output_path = output_path

    # Run analysis
    results = run_typed_analysis(config)

    # Save results
    if config.output_path:
        save_results(config.output_path, results, output_format)

    return results


def list_metrics(analyzer_name: str | None = None) -> None:
    """List available metrics for analyzers.

    This helps users discover what metrics are available so they can
    write test configurations before running analysis.

    Args:
        analyzer_name: Optional specific analyzer to show. If None, shows all.

    Example:
        >>> from oumi.analyze import list_metrics
        >>> list_metrics()  # Shows all analyzers
        >>> list_metrics("LengthAnalyzer")  # Shows specific analyzer
    """
    from oumi.analyze.discovery import print_analyzer_metrics

    print_analyzer_metrics(analyzer_name)


def generate_tests(analyzer_name: str) -> str:
    """Generate example test configurations for an analyzer.

    Args:
        analyzer_name: Name of the analyzer.

    Returns:
        YAML string with example test configurations.

    Example:
        >>> from oumi.analyze import generate_tests
        >>> yaml_config = generate_tests("LengthAnalyzer")
        >>> print(yaml_config)
    """
    from oumi.analyze.discovery import generate_test_template

    return generate_test_template(analyzer_name)


# ---------------------------------------------------------------------------
# Old Config Detection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI Runner
# ---------------------------------------------------------------------------


def _run_typed_analysis_cli(
    config: str,
    output: str | None,
    output_format: str,
    list_metrics_flag: bool,
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
        list_metrics_flag: Whether to just list available metrics.
        verbose: Enable verbose output.
        dataset_name: Override dataset name from CLI.
        dataset_path: Override dataset path from CLI.
        sample_count: Override sample count from CLI.
    """
    from oumi.analyze.config import TypedAnalyzeConfig

    try:
        # Handle --list-metrics
        if list_metrics_flag:
            cli_utils.CONSOLE.print("\n[bold cyan]Available Metrics[/bold cyan]\n")
            list_metrics()
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


# ---------------------------------------------------------------------------
# Typer Command
# ---------------------------------------------------------------------------


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
    list_metrics_flag: Annotated[
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
        list_metrics_flag: List available metrics without running analysis.
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
    if list_metrics_flag:
        _run_typed_analysis_cli(
            config=config or "",
            output=output,
            output_format="csv",
            list_metrics_flag=True,
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
        list_metrics_flag=False,
        verbose=verbose,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        sample_count=sample_count,
    )
