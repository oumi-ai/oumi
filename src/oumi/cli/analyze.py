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

"""CLI for the typed analyzer framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_analyze_config
from oumi.utils.io_utils import load_jsonlines
from oumi.utils.logging import logger

if TYPE_CHECKING:
    from oumi.analyze.config import TypedAnalyzeConfig
    from oumi.core.types.conversation import Conversation

_VALID_OUTPUT_FORMATS = ("csv", "json", "parquet")

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.ANALYZE, "Available Analysis Configs", "analyze"
)

_OLD_CONFIG_FIELDS = {
    "dataset_source",
    "dataset_format",
    "processor_name",
    "processor_kwargs",
    "is_multimodal",
    "trust_remote_code",
}


def load_conversations_from_path(
    path: str | Path,
    sample_count: int | None = None,
) -> list[Conversation]:
    """Load conversations from a JSONL file."""
    from oumi.core.types.conversation import Conversation

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
    """Load conversations from a HuggingFace dataset in Oumi format.

    Each item must be parseable by ``Conversation.from_dict`` — i.e., already
    have the Oumi ``{"messages": [...]}`` shape. Items that don't parse are
    skipped with a warning.
    """
    from datasets import load_dataset

    from oumi.core.types.conversation import Conversation

    logger.info(f"Loading dataset: {dataset_name} (split={split}, subset={subset})")

    dataset = load_dataset(dataset_name, subset, split=split)
    conversations: list[Conversation] = []
    total = len(dataset)
    limit = sample_count if sample_count is not None else total

    for i in range(min(limit, total)):
        item = dataset[i]
        try:
            conversations.append(Conversation.from_dict(item))
        except Exception as e:
            logger.warning(f"Failed to parse conversation at index {i}: {e}")

    logger.info(f"Loaded {len(conversations)} conversations from {dataset_name}")
    return conversations


def run_typed_analysis(
    config: TypedAnalyzeConfig,
    conversations: list | None = None,
) -> dict[str, Any]:
    """Run the typed analysis pipeline."""
    from oumi.analyze import create_analyzer_from_config
    from oumi.analyze.pipeline import AnalysisPipeline
    from oumi.analyze.testing.engine import TestEngine

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

    analyzers = []
    for analyzer_config in config.analyzers:
        analyzer = create_analyzer_from_config(
            analyzer_config.type,
            analyzer_config.params,
        )
        if analyzer is not None:
            analyzer.analyzer_id = analyzer_config.display_name
            analyzers.append(analyzer)

    if not analyzers:
        raise ValueError("No valid analyzers configured")

    pipeline = AnalysisPipeline(
        analyzers=analyzers,
        cache_dir=config.output_path,
    )

    results = pipeline.run(conversations)

    test_summary = None
    if config.tests:
        test_engine = TestEngine(config.tests)
        test_summary = test_engine.run(results)

    df = pipeline.to_dataframe()

    return {
        "results": results,
        "test_summary": test_summary,
        "dataframe": df,
        "conversations": conversations,
    }


def save_results(
    output_path: str | Path,
    results: dict[str, Any],
    output_format: str = "csv",
) -> None:
    """Save analysis results to disk."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results["dataframe"]
    if output_format == "csv":
        df.to_csv(output_dir / "analysis.csv", index=False)
    elif output_format == "json":
        df.to_json(output_dir / "analysis.json", orient="records", indent=2)
    elif output_format == "parquet":
        df.to_parquet(output_dir / "analysis.parquet", index=False)

    logger.info(f"Saved analysis DataFrame to {output_dir}/analysis.{output_format}")

    test_summary = results.get("test_summary")
    if test_summary:
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(test_summary.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved test results to {output_dir}/test_results.json")

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
    """Print a summary of analysis results to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    console.print("\n[bold cyan]Analysis Summary[/bold cyan]\n")
    console.print(f"Conversations analyzed: {len(results['conversations'])}")
    console.print(f"Analyzers run: {', '.join(results['results'].keys())}")

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

    df = results["dataframe"]
    console.print(
        f"\n[bold]DataFrame Shape:[/bold] {df.shape[0]} rows x {df.shape[1]} columns"
    )


def run_from_config_file(
    config_path: str | Path,
    output_path: str | None = None,
    output_format: str = "csv",
) -> dict[str, Any]:
    """Run analysis from a YAML configuration file."""
    from oumi.analyze.config import TypedAnalyzeConfig

    config = TypedAnalyzeConfig.from_yaml(config_path)
    if output_path:
        config.output_path = output_path

    results = run_typed_analysis(config)
    if config.output_path:
        save_results(config.output_path, results, output_format)

    return results


def list_metrics(analyzer_name: str | None = None) -> None:
    """List available metrics for analyzers."""
    from oumi.analyze.discovery import print_analyzer_metrics

    print_analyzer_metrics(analyzer_name)


def _check_old_config_format(config_path: str) -> None:
    """Check if a config file uses the old AnalyzeConfig (v1) format and exit if so."""
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
            "  - 'analyzers' entries use 'id' and 'instance_id'\n"
            "  - Metrics are accessed as '{instance_id}.{field}' "
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
    list_metrics_flag: bool,
    verbose: bool,
    dataset_name: str | None = None,
    dataset_path: str | None = None,
    sample_count: int | None = None,
) -> None:
    """Run analysis using the typed analyzer system."""
    from oumi.analyze.config import TypedAnalyzeConfig

    try:
        if list_metrics_flag:
            cli_utils.CONSOLE.print("\n[bold cyan]Available Metrics[/bold cyan]\n")
            list_metrics()
            return

        _check_old_config_format(config)
        with cli_utils.CONSOLE.status(
            "[green]Loading configuration...[/green]", spinner="dots"
        ):
            typed_config = TypedAnalyzeConfig.from_yaml(config)

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
            analyzer_names = [a.display_name for a in typed_config.analyzers]
            cli_utils.CONSOLE.print(f"[dim]Analyzers: {analyzer_names}[/dim]")

        with cli_utils.CONSOLE.status(
            "[green]Running analysis...[/green]", spinner="dots"
        ):
            results = run_typed_analysis(typed_config)

        print_summary(results)

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
    dataset_name: Annotated[
        str | None,
        typer.Option(
            "--dataset-name",
            "--dataset_name",
            help="Dataset name to analyze (overrides config).",
            rich_help_panel="Data",
        ),
    ] = None,
    dataset_path: Annotated[
        str | None,
        typer.Option(
            "--dataset-path",
            "--dataset_path",
            help="Path to dataset file in JSONL format (overrides config).",
            rich_help_panel="Data",
        ),
    ] = None,
    sample_count: Annotated[
        int | None,
        typer.Option(
            "--sample-count",
            "--sample_count",
            help="Number of samples to analyze (overrides config).",
            rich_help_panel="Data",
        ),
    ] = None,
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
    """Analyze a dataset to compute metrics and statistics."""
    if ctx.invoked_subcommand is not None:
        return

    if list_metrics_flag:
        _run_typed_analysis_cli(
            config=config or "",
            output=output,
            output_format="csv",
            list_metrics_flag=True,
            verbose=verbose,
        )
        return

    if config is not None:
        config = str(
            cli_utils.resolve_and_fetch_config(
                try_get_config_name_for_alias(config, AliasType.ANALYZE),
            )
        )

    if config is None:
        cli_utils.CONSOLE.print(
            "[red]Error:[/red] Missing option '--config' / '-c'.\n"
            "Run 'oumi analyze --help' for usage."
        )
        raise typer.Exit(code=1)

    output_format = output_format.lower()
    if output_format not in _VALID_OUTPUT_FORMATS:
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Invalid output format '{output_format}'. "
            f"Supported formats: {', '.join(_VALID_OUTPUT_FORMATS)}"
        )
        raise typer.Exit(code=1)

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
