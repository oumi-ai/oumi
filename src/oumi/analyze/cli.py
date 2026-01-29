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

"""CLI utilities for the typed analyzer framework.

This module provides functions for running the typed analyzer pipeline
from the command line, including support for YAML configuration files.
"""

import json
import logging
from pathlib import Path
from typing import Any

from oumi.analyze.base import ConversationAnalyzer, DatasetAnalyzer, MessageAnalyzer
from oumi.analyze.config import TypedAnalyzeConfig
from oumi.analyze.pipeline import AnalysisPipeline
from oumi.analyze.testing.engine import TestEngine
from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)


# Registry of available analyzers
ANALYZER_REGISTRY: dict[str, type] = {}


def register_analyzer(name: str):
    """Decorator to register an analyzer class.

    Args:
        name: Name to register the analyzer under.

    Returns:
        Decorator function.
    """

    def decorator(cls):
        ANALYZER_REGISTRY[name] = cls
        return cls

    return decorator


# Register built-in analyzers
def _register_builtin_analyzers():
    """Register built-in analyzers in the registry."""
    from oumi.analyze.analyzers.deduplication import DeduplicationAnalyzer
    from oumi.analyze.analyzers.length import LengthAnalyzer
    from oumi.analyze.analyzers.llm_analyzer import (
        CoherenceAnalyzer,
        FactualityAnalyzer,
        InstructionFollowingAnalyzer,
        LLMAnalyzer,
        SafetyAnalyzer,
        UsefulnessAnalyzer,
    )
    from oumi.analyze.analyzers.quality import DataQualityAnalyzer
    from oumi.analyze.analyzers.turn_stats import TurnStatsAnalyzer

    # Non-LLM analyzers (fast, cheap)
    ANALYZER_REGISTRY["length"] = LengthAnalyzer
    ANALYZER_REGISTRY["LengthAnalyzer"] = LengthAnalyzer
    ANALYZER_REGISTRY["quality"] = DataQualityAnalyzer
    ANALYZER_REGISTRY["DataQualityAnalyzer"] = DataQualityAnalyzer
    ANALYZER_REGISTRY["turn_stats"] = TurnStatsAnalyzer
    ANALYZER_REGISTRY["TurnStatsAnalyzer"] = TurnStatsAnalyzer

    # Dataset-level analyzers
    ANALYZER_REGISTRY["deduplication"] = DeduplicationAnalyzer
    ANALYZER_REGISTRY["DeduplicationAnalyzer"] = DeduplicationAnalyzer

    # LLM-based analyzers
    ANALYZER_REGISTRY["llm"] = LLMAnalyzer
    ANALYZER_REGISTRY["LLMAnalyzer"] = LLMAnalyzer
    ANALYZER_REGISTRY["usefulness"] = UsefulnessAnalyzer
    ANALYZER_REGISTRY["UsefulnessAnalyzer"] = UsefulnessAnalyzer
    ANALYZER_REGISTRY["safety"] = SafetyAnalyzer
    ANALYZER_REGISTRY["SafetyAnalyzer"] = SafetyAnalyzer
    ANALYZER_REGISTRY["factuality"] = FactualityAnalyzer
    ANALYZER_REGISTRY["FactualityAnalyzer"] = FactualityAnalyzer
    ANALYZER_REGISTRY["coherence"] = CoherenceAnalyzer
    ANALYZER_REGISTRY["CoherenceAnalyzer"] = CoherenceAnalyzer
    ANALYZER_REGISTRY["instruction_following"] = InstructionFollowingAnalyzer
    ANALYZER_REGISTRY["InstructionFollowingAnalyzer"] = InstructionFollowingAnalyzer


# Call on module import
_register_builtin_analyzers()


def get_analyzer_class(name: str) -> type | None:
    """Get an analyzer class by name.

    Args:
        name: Name of the analyzer (e.g., "length" or "LengthAnalyzer").

    Returns:
        The analyzer class or None if not found.
    """
    return ANALYZER_REGISTRY.get(name)


def create_analyzer_from_config(
    analyzer_id: str,
    params: dict[str, Any],
) -> MessageAnalyzer | ConversationAnalyzer | DatasetAnalyzer | None:
    """Create an analyzer instance from configuration.

    Args:
        analyzer_id: Analyzer type identifier.
        params: Analyzer-specific parameters.

    Returns:
        Analyzer instance or None if not found.
    """
    analyzer_class = ANALYZER_REGISTRY.get(analyzer_id)
    if analyzer_class is None:
        logger.warning(f"Unknown analyzer: {analyzer_id}")
        return None

    try:
        return analyzer_class(**params)
    except Exception as e:
        logger.error(f"Failed to create analyzer {analyzer_id}: {e}")
        return None


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
    from oumi.utils.io_utils import load_jsonlines

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = load_jsonlines(str(path))
    conversations = []

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
    conversations = []
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


def _item_to_conversation(item: Any, index: int) -> Conversation | None:
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
    from oumi.core.types.conversation import Message, Role

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
        for key in ["prompt", "instruction", "original-instruction", "question", "input"]:
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


def run_typed_analysis(
    config: TypedAnalyzeConfig,
    conversations: list[Conversation] | None = None,
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
    from oumi.analyze.custom_metrics import create_custom_metric

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
            analyzer_config.id,
            analyzer_config.params,
        )
        if analyzer is not None:
            # Always set analyzer_id to match config (use instance_id or id)
            # This ensures results are keyed by the config ID, not class name
            analyzer.analyzer_id = analyzer_config.instance_id or analyzer_config.id
            analyzers.append(analyzer)

    # Create custom metrics
    for custom_metric_config in config.custom_metrics:
        try:
            custom_metric = create_custom_metric(custom_metric_config)
            analyzers.append(custom_metric)
            logger.info(f"Created custom metric: {custom_metric_config.id}")
        except Exception as e:
            logger.error(
                f"Failed to create custom metric {custom_metric_config.id}: {e}"
            )

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
    summary = {
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


def run_from_config_file(
    config_path: str | Path,
    output_path: str | None = None,
    output_format: str = "parquet",
) -> dict[str, Any]:
    """Run analysis from a YAML configuration file.

    This is the main entry point for CLI usage.

    Args:
        config_path: Path to YAML configuration file.
        output_path: Optional output path override.
        output_format: Output format for DataFrame.

    Returns:
        Analysis results.
    """
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
                f"  [red]High severity failures: {test_summary.high_severity_failures}[/red]"
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
                        f"  {col}: mean={values.mean():.2f}, min={values.min()}, max={values.max()}"
                    )
                else:
                    try:
                        console.print(f"  {col}: {values.value_counts().head(3).to_dict()}")
                    except TypeError:
                        # Skip unhashable types
                        continue
