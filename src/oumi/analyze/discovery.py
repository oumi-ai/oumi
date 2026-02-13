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

"""Metric discovery utilities for the typed analyzer framework."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_analyzer_info(analyzer_class: type) -> dict[str, Any]:
    """Get detailed information about an analyzer's output metrics."""
    info: dict[str, Any] = {
        "name": analyzer_class.__name__,
        "metric_names": [],
        "metric_descriptions": {},
        "schema": {},
        "scope": analyzer_class.get_scope(),
        "config_schema": {},
    }

    try:
        info["metric_names"] = analyzer_class.get_metric_names()
        info["metric_descriptions"] = analyzer_class.get_metric_descriptions()
        info["schema"] = analyzer_class.get_result_schema()
    except TypeError:
        # Analyzer doesn't have a valid result type (e.g., abstract base class)
        logger.debug(
            f"Skipping metrics for {analyzer_class.__name__}: no valid result type"
        )

    try:
        info["config_schema"] = analyzer_class.get_config_schema()
    except Exception:
        logger.debug(f"Could not get config schema for {analyzer_class.__name__}")

    return info


def get_instance_metrics(
    analyzer_class: type,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Get available metrics for a specific analyzer instance.

    This attempts to instantiate the analyzer with the provided config to
    determine which metrics are available. If instantiation fails (e.g., due
    to mismatched config keys and __init__ parameters), falls back to returning
    all possible metrics from the class.

    Note:
        For this to filter metrics, the analyzer's config dict keys must match
        its __init__ parameters, and the analyzer must implement
        get_available_metric_names() instance method.

    Args:
        analyzer_class: The analyzer class.
        config: Optional configuration dictionary matching __init__ parameters.

    Returns:
        List of available metric names for this instance, or all class-level
        metrics if instantiation fails.
    """
    # Get all possible metrics from the class
    try:
        all_metrics = analyzer_class.get_metric_names()
    except (TypeError, AttributeError):
        return []

    # If the analyzer has get_available_metric_names() instance method,
    # we need to construct an instance to get filtered metrics
    if config is not None and hasattr(analyzer_class, "__init__"):
        try:
            # Try to create an instance with the config
            # This allows the analyzer to filter metrics based on its state
            instance = analyzer_class(**config)
            if hasattr(instance, "get_available_metric_names"):
                return instance.get_available_metric_names()
        except Exception:
            # If instantiation fails, fall back to class-level metrics
            pass

    return all_metrics


def list_available_metrics(
    include_duplicates: bool = False,
) -> dict[str, dict[str, Any]]:
    """List all available metrics from registered analyzers."""
    from oumi.core.registry import REGISTRY, RegistryType

    results = {}
    seen_classes = set()

    for name, analyzer_class in REGISTRY.get_all(RegistryType.SAMPLE_ANALYZER).items():
        class_name = analyzer_class.__name__
        if not include_duplicates and class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        info = get_analyzer_info(analyzer_class)
        # Include the registry key so callers can use it for lookups
        info["registry_id"] = name
        results[class_name if not include_duplicates else name] = info

    return results


def describe_analyzer(analyzer_class: type) -> str:
    """Get a human-readable description of an analyzer's metrics."""
    info = get_analyzer_info(analyzer_class)
    lines = [
        f"{info['name']} ({info['scope']} scope)",
        "",
        "Metrics:",
    ]

    metric_names = info.get("metric_names", [])
    metric_descriptions = info.get("metric_descriptions", {})
    schema = info.get("schema", {})
    properties = schema.get("properties", {})

    for metric_name in metric_names:
        prop_info = properties.get(metric_name, {})
        metric_type = _get_type_str(prop_info)
        description = metric_descriptions.get(metric_name, "")

        lines.append(f"  - {info['name']}.{metric_name} ({metric_type})")
        if description:
            lines.append(f"      {description}")

    return "\n".join(lines)


def get_metric_path(analyzer_name: str, metric_name: str) -> str:
    """Get the full metric path for use in test configurations.

    Args:
        analyzer_name: Name of the analyzer (e.g., "LengthAnalyzer").
        metric_name: Name of the metric field (e.g., "total_words").

    Returns:
        Full metric path (e.g., "LengthAnalyzer.total_words").
    """
    return f"{analyzer_name}.{metric_name}"


def print_analyzer_metrics(analyzer_name: str | None = None) -> None:
    """Pretty print available metrics for analyzers.

    Args:
        analyzer_name: Optional specific analyzer to show. If None, shows all.
    """
    metrics = list_available_metrics()

    # Filter to unique analyzers (avoid duplicates like "length" and "LengthAnalyzer")
    unique_metrics = {}
    seen_classes = set()
    for name, info in metrics.items():
        class_name = info.get("name", name)
        if class_name not in seen_classes:
            seen_classes.add(class_name)
            unique_metrics[class_name] = info

    _print_metrics_rich(unique_metrics, analyzer_name)


def _print_metrics_rich(
    metrics: dict[str, dict[str, Any]], analyzer_name: str | None
) -> None:
    """Print metrics using rich formatting."""
    from rich.console import Console

    console = Console()

    if analyzer_name:
        # Show specific analyzer
        if analyzer_name not in metrics:
            console.print(f"[red]Unknown analyzer: {analyzer_name}[/red]")
            console.print(f"Available: {', '.join(metrics.keys())}")
            return

        info = metrics[analyzer_name]
        _print_single_analyzer(console, analyzer_name, info)
    else:
        # Show all analyzers
        console.print("\n[bold cyan]Available Analyzers and Metrics[/bold cyan]\n")
        console.print(
            "Use these metric paths in your test configurations.\n"
            "Format: [cyan]AnalyzerName.metric_name[/cyan]\n"
        )

        for name, info in metrics.items():
            _print_single_analyzer(console, name, info)


def _print_single_analyzer(console: Any, name: str, info: dict[str, Any]) -> None:
    """Print metrics for a single analyzer."""
    from rich.table import Table

    scope_colors = {
        "message": "blue",
        "conversation": "green",
        "dataset": "magenta",
        "preference": "yellow",
    }
    scope = info.get("scope", "unknown")
    scope_color = scope_colors.get(scope, "white")

    console.print(f"[bold]{name}[/bold] [{scope_color}]({scope} scope)[/{scope_color}]")

    metric_names = info.get("metric_names", [])
    metric_descriptions = info.get("metric_descriptions", {})

    if not metric_names:
        console.print("  [dim]No metrics defined[/dim]\n")
        return

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Metric Path", style="cyan")
    table.add_column("Type", style="yellow", width=15)
    table.add_column("Description", style="white")

    schema = info.get("schema", {})
    properties = schema.get("properties", {})

    for metric_name in metric_names:
        path = f"{name}.{metric_name}"
        description = metric_descriptions.get(metric_name, "")

        # Get type from schema
        prop_info = properties.get(metric_name, {})
        metric_type = _get_type_str(prop_info)

        table.add_row(path, metric_type, description)

    console.print(table)
    console.print()


def _get_type_str(prop_info: dict) -> str:
    """Get a human-readable type string from JSON schema property info."""
    if not prop_info:
        return "any"

    # Handle anyOf (optional types)
    if "anyOf" in prop_info:
        types = []
        for option in prop_info["anyOf"]:
            if option.get("type") == "null":
                continue
            types.append(_get_type_str(option))
        return " | ".join(types) + " | None" if types else "any"

    prop_type = prop_info.get("type", "any")

    # Handle arrays
    if prop_type == "array":
        items = prop_info.get("items", {})
        item_type = items.get("type", "any")
        return f"list[{item_type}]"

    return prop_type


def generate_test_template(analyzer_name: str) -> str:
    """Generate a YAML test template for an analyzer's metrics.

    Args:
        analyzer_name: Name of the analyzer.

    Returns:
        YAML string with example test configurations.
    """
    metrics = list_available_metrics()

    if analyzer_name not in metrics:
        return f"# Unknown analyzer: {analyzer_name}"

    info = metrics[analyzer_name]
    metric_names = info.get("metric_names", [])
    metric_descriptions = info.get("metric_descriptions", {})
    schema = info.get("schema", {})
    properties = schema.get("properties", {})

    lines = [
        f"# Test templates for {analyzer_name}",
        f"# Scope: {info.get('scope', 'unknown')}",
        "",
        "tests:",
    ]

    for metric_name in metric_names[:5]:  # Show first 5 as examples
        description = metric_descriptions.get(metric_name, "")
        prop_info = properties.get(metric_name, {})
        metric_type = _get_type_str(prop_info)

        lines.append(f"  # {description}")
        lines.append(f"  - id: check_{metric_name}")

        if metric_type in ("bool", "boolean"):
            lines.append("    type: percentage")
            lines.append(f"    metric: {analyzer_name}.{metric_name}")
            lines.append('    condition: "== True"')
            lines.append("    max_percentage: 5.0")
        elif metric_type in ("int", "integer", "float", "number"):
            lines.append("    type: threshold")
            lines.append(f"    metric: {analyzer_name}.{metric_name}")
            lines.append('    operator: ">"')
            lines.append("    value: 1000  # Adjust as needed")
            lines.append("    max_percentage: 5.0")
        else:
            lines.append("    type: percentage")
            lines.append(f"    metric: {analyzer_name}.{metric_name}")
            lines.append('    condition: "!= None"')
            lines.append("    min_percentage: 95.0")

        lines.append("    severity: medium")
        lines.append("")

    return "\n".join(lines)
