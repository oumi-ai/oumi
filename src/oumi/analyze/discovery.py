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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from oumi.core.registry import REGISTRY, RegistryType

if TYPE_CHECKING:
    from rich.console import Console

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
    """Get available metrics, attempting to instantiate with config for filtering."""
    try:
        all_metrics = analyzer_class.get_metric_names()
    except (TypeError, AttributeError):
        return []

    if config is not None:
        try:
            if hasattr(analyzer_class, "from_config"):
                instance = analyzer_class.from_config(config)
            else:
                instance = analyzer_class(**config)
            return instance.get_available_metric_names()
        except Exception:
            pass

    return all_metrics


def list_available_metrics(
    include_duplicates: bool = False,
) -> dict[str, dict[str, Any]]:
    """List all available metrics from registered analyzers."""
    results = {}
    seen_classes = set()

    for name, analyzer_class in REGISTRY.get_all(RegistryType.SAMPLE_ANALYZER).items():
        class_name = analyzer_class.__name__
        if not include_duplicates and class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        info = get_analyzer_info(analyzer_class)
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


def print_analyzer_metrics(analyzer_name: str | None = None) -> None:
    """Pretty print available metrics for analyzers.

    Args:
        analyzer_name: Optional specific analyzer to show. If None, shows all.
    """
    metrics = list_available_metrics()
    _print_metrics_rich(metrics, analyzer_name)


def _print_metrics_rich(
    metrics: dict[str, dict[str, Any]], analyzer_name: str | None
) -> None:
    """Print metrics using rich formatting."""
    from rich.console import Console

    console = Console()

    if analyzer_name:
        if analyzer_name not in metrics:
            console.print(f"[red]Unknown analyzer: {analyzer_name}[/red]")
            console.print(f"Available: {', '.join(metrics.keys())}")
            return

        info = metrics[analyzer_name]
        _print_single_analyzer(console, analyzer_name, info)
    else:
        console.print("\n[bold cyan]Available Analyzers and Metrics[/bold cyan]\n")
        console.print(
            "Use these metric paths in your test configurations.\n"
            "Format: [cyan]AnalyzerName.metric_name[/cyan]\n"
        )

        for name, info in metrics.items():
            _print_single_analyzer(console, name, info)


def _print_single_analyzer(console: Console, name: str, info: dict[str, Any]) -> None:
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

        prop_info = properties.get(metric_name, {})
        metric_type = _get_type_str(prop_info)

        table.add_row(path, metric_type, description)

    console.print(table)
    console.print()


def _get_type_str(prop_info: dict) -> str:
    """Get a human-readable type string from JSON schema property info."""
    if not prop_info:
        return "any"

    if "anyOf" in prop_info:
        types = []
        for option in prop_info["anyOf"]:
            if option.get("type") == "null":
                continue
            types.append(_get_type_str(option))
        return " | ".join(types) + " | None" if types else "any"

    prop_type = prop_info.get("type", "any")

    if prop_type == "array":
        items = prop_info.get("items", {})
        item_type = items.get("type", "any")
        return f"list[{item_type}]"

    return prop_type
