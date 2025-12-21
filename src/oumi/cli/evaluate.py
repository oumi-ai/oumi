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
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.EVAL, "Available Evaluation Configs", "evaluate"
)


def evaluate(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name (e.g. llama3.1-8b).",
            rich_help_panel="Options",
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available evaluation configs.",
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
    # Model overrides
    model_name: Annotated[
        str | None,
        typer.Option(
            "--model.model_name",
            help="Model name or HuggingFace path.",
            rich_help_panel="Model",
        ),
    ] = None,
    # Generation overrides
    max_new_tokens: Annotated[
        int | None,
        typer.Option(
            "--generation.max_new_tokens",
            help="Maximum number of new tokens to generate.",
            rich_help_panel="Generation",
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option(
            "--generation.temperature",
            help="Sampling temperature (0.0 = deterministic).",
            rich_help_panel="Generation",
        ),
    ] = None,
    # Output overrides
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--output_dir",
            help="Output directory for evaluation results.",
            rich_help_panel="Output",
        ),
    ] = None,
):
    """Evaluate a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for evaluation.
        list_configs: List all available evaluation configs.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        model_name: Model name or HuggingFace path.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        output_dir: Output directory for evaluation results.
    """
    # Auto-collect overrides from dot-notation options (e.g., --model.model_name)
    option_overrides = cli_utils.collect_config_overrides(ctx)
    # Parse any additional extra args from command line
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    # Combine: explicit options take precedence (added last)
    all_overrides = extra_args + option_overrides

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.EVAL),
        )
    )
    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import evaluate as oumi_evaluate
        from oumi.core.configs import EvaluationConfig
        from oumi.core.distributed import is_world_process_zero
        # End imports

    # Load configuration
    parsed_config: EvaluationConfig = EvaluationConfig.from_yaml_and_arg_list(
        config, all_overrides, logger=logger
    )

    # Apply non-dot-notation overrides
    if output_dir is not None:
        parsed_config.output_dir = output_dir

    parsed_config.finalize_and_validate()

    if verbose:
        # Print configuration for verification
        parsed_config.print_config(logger)

    # Run evaluation
    with cli_utils.CONSOLE.status(
        "[green]Running evaluation...[/green]", spinner="dots"
    ):
        results = oumi_evaluate(parsed_config)

    if not is_world_process_zero():
        return
    # Make a best-effort attempt at parsing metrics.
    for task_result in results:
        table = Table(
            title="Evaluation Results",
            title_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Benchmark", style="cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Score", style="green")
        table.add_column("Std Error", style="dim")
        parsed_results = task_result.get("results", {})
        if not isinstance(parsed_results, dict):
            continue
        for task_name, metrics in parsed_results.items():
            # Get the benchmark display name from our benchmarks list

            if not isinstance(metrics, dict):
                # Skip if the metrics are not in a dict format
                table.add_row(
                    task_name,
                    "<unknown>",
                    "<unknown>",
                    "-",
                )
                continue
            benchmark_name: str = metrics.get("alias", task_name)
            # Process metrics
            for metric_name, value in metrics.items():
                metric_name: str = str(metric_name)
                if isinstance(value, int | float):
                    # Extract base metric name and type
                    base_name, *metric_type = metric_name.split(",")

                    # Skip if this is a stderr metric
                    # we'll handle it with the main metric
                    if base_name.endswith("_stderr"):
                        continue

                    # Get corresponding stderr if it exists
                    stderr_key = f"{base_name}_stderr,{metric_type[0] if metric_type else 'none'}"  # noqa E501
                    stderr_value = metrics.get(stderr_key)
                    stderr_display = (
                        f"±{stderr_value:.2%}" if stderr_value is not None else "-"
                    )

                    # Clean up metric name
                    clean_metric = base_name.replace("_", " ").title()

                    if isinstance(value, float):
                        if value > 1:
                            value_str = f"{value:.2f}"
                        else:
                            value_str = f"{value:.2%}"
                    else:
                        # Includes ints
                        value_str = str(value)
                    table.add_row(
                        benchmark_name,
                        clean_metric,
                        value_str,
                        stderr_display,
                    )
        cli_utils.CONSOLE.print(table)
