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

from typing import Annotated, List, Optional

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


# Get list of common evaluation configurations for autocompletion
def get_config_examples() -> List[str]:
    """Get a list of example evaluation config paths for autocompletion."""
    return [
        "configs/recipes/llama3_1/evaluation/8b_eval.yaml", 
        "configs/recipes/phi3/evaluation/eval.yaml",
        "configs/recipes/qwq/evaluation/eval.yaml",
        "configs/examples/berry_bench/evaluation/eval.yaml"
    ]


# Common evaluation parameters for autocompletion
def get_evaluation_params() -> List[str]:
    """Get a list of common evaluation parameters for autocompletion."""
    return [
        "model.model_name",
        "inference_engine.engine_type",
        "backend",
        "output_path",
        "benchmark_names",
        "n_samples",
        "generation.temperature",
        "generation.max_tokens"
    ]


def evaluate(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, 
            help="Path to the configuration file for evaluation.",
            autocompletion=get_config_examples
        ),
    ],
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model", "-m",
            help="Model to evaluate"
        ),
    ] = None,
    engine: Annotated[
        Optional[str],
        typer.Option(
            "--engine", "-e", 
            help="Inference engine to use"
        ),
    ] = None,
    param: Annotated[
        List[str],
        typer.Option(
            "--param", "-p", 
            help="Override config parameters (e.g. backend=lm-harness)",
            autocompletion=get_evaluation_params
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Evaluate a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for evaluation.
        model: Model to evaluate (overrides config).
        engine: Inference engine to use (overrides config).
        param: Override config parameters.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Add model and engine to extra_args if provided
    if model:
        extra_args.append(f"model.model_name={model}")
    
    if engine:
        extra_args.append(f"inference_engine.engine_type={engine}")
    
    # Add any additional parameters passed via --param
    if param:
        extra_args.extend(param)

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
        # End imports

    # Load configuration
    parsed_config: EvaluationConfig = EvaluationConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    # Run evaluation
    with cli_utils.CONSOLE.status(
        "[green]Running evaluation...[/green]", spinner="dots"
    ):
        results = oumi_evaluate(parsed_config)
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
                if isinstance(value, (int, float)):
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
