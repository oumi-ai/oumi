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
from rich.panel import Panel
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


def prompt(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for prompt optimization.",
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Optimize prompts for language models.

    This command uses state-of-the-art optimization algorithms like MIPRO and GEPA
    to automatically improve prompts, few-shot examples, and generation hyperparameters.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for prompt optimization.
        level: The logging level for the specified command.
        verbose: Whether to print verbose output.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.PROMPT),
        )
    )

    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import optimize_prompt as oumi_optimize_prompt
        from oumi.core.configs.prompt_config import PromptOptimizationConfig
        # End imports

    # Load configuration
    parsed_config: PromptOptimizationConfig = (
        PromptOptimizationConfig.from_yaml_and_arg_list(
            config, extra_args, logger=logger
        )
    )
    parsed_config.finalize_and_validate()

    if verbose:
        parsed_config.print_config(logger)

    # Run prompt optimization
    optimizer_name = parsed_config.optimization.optimizer.upper()
    message = (
        f"[cyan]Starting prompt optimization with {optimizer_name} optimizer[/cyan]"
    )
    cli_utils.CONSOLE.print(
        Panel(
            message,
            title="Prompt Optimization",
            border_style="green",
        )
    )

    with cli_utils.CONSOLE.status(
        "[green]Optimizing prompts...[/green]", spinner="dots"
    ):
        results = oumi_optimize_prompt(parsed_config)

    # Display results
    table = Table(
        title="Optimization Results",
        title_style="bold magenta",
        show_edge=False,
        show_lines=True,
    )

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Optimizer", parsed_config.optimization.optimizer.upper())
    table.add_row("Final Score", f"{results['final_score']:.4f}")
    table.add_row("Trials", str(results["num_trials"]))
    table.add_row("Output Directory", results["output_dir"])

    cli_utils.CONSOLE.print("\n")
    cli_utils.CONSOLE.print(table)

    output_dir = results["output_dir"]
    cli_utils.CONSOLE.print(
        f"\n[green]Optimization complete! Results saved to {output_dir}[/green]\n"
    )

    cli_utils.CONSOLE.print(
        Panel(
            "[yellow]Next Steps:[/yellow]\n\n"
            f"1. Review the optimized prompt in: "
            f"[cyan]{output_dir}/optimized_prompt.txt[/cyan]\n"
            f"2. Review few-shot examples in: "
            f"[cyan]{output_dir}/optimized_demos.jsonl[/cyan]\n"
            f"3. Review hyperparameters in: "
            f"[cyan]{output_dir}/optimized_hyperparameters.json[/cyan]\n"
            f"4. Use the optimized prompt in your inference config\n",
            title="Usage",
            border_style="yellow",
        )
    )
