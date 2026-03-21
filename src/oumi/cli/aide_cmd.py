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

"""CLI command for AIDE agentic code optimization."""

from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_aide_config
from oumi.utils.logging import logger

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.AIDE, "Available AIDE Configs", "aide"
)


def aide(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name (e.g. smollm-135m).",
            rich_help_panel="Options",
            autocompletion=complete_aide_config,
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available AIDE configs.",
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
    # AIDE overrides
    steps: Annotated[
        int | None,
        typer.Option(
            "--aide.steps",
            help="Number of AIDE search steps.",
            rich_help_panel="AIDE",
        ),
    ] = None,
    target_metric: Annotated[
        str | None,
        typer.Option(
            "--aide.target_metric",
            help="Metric to optimize.",
            rich_help_panel="AIDE",
        ),
    ] = None,
    output_dir: Annotated[
        str | None,
        typer.Option(
            "--aide.output_dir",
            help="Output directory for results.",
            rich_help_panel="Output",
        ),
    ] = None,
):
    """Run AI-driven agentic code optimization.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for AIDE optimization.
        list_configs: List all available AIDE configs.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        steps: Number of AIDE search steps.
        target_metric: Metric to optimize.
        output_dir: Output directory for results.
    """
    # Auto-collect overrides from dot-notation options (e.g., --aide.steps)
    option_overrides = cli_utils.collect_config_overrides(ctx)
    # Parse any additional extra args from command line
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    # Combine: explicit options take precedence (added last)
    all_overrides = extra_args + option_overrides

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.AIDE),
        )
    )
    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi.aide import aide as oumi_aide
        from oumi.core.configs import AideConfig
        from oumi.utils.torch_utils import (
            device_cleanup,
            limit_per_process_memory,
        )

        # End imports

    cli_utils.configure_common_env_vars()

    parsed_config: AideConfig = AideConfig.from_yaml_and_arg_list(
        config, all_overrides, logger=logger
    )
    parsed_config.finalize_and_validate()

    limit_per_process_memory()
    device_cleanup()

    # Run AIDE optimization
    result = oumi_aide(parsed_config, verbose=verbose)

    device_cleanup()

    cli_utils.CONSOLE.print("\n[bold green]AIDE optimization complete.[/bold green]")
    cli_utils.CONSOLE.print(f"  Best metric: {result.best_metric}")
    cli_utils.CONSOLE.print(
        f"  Solutions: {result.good_solutions} good, "
        f"{result.buggy_solutions} buggy ({result.total_steps} total steps)"
    )
    cli_utils.CONSOLE.print(f"  Best solution: {result.best_solution_path}")
    cli_utils.CONSOLE.print(f"  Journal: {result.journal_path}")
