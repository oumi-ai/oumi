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
from oumi.utils.logging import logger


def synth(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for synthesis.",
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Synthesize a dataset.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for synthesis.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(cli_utils.resolve_and_fetch_config(config))

    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import synthesize as oumi_synthesize
        from oumi.core.configs.synthesis_config import SynthesisConfig
        # End imports

    # Load configuration
    parsed_config: SynthesisConfig = SynthesisConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    # Run synthesis
    with cli_utils.CONSOLE.status(
        "[green]Synthesizing dataset...[/green]", spinner="dots"
    ):
        results = oumi_synthesize(parsed_config)

    # Display results if no output path is specified
    if not parsed_config.output_path:
        table = Table(
            title="Synthesis Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        table.add_column("Sample", style="green")
        for i, result in enumerate(results[:5]):  # Show first 5 samples
            table.add_row(f"Sample {i + 1}: {repr(result)}")
        if len(results) > 5:
            table.add_row(f"... and {len(results) - 5} more samples")
        cli_utils.CONSOLE.print(table)
        cli_utils.CONSOLE.print(
            f"\n[green]Successfully synthesized {len(results)} samples[/green]"
        )
    else:
        cli_utils.CONSOLE.print(
            f"[green]Successfully synthesized {len(results)} samples and saved to "
            f"{parsed_config.output_path}[/green]"
        )
