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

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_synth_config
from oumi.utils.logging import logger

_MAX_TABLE_ROWS = 1
_MAX_REPRESENTATION_LENGTH = 200
_TABLE_COLUMNS_TO_DISPLAY = 6

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.SYNTH, "Available Synthesis Configs", "synth"
)


def synth(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name for synthesis.",
            rich_help_panel="Options",
            autocompletion=complete_synth_config,
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available synthesis configs.",
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
    # Output options
    output_path: Annotated[
        str | None,
        typer.Option(
            "--output_path",
            help="Output path for synthesized dataset.",
            rich_help_panel="Output",
        ),
    ] = None,
):
    """Synthesize a dataset.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for synthesis.
        list_configs: List all available synthesis configs.
        level: The logging level for the specified command.
        output_path: Output path for synthesized dataset.
    """
    # Auto-collect overrides from dot-notation options
    option_overrides = cli_utils.collect_config_overrides(ctx)
    # Parse any additional extra args from command line
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    # Combine: explicit options take precedence (added last)
    all_overrides = extra_args + option_overrides

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.SYNTH),
        )
    )

    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi import synthesize as oumi_synthesize
        from oumi.core.configs.synthesis_config import SynthesisConfig
        # End imports

    # Load configuration
    parsed_config: SynthesisConfig = SynthesisConfig.from_yaml_and_arg_list(
        config, all_overrides, logger=logger
    )

    # Apply non-dot-notation overrides
    if output_path is not None:
        parsed_config.output_path = output_path

    parsed_config.finalize_and_validate()

    config_output_path = parsed_config.output_path
    if not config_output_path:
        cwd = Path.cwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_output_path = cwd / f"oumi_synth_results_{timestamp}.jsonl"
        if auto_output_path.exists():
            i = 1
            while auto_output_path.exists():
                auto_output_path = cwd / f"oumi_synth_results_{timestamp}_{i}.jsonl"
                i += 1
        parsed_config.output_path = auto_output_path.as_posix()

    # Run synthesis
    with cli_utils.CONSOLE.status(
        "[green]Synthesizing dataset...[/green]", spinner="dots"
    ):
        results = oumi_synthesize(parsed_config)

    if not results:
        cli_utils.CONSOLE.print(
            "No results found, please check your configuration and try again."
            "Report this issue at https://github.com/oumi-ai/oumi/issues"
        )
        return

    # Display results table
    table = Table(
        title="Synthesis Results",
        title_style="bold magenta",
        show_edge=False,
        show_lines=True,
    )
    columns = list(results[0].keys())
    column_count = len(columns)
    additional_column = (
        f"... and {column_count - _TABLE_COLUMNS_TO_DISPLAY + 1} more columns..."
    )
    if column_count > _TABLE_COLUMNS_TO_DISPLAY:
        # Keep first N-1 columns and add the additional column
        columns = columns[: _TABLE_COLUMNS_TO_DISPLAY - 1]
        columns.append(additional_column)
    for column in columns:
        table.add_column(column, style="green")
    for i, result in enumerate(results[:_MAX_TABLE_ROWS]):  # Show first 5 samples
        representations = []
        for column in columns:
            if column == additional_column:
                representation = "..."
            else:
                representation = repr(result[column])
                if len(representation) > _MAX_REPRESENTATION_LENGTH:
                    representation = representation[:_MAX_REPRESENTATION_LENGTH] + "..."
            representations.append(representation)
        table.add_row(*representations)
    cli_utils.CONSOLE.print(table)
    if len(results) > _MAX_TABLE_ROWS:
        cli_utils.CONSOLE.print(
            f"... and {len(results) - _MAX_TABLE_ROWS} more samples"
        )
    cli_utils.CONSOLE.print(
        f"\n[green]Successfully synthesized {len(results)} samples and saved to "
        f"{parsed_config.output_path}[/green]"
    )
    cli_utils.CONSOLE.print(
        f"\n\n[green]To train a model, run: oumi train -c "
        f"path/to/your/train/config.yaml\n\n"
        f"If you included a 'conversation' chat attribute in your config, update the "
        f"config to use your new dataset:\n"
        f"data:\n"
        f"  train:\n"
        f"    datasets:\n"
        f'      - dataset_name: "text_sft_jsonl"\n'
        f'        dataset_path: "{parsed_config.output_path}"\n'
        f"[/green]"
    )
