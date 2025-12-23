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

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from oumi.cli import cli_utils
from oumi.cli.alias import AliasType
from oumi.cli.completions import complete_judge_config

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.JUDGE, "Available Judge Configs", "judge dataset"
)


def judge_dataset_file(
    ctx: typer.Context,
    # Main options
    judge_config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path or config name (e.g. safety, truthfulness).",
            rich_help_panel="Options",
            autocompletion=complete_judge_config,
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available judge configs.",
            callback=_list_configs_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    # I/O options
    input_file: Annotated[
        str,
        typer.Option(
            "--input",
            help="Path to the dataset input file (jsonl).",
            rich_help_panel="I/O",
        ),
    ] = "",
    output_file: Annotated[
        str | None,
        typer.Option(
            "--output",
            help="Path to the output file (jsonl).",
            rich_help_panel="I/O",
        ),
    ] = None,
    display_raw_output: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Display raw judge output.",
            rich_help_panel="Output",
        ),
    ] = False,
):
    """Judge a dataset."""
    # Delayed import
    from oumi import judge

    judge_file(
        ctx=ctx,
        judge_config=judge_config,
        input_file=input_file,
        output_file=output_file,
        display_raw_output=display_raw_output,
        judgment_fn=judge.judge_dataset_file,
    )


def judge_conversations_file(
    ctx: typer.Context,
    # Main options
    judge_config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path or config name (e.g. safety, truthfulness).",
            rich_help_panel="Options",
            autocompletion=complete_judge_config,
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available judge configs.",
            callback=_list_configs_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    # I/O options
    input_file: Annotated[
        str,
        typer.Option(
            "--input",
            help="Path to the conversations input file (jsonl).",
            rich_help_panel="I/O",
        ),
    ] = "",
    output_file: Annotated[
        str | None,
        typer.Option(
            "--output",
            help="Path to the output file (jsonl).",
            rich_help_panel="I/O",
        ),
    ] = None,
    display_raw_output: Annotated[
        bool,
        typer.Option(
            "--raw",
            help="Display raw judge output.",
            rich_help_panel="Output",
        ),
    ] = False,
):
    """Judge a list of conversations."""
    # Delayed import
    from oumi import judge

    judge_file(
        ctx=ctx,
        judge_config=judge_config,
        input_file=input_file,
        output_file=output_file,
        display_raw_output=display_raw_output,
        judgment_fn=judge.judge_conversations_file,
    )


def judge_file(
    ctx: typer.Context,
    judge_config: Annotated[
        str,
        typer.Option(
            "--config",
            help="Path to the judge config file",
        ),
    ],
    input_file: Annotated[
        str, typer.Option("--input", help="Path to the dataset input file (jsonl)")
    ],
    output_file: Annotated[
        str | None,
        typer.Option("--output", help="Path to the output file (jsonl)"),
    ] = None,
    display_raw_output: bool = False,
    *,
    judgment_fn: Callable[..., list[Any]],
):
    """Judge a dataset or list of conversations."""
    # Delayed import
    from oumi.cli.alias import try_get_config_name_for_alias
    from oumi.core.configs.judge_config import JudgeConfig

    # Load configs
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Resolve alias and fetch config
    judge_config_path = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(judge_config, AliasType.JUDGE),
        )
    )

    # Resolve judge config
    judge_config_obj = JudgeConfig.from_path(
        path=judge_config_path, extra_args=extra_args
    )

    # Ensure the dataset input file is provided and exists
    if not input_file:
        cli_utils.CONSOLE.print("[red]Error:[/red] --input is required.")
        raise typer.Exit(code=1)
    if not Path(input_file).exists():
        cli_utils.CONSOLE.print(
            f"[red]Error:[/red] Input file not found: '{input_file}'"
        )
        raise typer.Exit(code=1)

    # Judge the dataset
    judge_outputs = judgment_fn(
        judge_config=judge_config_obj,
        input_file=input_file,
        output_file=output_file,
    )

    # Calculate the overall score
    overall_score = 0.0
    for judge_output in judge_outputs:
        judgment_score = judge_output.field_scores.get("judgment", None)
        if judgment_score is not None:
            overall_score += judgment_score
        else:
            overall_score = None
            break

    # Display the overall score
    if overall_score is not None:
        overall_score = overall_score / len(judge_outputs)
        cli_utils.CONSOLE.print(
            f"\n[bold blue]Overall Score: {overall_score:.2%}[/bold blue]"
        )

    # Display the judge outputs if no output file was specified
    if not output_file:
        table = Table(
            title="Judge Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        table.add_column("Judgment", style="cyan")
        table.add_column("Judgment Score", style="green")
        table.add_column("Explanation", style="yellow")
        if display_raw_output:
            table.add_column("Raw Output", style="white")

        for judge_output in judge_outputs:
            judgment_value = str(judge_output.field_values.get("judgment", "N/A"))
            judgment_score = str(judge_output.field_scores.get("judgment", "N/A"))
            explanation_value = str(judge_output.field_values.get("explanation", "N/A"))

            if display_raw_output:
                table.add_row(
                    judgment_value,
                    judgment_score,
                    explanation_value,
                    judge_output.raw_output,
                )
            else:
                table.add_row(judgment_value, judgment_score, explanation_value)

        cli_utils.CONSOLE.print(table)
    else:
        typer.echo(f"Results saved to {output_file}")
