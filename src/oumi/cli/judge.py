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
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.table import Table

from oumi.cli import cli_utils
from oumi.cli.alias import AliasType
from oumi.cli.completions import complete_judge_config

if TYPE_CHECKING:
    from oumi.judges.base_judge import JudgeOutput

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
    """Judge a dataset.

    Examples:
        oumi judge dataset -c judge.yaml --input data.jsonl

        oumi judge dataset -c safety --input outputs.jsonl  # Using alias

        oumi judge dataset -c config.yaml --input data.jsonl --output results.jsonl
    """
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
    """Judge a list of conversations.

    Examples:
        oumi judge conversations -c judge.yaml --input conversations.jsonl

        oumi judge conversations -c instruction-following --input chats.jsonl
    """
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
            "-c",
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

    if not judge_outputs:
        cli_utils.CONSOLE.print("[yellow]No judge outputs were produced.[/yellow]")
        if output_file:
            cli_utils.CONSOLE.print(f"[green]Results saved to {output_file}[/green]")
        return

    # Display the overall score, if every row produced one
    row_scores = [_get_row_score(judge_output) for judge_output in judge_outputs]
    if all(score is not None for score in row_scores):
        overall_score = sum(row_scores) / len(row_scores)  # type: ignore[arg-type]
        cli_utils.CONSOLE.print(
            f"\n[bold blue]Overall Score: {overall_score:.2%}[/bold blue]"
        )

    # Display the judge outputs if no output file was specified
    if not output_file:
        # One column per output field, so that multi-criteria (rubric) judges show
        # every criterion rather than just a "judgment" field they don't have.
        field_keys = [
            output_field.field_key
            for output_field in (judge_outputs[0].output_fields or [])
        ]

        # Explanations are long, and a rubric judge has one per criterion; several of
        # them side by side make the table unreadable. Keep a lone explanation (the
        # single-judgment case), and point elsewhere for the rest.
        explanation_keys = [key for key in field_keys if _is_explanation_key(key)]
        if len(explanation_keys) > 1:
            field_keys = [key for key in field_keys if key not in explanation_keys]
            cli_utils.CONSOLE.print(
                f"[dim]Omitting {len(explanation_keys)} explanation columns from the "
                "table. Use --output to write them to a file, or --raw to print the "
                "judge's full response.[/dim]"
            )

        table = Table(
            title="Judge Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        for field_key in field_keys:
            table.add_column(field_key, style="cyan")
        table.add_column("Score", style="green")
        if display_raw_output:
            table.add_column("Raw Output", style="white")

        for judge_output, row_score in zip(judge_outputs, row_scores):
            row = [
                str(judge_output.field_values.get(field_key, "N/A"))
                for field_key in field_keys
            ]
            row.append("N/A" if row_score is None else f"{row_score:.2f}")
            if display_raw_output:
                row.append(judge_output.raw_output)
            table.add_row(*row)

        cli_utils.CONSOLE.print(table)
    else:
        cli_utils.CONSOLE.print(f"[green]Results saved to {output_file}[/green]")


def _is_explanation_key(field_key: str) -> bool:
    """Return True if an output field holds an explanation rather than a judgment."""
    # Delayed imports
    from oumi.core.configs.params.rubric_judge_params import EXPLANATION_SUFFIX
    from oumi.judges.simple_judge import EXPLANATION_KEY

    return field_key == EXPLANATION_KEY or field_key.endswith(EXPLANATION_SUFFIX)


def _get_row_score(judge_output: "JudgeOutput") -> float | None:
    """Return a single score for one judged row.

    Multi-criteria judges set `aggregate_score` across their criteria; the other
    judges leave it None and expose a single "judgment" field score.
    """
    if judge_output.aggregate_score is not None:
        return judge_output.aggregate_score
    return judge_output.field_scores.get("judgment", None)
