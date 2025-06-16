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

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional, Union

import typer
from rich.table import Table

from oumi.cli import cli_utils

if TYPE_CHECKING:
    from oumi.core.configs import InferenceConfig
    from oumi.core.configs.judge_config_v2 import JudgeConfig


def _resolve_judge_config(
    config: str, extra_args: list[str]
) -> Union["JudgeConfig", str]:
    """Resolve judge config from either a file path or built-in judge name.

    Args:
        config: Either a file path to a YAML config or a built-in judge name
        extra_args: Additional CLI arguments for config override

    Returns:
        JudgeConfig object if file path provided, or string if built-in judge name
    """
    from oumi.core.configs.judge_config_v2 import JudgeConfig
    from oumi.judges_v2.builtin_simple_judges.registry import BuiltinJudgeRegistry

    # Check if it's a built-in judge name
    if BuiltinJudgeRegistry.get_config(config) is not None:
        # It's a built-in judge, return the string name (SimpleJudge will resolve it)
        return config

    # Check if it's a file path
    if Path(config).exists():
        return JudgeConfig.from_yaml_and_arg_list(config, extra_args)

    # Neither built-in nor existing file
    typer.echo(f"Error: '{config}' is not a built-in judge or an existing config file.")
    typer.echo("To see all available built-in judges, run: `oumi judge-v2 list-judges`")
    typer.echo("Please provide a built-in judge name or a path to a valid config file.")
    raise typer.Exit(code=1)


def _load_inference_config(config: str, extra_args: list[str]) -> "InferenceConfig":
    from oumi.core.configs import InferenceConfig

    if not Path(config).exists():
        typer.echo(f"Config file not found: '{config}'")
        raise typer.Exit(code=1)

    return InferenceConfig.from_yaml_and_arg_list(config, extra_args)


def judge_file(
    ctx: typer.Context,
    judge_config: Annotated[
        str,
        typer.Option(
            "--judge-config",
            help="Path to the judge config file or built-in judge name",
        ),
    ],
    inference_config: Annotated[
        str,
        typer.Option("--inference-config", help="Path to the inference config file"),
    ],
    input_file: Annotated[
        str, typer.Option("--input-file", help="Path to the dataset input file (jsonl)")
    ],
    output_file: Annotated[
        Optional[str],
        typer.Option("--output-file", help="Path to the output file (jsonl)"),
    ] = None,
    display_raw_output: bool = False,
):
    """Judge a dataset."""
    # Delayed imports
    from oumi import judge_v2
    # End imports

    # Load configs
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Resolve judge config (could be built-in or file path)
    judge_config_obj = _resolve_judge_config(judge_config, extra_args)

    # Load inference config from file
    inference_config_path = str(
        cli_utils.resolve_and_fetch_config(
            inference_config,
        )
    )
    inference_config_obj = _load_inference_config(inference_config_path, extra_args)

    # Ensure the dataset input file exists
    if not Path(input_file).exists():
        typer.echo(f"Input file not found: '{input_file}'")
        raise typer.Exit(code=1)

    # Judge the dataset
    judge_outputs = judge_v2.judge_file(
        judge_config=judge_config_obj,
        inference_config=inference_config_obj,
        input_file=input_file,
        output_file=output_file,
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


def list_builtin_judges():
    """List all available built-in judges."""
    from oumi.judges_v2.builtin_simple_judges.registry import BuiltinJudgeRegistry

    available_judges = BuiltinJudgeRegistry.list_available_judges()

    if not available_judges:
        typer.echo("No built-in judges available.")
        return

    table = Table(
        title="Available Judges",
        title_style="bold magenta",
        show_edge=False,
        show_lines=True,
    )
    table.add_column("Category", style="cyan")
    table.add_column("Judges", style="green")

    for category, judges in available_judges.items():
        judges_str = ", ".join(judges)
        table.add_row(category, judges_str)

    cli_utils.CONSOLE.print(table)

    typer.echo("\n\nUsage examples:\n")
    typer.echo("  # Use a built-in judge:")
    typer.echo("  oumi judge-v2 dataset \\")
    typer.echo("    --judge-config qa/relevance \\")
    typer.echo("    --inference-config inference_config.yaml \\")
    typer.echo("    --input-file input_data.jsonl\n")
    typer.echo("  # Use a custom judge config:")
    typer.echo("  oumi judge-v2 dataset \\")
    typer.echo("    --judge-config judge_config.yaml \\")
    typer.echo("    --inference-config inference_config.yaml \\")
    typer.echo("    --input-file input_data.jsonl\n ")
