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

import os
from typing import Annotated, Final

import typer
from rich.table import Table

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.cli.completions import complete_infer_config
from oumi.utils.logging import logger

_DEFAULT_CLI_PDF_DPI: Final[int] = 200

_list_configs_callback = cli_utils.create_list_configs_callback(
    AliasType.INFER, "Available Inference Configs", "infer"
)


def infer(
    ctx: typer.Context,
    # Main options
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path or config name (e.g. llama3.1-8b).",
            rich_help_panel="Options",
            autocompletion=complete_infer_config,
        ),
    ],
    list_configs: Annotated[
        bool,
        typer.Option(
            "--list",
            help="List all available inference configs.",
            callback=_list_configs_callback,
            is_eager=True,
            rich_help_panel="Options",
        ),
    ] = False,
    interactive: Annotated[
        bool,
        typer.Option(
            "-i",
            "--interactive",
            help="Run in an interactive session.",
            rich_help_panel="Options",
        ),
    ] = False,
    image: Annotated[
        str | None,
        typer.Option(
            "--image",
            help=(
                "File path or URL of an input image to be used with image+text VLLMs. "
                "Only used in interactive mode."
            ),
            rich_help_panel="Options",
        ),
    ] = None,
    system_prompt: Annotated[
        str | None,
        typer.Option(
            "--system-prompt",
            help=(
                "System prompt for task-specific instructions. "
                "Only used in interactive mode."
            ),
            rich_help_panel="Options",
        ),
    ] = None,
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
    top_p: Annotated[
        float | None,
        typer.Option(
            "--generation.top_p",
            help="Nucleus sampling threshold.",
            rich_help_panel="Generation",
        ),
    ] = None,
    # I/O overrides
    input_path: Annotated[
        str | None,
        typer.Option(
            "--input_path",
            help="Path to input file with prompts (JSONL).",
            rich_help_panel="I/O",
        ),
    ] = None,
    output_path: Annotated[
        str | None,
        typer.Option(
            "--output_path",
            help="Path to output file for generated text.",
            rich_help_panel="I/O",
        ),
    ] = None,
):
    """Run inference on a model.

    If `input_filepath` is provided in the configuration file, inference will run on
    those input examples. Otherwise, inference will run interactively with user-provided
    inputs.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        list_configs: List all available inference configs.
        interactive: Whether to run in an interactive session.
        image: Path to the input image for `image+text` VLLMs.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
        model_name: Model name or HuggingFace path.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        input_path: Path to input file with prompts.
        output_path: Path to output file for generated text.
    """
    # Auto-collect overrides from dot-notation options (e.g., --model.model_name)
    option_overrides = cli_utils.collect_config_overrides(ctx)
    # Parse any additional extra args from command line
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    # Combine: explicit options take precedence (added last)
    all_overrides = extra_args + option_overrides

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.INFER),
        )
    )

    # Delayed imports
    from oumi import infer as oumi_infer
    from oumi import infer_interactive as oumi_infer_interactive
    from oumi.core.configs import InferenceConfig
    from oumi.utils.image_utils import (
        create_png_bytes_from_image_list,
        load_image_png_bytes_from_path,
        load_image_png_bytes_from_url,
        load_pdf_pages_from_path,
        load_pdf_pages_from_url,
    )
    # End imports

    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, all_overrides, logger=logger
    )

    # Apply non-dot-notation overrides
    if input_path is not None:
        parsed_config.input_path = input_path
    if output_path is not None:
        parsed_config.output_path = output_path

    parsed_config.finalize_and_validate()

    if verbose:
        # Print configuration for verification
        parsed_config.print_config(logger)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_image_png_bytes: list[bytes] | None = None
    if image:
        image_lower = image.lower()
        if image_lower.startswith("http://") or image_lower.startswith("https://"):
            if image_lower.endswith(".pdf"):
                input_image_png_bytes = create_png_bytes_from_image_list(
                    load_pdf_pages_from_url(image, dpi=_DEFAULT_CLI_PDF_DPI)
                )
            else:
                input_image_png_bytes = [load_image_png_bytes_from_url(image)]
        else:
            if image_lower.endswith(".pdf"):
                input_image_png_bytes = create_png_bytes_from_image_list(
                    load_pdf_pages_from_path(image, dpi=_DEFAULT_CLI_PDF_DPI)
                )
            else:
                input_image_png_bytes = [load_image_png_bytes_from_path(image)]
    if parsed_config.input_path:
        if interactive:
            logger.warning(
                "Input path provided, skipping interactive inference. "
                "To run in interactive mode, do not provide an input path."
            )
        # Display loading spinner while running batch inference
        with cli_utils.CONSOLE.status(
            "[green]Running inference...[/green]", spinner="dots"
        ):
            generations = oumi_infer(parsed_config)
        # Don't print results if output_filepath is provided.
        if parsed_config.output_path:
            return
        table = Table(
            title="Inference Results",
            title_style="bold magenta",
            show_edge=False,
            show_lines=True,
        )
        table.add_column("Conversation", style="green")
        for generation in generations:
            table.add_row(repr(generation))
        cli_utils.CONSOLE.print(table)
        return
    if not interactive:
        logger.warning(
            "No input path provided, running in interactive mode. "
            "To run with an input path, provide one in the configuration file."
        )
    return oumi_infer_interactive(
        parsed_config,
        input_image_bytes=input_image_png_bytes,
        system_prompt=system_prompt,
        console=cli_utils.CONSOLE,
    )
