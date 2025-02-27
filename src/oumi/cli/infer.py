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
from typing import Annotated, Final, Optional

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.utils.logging import logger

_DEFAULT_CLI_PDF_DPI: Final[int] = 200


def infer(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ],
    interactive: Annotated[
        bool,
        typer.Option("-i", "--interactive", help="Run in an interactive session."),
    ] = False,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help=(
                "File path or URL of an input image to be used with image+text VLLMs. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option(
            "--system-prompt",
            help=(
                "System prompt for task-specific instructions. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Run inference on a model.

    If `input_filepath` is provided in the configuration file, inference will run on
    those input examples. Otherwise, inference will run interactively with user-provided
    inputs.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        output_dir: Directory to save configs
        (defaults to OUMI_DIR env var or ~/.oumi/fetch).
        interactive: Whether to run in an interactive session.
        image: Path to the input image for `image+text` VLLMs.
        system_prompt: System prompt for task-specific instructions.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            config,
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
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_image_png_bytes: Optional[list[bytes]] = None
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

    if interactive:
        if parsed_config.input_path:
            logger.warning(
                "Interactive inference requested, skipping reading from `input_path`."
            )
        return oumi_infer_interactive(
            parsed_config,
            input_image_bytes=input_image_png_bytes,
            system_prompt=system_prompt,
        )
    if parsed_config.input_path is None:
        raise ValueError("One of `--interactive` or `input_path` must be provided.")
    generations = oumi_infer(parsed_config)

    # Don't print results if output_filepath is provided.
    if parsed_config.output_path:
        return

    for generation in generations:
        print("------------")
        print(repr(generation))
    print("------------")
