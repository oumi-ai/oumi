import os
from typing import Optional

import typer
from typing_extensions import Annotated

import oumi.core.cli.cli_utils as cli_utils
from oumi import infer as oumi_infer
from oumi import infer_interactive as oumi_infer_interactive
from oumi.core.configs import InferenceConfig
from oumi.utils.image_utils import load_image_png_bytes_from_path
from oumi.utils.logging import logger


def infer(
    ctx: typer.Context,
    config: Annotated[
        Optional[str],
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option("-i", "--interactive", help="Run in an interactive session."),
    ] = False,
    image: Annotated[
        Optional[str],
        typer.Option(
            "--image",
            help=(
                "File path of an input image to be used with `image+text` VLLMs. "
                "Only used in interactive mode."
            ),
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
        interactive: Whether to run in an interactive session.
        image: Path to the input image for `image+text` VLLMs.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    input_image_png_bytes: Optional[bytes] = (
        load_image_png_bytes_from_path(image) if image else None
    )

    if interactive:
        if parsed_config.input_path:
            logger.warning(
                "Interactive inference requested, skipping reading from "
                "`input_path`."
            )
        return oumi_infer_interactive(
            parsed_config, input_image_bytes=input_image_png_bytes
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
