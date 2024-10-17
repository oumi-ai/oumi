import io
from typing import Optional

import PIL.Image
import typer
from typing_extensions import Annotated

import oumi.core.cli.cli_utils as cli_utils
from oumi import infer as oumi_infer
from oumi import infer_interactive as oumi_infer_interactive
from oumi.core.configs import InferenceConfig
from oumi.utils.logging import logger


def _load_image_png_bytes(input_image_filepath: str) -> bytes:
    try:
        image_bin = PIL.Image.open(input_image_filepath).convert("RGB")

        output = io.BytesIO()
        image_bin.save(output, format="PNG")
        return output.getvalue()
    except Exception:
        logger.error(f"Failed to load image from path: {input_image_filepath}")
        raise


def infer(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS,
            help="Path to the configuration file for inference.",
        ),
    ],
    detach: Annotated[
        bool,
        typer.Option("-d", "--detach", help="Do not run in an interactive session."),
    ] = False,
    image: Annotated[
        Optional[str],
        typer.Option(
            "-i",
            "--image",
            help=(
                "File path of an input image to be used with `image+text` VLLMs. "
                "Only used in interactive mode."
            ),
        ),
    ] = None,
):
    """Run inference on a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        detach: Do not run in an interactive session.
        image: Path to the input image for `image+text` VLLMs.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()

    input_image_png_bytes: Optional[bytes] = (
        _load_image_png_bytes(image) if image else None
    )

    if detach:
        if parsed_config.generation.input_filepath is None:
            raise ValueError(
                "`input_filepath` must be provided for non-interactive mode."
            )
        oumi_infer(config=parsed_config)
    else:
        oumi_infer_interactive(parsed_config, input_image_bytes=input_image_png_bytes)
