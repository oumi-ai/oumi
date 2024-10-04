import typer
from typing_extensions import Annotated

import oumi.core.cli.utils as utils
import oumi.infer
from oumi.core.configs import InferenceConfig
from oumi.utils.logging import logger


def infer(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *utils.CONFIG_FLAGS, help="Path to the configuration file for inference."
        ),
    ],
    interactive: Annotated[
        bool, typer.Option("-i", "--interactive", help="Run in an interactive sesion.")
    ] = True,
):
    """Run inference on a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        interactive: Run in an interactive sesion.
    """
    extra_args = utils.parse_extra_cli_args(ctx)
    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()

    if interactive:
        oumi.infer.infer_interactive(parsed_config)
    else:
        if parsed_config.generation.input_filepath is None:
            raise ValueError(
                "`input_filepath` must be provided for non-interactive mode."
            )
        oumi.infer.infer(
            model_params=parsed_config.model,
            generation_config=parsed_config.generation,
            input=[],
        )
