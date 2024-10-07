import typer
from typing_extensions import Annotated

import oumi.core.cli.cli_utils as cli_utils
import oumi.infer
from oumi.core.configs import InferenceConfig
from oumi.utils.logging import logger


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
):
    """Run inference on a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for inference.
        detach: Do not run in an interactive session.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    parsed_config: InferenceConfig = InferenceConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()
    if detach:
        if parsed_config.generation.input_filepath is None:
            raise ValueError(
                "`input_filepath` must be provided for non-interactive mode."
            )
        oumi.infer.infer(
            model_params=parsed_config.model,
            generation_config=parsed_config.generation,
            input=[],
        )
    else:
        oumi.infer.infer_interactive(parsed_config)
