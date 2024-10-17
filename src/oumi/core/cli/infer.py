import typer
from typing_extensions import Annotated

import oumi.core.cli.cli_utils as cli_utils
from oumi import infer as oumi_infer
from oumi import infer_interactive as oumi_infer_interactive
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
    if not detach:
        return oumi_infer_interactive(parsed_config)
    if parsed_config.generation.input_filepath is None:
        raise ValueError("`input_filepath` must be provided for non-interactive mode.")
    generations = oumi_infer(parsed_config)

    # Don't print results if output_filepath is provided.
    if parsed_config.generation.output_filepath:
        return

    if len(generations) > 10:
        logger.warning(
            f"Outputting only the first 10 generations out of {len(generations)}"
        )
        generations = generations[:10]

    for generation in generations:
        print("------------")
        print(repr(generation))
    print("------------")
