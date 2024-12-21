from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.utils.logging import logger


def evaluate(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training."
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Evaluate a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for evaluation.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    # Delayed imports
    from oumi import evaluate as oumi_evaluate
    from oumi.core.configs import EvaluationConfig
    # End imports

    # Load configuration
    parsed_config: EvaluationConfig = EvaluationConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    # Run evaluation
    oumi_evaluate(parsed_config)
