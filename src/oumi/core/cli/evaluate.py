from typing import Annotated

import typer

import oumi.core.cli.cli_utils as cli_utils
from oumi import evaluate as oumi_evaluate
from oumi.core.configs import EvaluationConfig
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
    # Load configuration
    parsed_config: EvaluationConfig = EvaluationConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()

    # Run evaluation
    oumi_evaluate(parsed_config)
