from typing import Annotated

import typer

import oumi.core.cli.cli_utils as cli_utils
from oumi import train as oumi_train
from oumi.core.configs import TrainingConfig
from oumi.core.distributed import set_random_seeds
from oumi.utils.logging import logger
from oumi.utils.torch_utils import (
    device_cleanup,
    limit_per_process_memory,
)


def train(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for training."
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
):
    """Train a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        level: The logging level for the specified command.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)
    parsed_config: TrainingConfig = TrainingConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.validate()

    limit_per_process_memory()
    device_cleanup()
    set_random_seeds(parsed_config.training.seed)

    # Run training
    oumi_train(parsed_config)

    device_cleanup()
