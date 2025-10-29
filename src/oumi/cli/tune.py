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

from typing import Annotated

import typer

import oumi.cli.cli_utils as cli_utils
from oumi.cli.alias import AliasType, try_get_config_name_for_alias
from oumi.utils.logging import logger


def tune(
    ctx: typer.Context,
    config: Annotated[
        str,
        typer.Option(
            *cli_utils.CONFIG_FLAGS, help="Path to the configuration file for tuning."
        ),
    ],
    level: cli_utils.LOG_LEVEL_TYPE = None,
    verbose: cli_utils.VERBOSE_TYPE = False,
):
    """Tune hyperparameters for a model.

    Args:
        ctx: The Typer context object.
        config: Path to the configuration file for training.
        level: The logging level for the specified command.
        verbose: Enable verbose logging with additional debug information.
    """
    extra_args = cli_utils.parse_extra_cli_args(ctx)

    config = str(
        cli_utils.resolve_and_fetch_config(
            try_get_config_name_for_alias(config, AliasType.TUNE),
        )
    )
    with cli_utils.CONSOLE.status(
        "[green]Loading configuration...[/green]", spinner="dots"
    ):
        # Delayed imports
        from oumi.core.configs import TuningConfig
        from oumi.tune import tune as oumi_tune
        from oumi.utils.torch_utils import (
            device_cleanup,
            limit_per_process_memory,
        )
        # End imports

    cli_utils.configure_common_env_vars()

    parsed_config: TuningConfig = TuningConfig.from_yaml_and_arg_list(
        config, extra_args, logger=logger
    )
    parsed_config.finalize_and_validate()

    limit_per_process_memory()
    device_cleanup()

    # Run training
    oumi_tune(parsed_config, verbose=verbose)

    device_cleanup()
