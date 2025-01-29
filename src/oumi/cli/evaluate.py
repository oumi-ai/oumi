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
