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

import logging
import os
from enum import Enum
from typing import Annotated, Optional

import typer

from oumi.utils.logging import logger

CONTEXT_ALLOW_EXTRA_ARGS = {"allow_extra_args": True, "ignore_unknown_options": True}
CONFIG_FLAGS = ["--config", "-c"]


def parse_extra_cli_args(ctx: typer.Context) -> list[str]:
    """Parses extra CLI arguments into a list of strings.

    Args:
        ctx: The Typer context object.

    Returns:
        List[str]: The extra CLI arguments
    """
    args = []

    # The following formats are supported:
    # 1. Space separated: "--foo" "2"
    # 2. `=`-separated: "--foo=2"
    try:
        num_args = len(ctx.args)
        idx = 0
        while idx < num_args:
            original_key = ctx.args[idx]
            key = original_key.strip()
            if not key.startswith("--"):
                raise typer.BadParameter(
                    "Extra arguments must start with '--'. "
                    f"Found argument `{original_key}` at position {idx}: `{ctx.args}`"
                )
            # Strip leading "--"

            key = key[2:]
            pos = key.find("=")
            if pos >= 0:
                # '='-separated argument
                value = key[(pos + 1) :].strip()
                key = key[:pos].strip()
                if not key:
                    raise typer.BadParameter(
                        "Empty key name for `=`-separated argument. "
                        f"Found argument `{original_key}` at position {idx}: "
                        f"`{ctx.args}`"
                    )
                idx += 1
            else:
                # Space separated argument
                if idx + 1 >= num_args:
                    raise typer.BadParameter(
                        "Trailing argument has no value assigned. "
                        f"Found argument `{original_key}` at position {idx}: "
                        f"`{ctx.args}`"
                    )
                value = ctx.args[idx + 1].strip()
                idx += 2

            if value.startswith("--"):
                logger.warning(
                    f"Argument value ('{value}') starts with `--`! "
                    f"Key: '{original_key}'"
                )

            cli_arg = f"{key}={value}"
            args.append(cli_arg)
    except ValueError:
        bad_args = " ".join(ctx.args)
        raise typer.BadParameter(
            "Extra arguments must be in `--argname value` pairs. "
            f"Recieved: `{bad_args}`"
        )
    logger.debug(f"\n\nParsed CLI args:\n{args}\n\n")
    return args


def configure_common_env_vars() -> None:
    """Sets common environment variables if needed."""
    if "ACCELERATE_LOG_LEVEL" not in os.environ:
        os.environ["ACCELERATE_LOG_LEVEL"] = "info"
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LogLevel(str, Enum):
    """The available logging levels."""

    DEBUG = logging.getLevelName(logging.DEBUG)
    INFO = logging.getLevelName(logging.INFO)
    WARNING = logging.getLevelName(logging.WARNING)
    ERROR = logging.getLevelName(logging.ERROR)
    CRITICAL = logging.getLevelName(logging.CRITICAL)


def set_log_level(level: Optional[LogLevel]):
    """Sets the logging level for the current command.

    Args:
        level (Optional[LogLevel]): The log level to use.
    """
    if not level:
        return
    uppercase_level = level.upper()
    logger.setLevel(uppercase_level)
    print(f"Set log level to {uppercase_level}")


LOG_LEVEL_TYPE = Annotated[
    Optional[LogLevel],
    typer.Option(
        "--log-level",
        "-log",
        help="The logging level for the specified command.",
        show_default=False,
        show_choices=True,
        case_sensitive=False,
        callback=set_log_level,
    ),
]
