import logging
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
    # Bundle the args into key-value pairs. Throws a ValueError if the number of args is
    # odd.
    pairs = zip(*[iter(ctx.args)] * 2, strict=True)  # type: ignore
    try:
        for key, value in pairs:
            if not key.startswith("--"):
                raise typer.BadParameter(
                    "Extra arguments must start with '--'. "
                    f"Found argument `{key}` with value `{value}`"
                )
            cli_arg = f"{key[2:]}={value}"
            args.append(cli_arg)
    except ValueError:
        bad_args = " ".join(ctx.args)
        raise typer.BadParameter(
            "Extra arguments must be in `--argname value` pairs. "
            f"Recieved: `{bad_args}`"
        )
    return args


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
