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
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from oumi.utils.cli_styling import StyleLevel, get_style_level

# Define log colors theme
LOG_THEME = Theme(
    {
        "logging.level.debug": "dim cyan",
        "logging.level.info": "bright_blue",
        "logging.level.warning": "bright_yellow",
        "logging.level.error": "bright_red",
        "logging.level.critical": "bold bright_white on red",
        "logging.time": "dim bright_white",
        "logging.file": "dim bright_magenta",
        "logging.line": "dim bright_white",
        "logging.rank": "bright_cyan",
        "logging.thread": "dim green",
        "logging.pid": "dim blue",
        "logging.name": "green",
    }
)


def get_logger(
    name: str,
    level: str = "info",
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Gets a logger instance with the specified name and log level.

    Args:
        name : The name of the logger.
        level (optional): The log level to set for the logger. Defaults to "info".
        log_dir (optional): Directory to store log files. Defaults to None.

    Returns:
        logging.Logger: The logger instance.
    """
    if name not in logging.Logger.manager.loggerDict:
        configure_logger(name, level=level, log_dir=log_dir)

    logger = logging.getLogger(name)
    return logger


def _detect_rank() -> int:
    """Detects rank.

    Reading the rank from the environment variables instead of
    get_device_rank_info to avoid circular imports.
    """
    for var_name in (
        "RANK",
        "SKYPILOT_NODE_RANK",  # SkyPilot
        "PMI_RANK",  # HPC
    ):
        rank = os.environ.get(var_name, None)
        if rank is not None:
            rank = int(rank)
            if rank < 0:
                raise ValueError(f"Negative rank: {rank} specified in '{var_name}'!")
            return rank
    return 0


def configure_logger(
    name: str,
    level: str = "info",
    log_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Configures a logger with the specified name and log level."""
    logger = logging.getLogger(name)

    # Remove any existing handlers
    logger.handlers = []

    # Configure the logger
    logger.setLevel(level.upper())

    device_rank = _detect_rank()

    # Add console handler for primary rank only
    if device_rank == 0:
        # Standard formatter used for all log modes
        std_formatter = logging.Formatter(
            "[%(asctime)s][%(name)s]"
            f"[rank{device_rank}]"
            "[pid:%(process)d][%(threadName)s]"
            "[%(levelname)s]][%(filename)s:%(lineno)s] %(message)s"
        )

        # Determine whether to use rich logging based on style level
        if get_style_level() == StyleLevel.FULL:
            console = Console(theme=LOG_THEME)
            handler = RichHandler(
                console=console,
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_level=True,
                show_path=True,
                enable_link_path=True,
                log_time_format="%Y-%m-%d %H:%M:%S",
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(std_formatter)

        handler.setLevel(level.upper())
        logger.addHandler(handler)

    # Add a file handler if log_dir is provided (for all ranks)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # For file logging, always use standard formatter
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s]"
            f"[rank{device_rank}]"
            "[pid:%(process)d][%(threadName)s]"
            "[%(levelname)s]][%(filename)s:%(lineno)s] %(message)s"
        )

        file_handler = logging.FileHandler(log_dir / f"rank_{device_rank:04d}.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level.upper())
        logger.addHandler(file_handler)

    logger.propagate = False


def update_logger_level(name: str, level: str = "info") -> None:
    """Updates the log level of the logger.

    Args:
        name (str): The logger instance to update.
        level (str, optional): The log level to set for the logger. Defaults to "info".
    """
    logger = get_logger(name, level=level)
    logger.setLevel(level.upper())

    for handler in logger.handlers:
        handler.setLevel(level.upper())


def configure_dependency_warnings(level: Union[str, int] = "info") -> None:
    """Ignores non-critical warnings from dependencies, unless in debug mode.

    Args:
        level (str, optional): The log level to set for the logger. Defaults to "info".
    """
    level_value = logging.DEBUG
    if isinstance(level, str):
        level_value = logging.getLevelName(level.upper())
        if not isinstance(level_value, int):
            raise TypeError(
                f"getLevelName() mapped log level name to non-integer: "
                f"{type(level_value)}!"
            )
    elif isinstance(level, int):
        level_value = int(level)

    if level_value > logging.DEBUG:
        warnings.filterwarnings(action="ignore", category=UserWarning, module="torch")
        warnings.filterwarnings(
            action="ignore", category=UserWarning, module="huggingface_hub"
        )
        warnings.filterwarnings(
            action="ignore", category=UserWarning, module="transformers"
        )


# Default logger for the package
logger = get_logger("oumi")


def show_log_formatting() -> None:
    """
    Display examples of each log level to demonstrate the formatting.
    Useful for testing the log colors and styling.
    """
    # Show a sample of each log level with descriptive messages
    logger.debug("DEBUG level: Detailed information for debugging purposes")
    logger.info("INFO level: General information about program operation")
    logger.warning("WARNING level: Indication of a potential problem")
    logger.error("ERROR level: An issue that needs attention")

    # Show traceback formatting
    try:
        1 / 0
    except Exception as e:
        logger.exception(f"EXCEPTION with traceback: {e}")

    logger.critical("CRITICAL level: A serious error affecting program operation")
