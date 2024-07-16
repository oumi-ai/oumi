import logging
import warnings
from typing import Union


def get_logger(name: str, level: str = "info") -> logging.Logger:
    """Gets a logger instance with the specified name and log level.

    Args:
        name (str): The name of the logger.
        level (str, optional): The log level to set for the logger. Defaults to "info".

    Returns:
        logging.Logger: The logger instance.
    """
    if name not in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(level.upper())

        # Default log format
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s]"
            "[%(filename)s:%(lineno)s] %(message)s"
        )

        # Add a console handler to the logger
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.upper())

        logger.addHandler(console_handler)
        logger.propagate = False
    else:
        logger = logging.getLogger(name)

    return logger


def update_logger_level(name: str, level: str = "info") -> None:
    """Updates the log level of the logger.

    Args:
        name (str): The logger instance to update.
        level (str, optional): The log level to set for the logger. Defaults to "info".
    """
    logger = logging.getLogger(name)
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
logger = get_logger("lema")
