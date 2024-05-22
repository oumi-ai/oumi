import logging


def get_logger(name, level="info") -> logging.Logger:
    """Get a logger instance with the specified name and log level.

    Args:
        name (str): The name of the logger.
        level (str, optional): The log level to set for the logger. Defaults to "info".

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)

    if name not in logging.Logger.manager.loggerDict:
        # Default log format
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s]"
            "[%(pathname)s:%(lineno)s] %(message)s"
        )

        # Add a console handler to the logger
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.upper())

        logger.addHandler(console_handler)

    return logger
