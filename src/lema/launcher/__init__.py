import lema.launcher.clouds as clouds  # Ensure that the clouds are registered
from lema.launcher.launcher import Launcher
from lema.utils import logging

logging.configure_dependency_warnings()


__all__ = [
    "clouds",
    "Launcher",
]
