import lema.launcher.clouds as clouds  # Ensure that the clouds are registered
from lema.core.types.configs import JobConfig
from lema.core.types.params.job_resources import JobResources, StorageMount
from lema.launcher.launcher import (
    Launcher,
    down,
    get_cloud,
    run,
    status,
    stop,
    up,
    which_clouds,
)
from lema.utils import logging

logging.configure_dependency_warnings()


__all__ = [
    "clouds",
    "down",
    "get_cloud",
    "JobConfig",
    "JobResources",
    "Launcher",
    "StorageMount",
    "run",
    "status",
    "stop",
    "up",
    "which_clouds",
]
