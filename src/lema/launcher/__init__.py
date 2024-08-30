"""Launcher module for the LeMa (Learning Machines) library.

This module provides functionality for launching and managing jobs across various
cloud platforms.

Example:
    >>> from lema.launcher import Launcher, JobConfig
    >>> launcher = Launcher()
    >>> job_config = JobConfig(name="my_job", command="python train.py")
    >>> launcher.run(job_config)

Note:
    This module integrates with various cloud platforms. Ensure that the necessary
    credentials and configurations are set up for the cloud platform you intend to use.
"""

import lema.launcher.clouds as clouds  # Ensure that the clouds are registered
from lema.core.configs import JobConfig, JobResources, StorageMount
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
