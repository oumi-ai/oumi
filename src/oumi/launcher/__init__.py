"""Launcher module for the Oumi (Open Universal Machine Intelligence) library.

This module provides functionality for launching and managing jobs across various
cloud platforms.

Example:
    >>> from oumi.launcher import Launcher, JobConfig, JobResources
    >>> launcher = Launcher()
    >>> job_resources = JobResources(cloud="local")
    >>> job_config = JobConfig(
    ...     name="my_job", resources=job_resources, run="python train.py"
    ... )
    >>> job_status = launcher.up(job_config, cluster_name="my_cluster")

Note:
    This module integrates with various cloud platforms. Ensure that the necessary
    credentials and configurations are set up for the cloud platform you intend to use.
"""

import oumi.launcher.clouds as clouds  # Ensure that the clouds are registered
from oumi.core.configs import JobConfig, JobResources, StorageMount
from oumi.launcher.launcher import (
    Launcher,
    cancel,
    down,
    get_cloud,
    run,
    status,
    stop,
    up,
    which_clouds,
)
from oumi.utils import logging

logging.configure_dependency_warnings()


__all__ = [
    "cancel",
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
