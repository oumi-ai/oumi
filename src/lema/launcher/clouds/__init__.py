"""Clouds module for the LeMa (Learning Machines) library.

This module provides implementations for various cloud platforms that can be used
with the LeMa launcher for running and managing jobs.

Example:
    >>> from lema.launcher.clouds import LocalCloud
    >>> local_cloud = LocalCloud()
    >>> local_cloud.run_job(job_config)

Note:
    Ensure that you have the necessary credentials and configurations set up
    for the cloud platform you intend to use.
"""

from lema.launcher.clouds.local_cloud import LocalCloud
from lema.launcher.clouds.polaris_cloud import PolarisCloud
from lema.launcher.clouds.sky_cloud import SkyCloud
from lema.utils import logging

logging.configure_dependency_warnings()


__all__ = [
    "LocalCloud",
    "PolarisCloud",
    "SkyCloud",
]
