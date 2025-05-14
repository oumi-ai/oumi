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

"""Clouds module for the Oumi (Open Universal Machine Intelligence) library.

This module provides implementations for various cloud platforms that can be used
with the Oumi launcher for running and managing jobs.

Example:
    >>> from oumi.launcher import JobConfig, JobResources
    >>> from oumi.launcher.clouds import LocalCloud
    >>> local_cloud = LocalCloud()
    >>> job_resources = JobResources(cloud="local")
    >>> job_config = JobConfig(
    ...     name="my_job", resources=job_resources, run="python train.py"
    ... )
    >>> job_status = local_cloud.up_cluster(job_config, name="my_cluster")

Note:
    Ensure that you have the necessary credentials and configurations set up
    for the cloud platform you intend to use.
"""

from oumi.launcher.clouds.local_cloud import LocalCloud
from oumi.launcher.clouds.polaris_cloud import PolarisCloud
from oumi.launcher.clouds.sky_cloud import SkyCloud
from oumi.launcher.clouds.slurm_cloud import SlurmCloud
from oumi.utils import logging

logging.configure_dependency_warnings()


__all__ = [
    "LocalCloud",
    "PolarisCloud",
    "SkyCloud",
    "SlurmCloud",
]
