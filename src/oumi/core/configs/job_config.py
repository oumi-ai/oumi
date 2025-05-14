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

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from typing_extensions import override

from oumi.core.configs.base_config import BaseConfig
from oumi.utils.logging import logger
from oumi.utils.str_utils import (
    get_editable_install_override_env_var,
    set_oumi_install_editable,
)
from oumi.utils.version_utils import is_dev_build


@dataclass
class StorageMount:
    """A storage system mount to attach to a node."""

    source: str = MISSING
    """The remote path to mount the local path to (Required).

    e.g. 'gs://bucket/path' for GCS, 's3://bucket/path' for S3, or 'r2://path' for
    R2.
    """

    store: str = MISSING
    """The remote storage solution (Required).

    Must be one of 's3', 'gcs' or 'r2'.
    """


@dataclass
class JobResources:
    """Resources required for a single node in a job."""

    cloud: str = MISSING
    """The cloud used to run the job (required).

    Options:
        - aws: Amazon Web Services
        - azure: Microsoft Azure
        - gcp: Google Cloud Platform
        - lambda: Lambda Cloud
        - local: The local machine launching the job
        - polaris: The Polaris cluster at Argonne National Laboratory
        - runpod: RunPod
    """

    region: Optional[str] = None
    """The region to use (optional). Supported values vary by environment."""

    zone: Optional[str] = None
    """The zone to use (optional). Supported values vary by environment."""

    accelerators: Optional[str] = None
    """Accelerator type (optional). Supported values vary by environment.

    For GCP you may specify the accelerator name and count, e.g. "V100:4".
    """

    cpus: Optional[str] = None
    """Number of vCPUs to use per node (optional).

    Sky-based clouds support strings with
    modifiers, e.g. "2+" to indicate at least 2 vCPUs.
    """

    memory: Optional[str] = None
    """Memory to allocate per node in GiB (optional).

    Sky-based clouds support strings
    with modifiers, e.g. "256+" to indicate at least 256 GB.
    """

    instance_type: Optional[str] = None
    """Instance type to use (optional).

    Supported values vary by environment.
    The instance type is automatically inferred if `accelerators` is specified.
    """

    use_spot: bool = False
    """Whether the cluster should use spot instances (optional).

    If unspecified, defaults to False (on-demand instances).
    """

    disk_size: Optional[int] = None
    """Disk size in GiB to allocate for OS (mounted at /) (optional)

    Ignored by Polaris.
    """

    disk_tier: Optional[str] = None
    """Disk tier to use for OS (optional).

    For sky-based clouds this Could be one of 'low', 'medium', 'high', 'ultra', or
    'best' (default: None). As of Feb '25, only AWS, Azure, GCP, and OCI support
    disk tiers.
    """

    image_id: Optional[str] = None
    """The image id used to boot the instances (optional).

    You can specify a docker by using the format `docker:<image_id>`.
    This field is not applicable for all clouds.
    """


@dataclass
class JobConfig(BaseConfig):
    """Configuration for launching jobs on a cluster."""

    name: Optional[str] = None
    """Job name (optional). Only used for display purposes."""

    user: Optional[str] = None
    """The user that the job will run as (optional). Required only for Polaris."""

    working_dir: str = MISSING
    """The local directory containing the scripts required to execute this job.

    This directory will be copied to the remote node before the job is executed.
    """

    num_nodes: int = 1
    """The number of nodes to use for the job. Defaults to 1."""

    resources: JobResources = field(default_factory=JobResources)
    """The resources required for each node in the job."""

    envs: dict[str, str] = field(default_factory=dict)
    """The environment variables to set on the node."""

    file_mounts: dict[str, str] = field(default_factory=dict)
    """File mounts to attach to the node.

    For mounting (copying) local directories, the key is the file path on the remote
    and the value is the local path.
    The keys of `file_mounts` cannot be shared with `storage_mounts`.
    """

    storage_mounts: dict[str, StorageMount] = field(default_factory=dict)
    """Storage system mounts to attach to the node.

    For mounting remote storage solutions, the key is the file path on the remote
    and the value is a StorageMount.
    The keys of `storage_mounts` cannot be shared with `file_mounts`.
    """

    setup: Optional[str] = None
    """The setup script to run on every node. Optional.

    `setup` will always be executed before `run`. In sky-based clouds, `setup` is
    executed only once upon cluster creation, not once per job.

    ex) pip install -r requirements.txt
    """

    run: str = MISSING
    """The script to run on every node. Required. Runs after `setup`."""

    @override
    def __finalize_and_validate__(self):
        """Finalizes and validates the configuration."""
        # (experimental) If the OUMI_FORCE_EDITABLE_INSTALL env var is set to a truthy
        # value, and we're running a dev build of oumi, attempt to modify the setup/run
        # scripts in the job config to install Oumi in editable mode from source, as
        # opposed to installing from PyPI.
        # This is intended for developers who are modifying Oumi source code and need to
        # test their changes in a remote job; by default, all of our job configs install
        # Oumi from PyPI.
        if get_editable_install_override_env_var() and is_dev_build():
            logger.info("-" * 80)
            logger.info(
                "OUMI_FORCE_EDITABLE_INSTALL detected! Attempting to modify job "
                "config to install Oumi in editable mode from source..."
            )
            if self.setup:
                logger.info("Modifying setup script...")
                self.setup = set_oumi_install_editable(self.setup)
            if self.run:
                logger.info("Modifying run script...")
                self.run = set_oumi_install_editable(self.run)
            logger.info("-" * 80)
