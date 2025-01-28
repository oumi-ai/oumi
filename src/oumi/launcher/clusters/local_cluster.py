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

import uuid
from copy import deepcopy
from typing import Any, Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, JobStatus
from oumi.launcher.clients.local_client import LocalClient
from oumi.utils.logging import logger


def _validate_job_config(job: JobConfig) -> None:
    """Validates the provided job configuration.

    Args:
        job: The job to validate.
    """
    if not job.working_dir:
        raise ValueError("Working directory must be provided for local jobs.")
    if not job.run:
        raise ValueError("Run script must be provided for local jobs.")
    if job.resources.cloud != "local":
        raise ValueError(
            f"`Resources.cloud` must be `local`. "
            f"Unsupported cloud: {job.resources.cloud}"
        )
    # Warn that other resource parameters are unused for local jobs.
    if job.resources.region:
        logger.warning("Region is unused for local jobs.")
    if job.resources.zone:
        logger.warning("Zone is unused for local jobs.")
    if job.resources.accelerators:
        logger.warning("Accelerators are unused for local jobs.")
    if job.resources.cpus:
        logger.warning("CPUs are unused for local jobs.")
    if job.resources.memory:
        logger.warning("Memory is unused for local jobs.")
    if job.resources.instance_type:
        logger.warning("Instance type is unused for local jobs.")
    if job.resources.disk_size:
        logger.warning("Disk size is unused for local jobs.")
    if job.resources.instance_type:
        logger.warning("Instance type is unused for local jobs.")
    # Warn that storage mounts are currently unsupported.
    if len(job.storage_mounts.items()) > 0:
        logger.warning("Storage mounts are currently unsupported for local jobs.")
    if len(job.file_mounts.items()) > 0:
        logger.warning("File mounts are currently unsupported for local jobs.")


class LocalCluster(BaseCluster):
    """A cluster implementation for running jobs locally."""

    def __init__(self, name: str, client: LocalClient) -> None:
        """Initializes a new instance of the LocalCluster class."""
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two LocalClusters are equal."""
        if not isinstance(other, LocalCluster):
            return False
        return self.name() == other.name()

    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the jobs on this cluster if it exists, else returns None."""
        for job in self.get_jobs():
            if job.id == job_id:
                return job
        return None

    def get_jobs(self) -> list[JobStatus]:
        """Lists the jobs on this cluster."""
        jobs = self._client.list_jobs()
        for job in jobs:
            job.cluster = self._name
        return jobs

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the specified job on this cluster."""
        self._client.cancel(job_id)
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} not found.")
        return job

    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster.

        Args:
            job: The job to run.

        Returns:
            The job status.
        """
        job_copy = deepcopy(job)
        _validate_job_config(job_copy)
        if not job_copy.name:
            job_copy.name = uuid.uuid1().hex
        status = self._client.submit_job(job_copy)
        status.cluster = self._name
        return status

    def stop(self) -> None:
        """Cancels all jobs, running or queued."""
        for job in self.get_jobs():
            self.cancel_job(job.id)

    def down(self) -> None:
        """Cancels all jobs, running or queued."""
        for job in self.get_jobs():
            self.cancel_job(job.id)
