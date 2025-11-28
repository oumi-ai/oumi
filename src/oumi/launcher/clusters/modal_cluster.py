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

"""Modal cluster implementation for running jobs on Modal.com infrastructure."""

import io
import uuid
from copy import deepcopy
from typing import Any, Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, JobStatus
from oumi.launcher.clients.modal_client import ModalClient
from oumi.utils.logging import logger


def _validate_job_config(job: JobConfig) -> None:
    """Validates the provided job configuration for Modal.

    Args:
        job: The job to validate.

    Raises:
        ValueError: If the job configuration is invalid.
    """
    if not job.run:
        raise ValueError("Run script must be provided for Modal jobs.")
    if job.resources.cloud != "modal":
        raise ValueError(
            f"`Resources.cloud` must be `modal`. "
            f"Unsupported cloud: {job.resources.cloud}"
        )

    # Warn about unsupported features
    if job.num_nodes > 1:
        logger.warning(
            "Multi-node jobs are in early preview on Modal. "
            "Only single-node multi-GPU is fully supported. "
            f"Requested {job.num_nodes} nodes."
        )

    if job.resources.region:
        logger.warning(
            "Region specification is not supported on Modal. "
            "Modal automatically selects the best region."
        )

    if job.resources.zone:
        logger.warning(
            "Zone specification is not supported on Modal. "
            "Modal automatically handles zone selection."
        )

    if job.resources.disk_size:
        logger.warning(
            "Custom disk size is not directly supported on Modal. "
            "Modal provides ephemeral storage automatically."
        )

    if job.resources.disk_tier:
        logger.warning("Disk tier is not supported on Modal.")

    if job.resources.instance_type:
        logger.warning(
            "Instance type specification is not supported on Modal. "
            "Use accelerators to specify GPU type instead."
        )

    if job.resources.use_spot:
        logger.warning(
            "Spot instances are not a concept on Modal. "
            "Modal uses a serverless pricing model."
        )

    if job.resources.image_id or job.resources.image_id_map:
        logger.warning(
            "Custom image IDs are not supported on Modal. "
            "Use the setup script to install dependencies, or configure "
            "a custom Modal Image in the generated app."
        )

    if job.file_mounts:
        logger.warning(
            "File mounts are handled via Modal's working directory sync. "
            "Consider using storage_mounts with Modal Volumes for large data."
        )


class ModalCluster(BaseCluster):
    """A cluster implementation for running jobs on Modal.com.

    Note: Modal is a serverless platform, so the concept of a "cluster" is
    different from traditional cloud providers. This class provides a unified
    interface for managing Modal jobs within the Oumi launcher framework.

    A single ModalCluster manages all jobs submitted to Modal under a given
    namespace (cluster name).
    """

    def __init__(self, name: str, client: ModalClient) -> None:
        """Initializes a new instance of the ModalCluster class.

        Args:
            name: The name of the cluster (used for job organization)
            client: The ModalClient instance for API communication
        """
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two ModalClusters are equal."""
        if not isinstance(other, ModalCluster):
            return False
        return self.name() == other.name()

    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the job on this cluster if it exists.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            JobStatus if found, None otherwise
        """
        status = self._client.get_job(job_id)
        if status:
            status.cluster = self._name
        return status

    def get_jobs(self) -> list[JobStatus]:
        """Lists all jobs on this cluster.

        Returns:
            List of JobStatus objects for all tracked jobs
        """
        jobs = self._client.list_jobs()
        for job in jobs:
            job.cluster = self._name
        return jobs

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the specified job on this cluster.

        Args:
            job_id: The ID of the job to cancel

        Returns:
            Updated JobStatus

        Raises:
            RuntimeError: If the job is not found
        """
        status = self._client.cancel(job_id)
        if status is None:
            raise RuntimeError(f"Job {job_id} not found on Modal cluster {self._name}")
        status.cluster = self._name
        return status

    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on Modal.

        This method:
        1. Validates the job configuration
        2. Generates a Modal app from the JobConfig
        3. Submits the app to Modal
        4. Returns the initial job status

        Args:
            job: The job configuration to run

        Returns:
            JobStatus with the initial job state
        """
        job_copy = deepcopy(job)
        _validate_job_config(job_copy)

        if not job_copy.name:
            job_copy.name = f"oumi-modal-{uuid.uuid4().hex[:8]}"

        status = self._client.submit_job(job_copy)
        status.cluster = self._name
        return status

    def stop(self) -> None:
        """Stops the cluster.

        For Modal (serverless), this cancels all running jobs but doesn't
        actually stop any infrastructure since there's no persistent cluster.
        """
        logger.info(
            f"Stopping Modal cluster '{self._name}' - cancelling all running jobs"
        )
        for job in self.get_jobs():
            if not job.done:
                try:
                    self.cancel_job(job.id)
                except RuntimeError as e:
                    logger.warning(f"Failed to cancel job {job.id}: {e}")

    def down(self) -> None:
        """Tears down the cluster.

        For Modal (serverless), this is equivalent to stop() since there's
        no persistent infrastructure to tear down.
        """
        logger.info(
            f"Tearing down Modal cluster '{self._name}' - "
            "cancelling all running jobs (no persistent infrastructure)"
        )
        self.stop()

    def get_logs_stream(
        self, cluster_name: str, job_id: Optional[str] = None
    ) -> io.TextIOBase:
        """Gets a stream that tails the logs of the target job.

        Args:
            cluster_name: The name of the cluster (unused for Modal)
            job_id: The ID of the job to tail logs for. If unspecified,
                   returns logs for the most recent job.

        Returns:
            A text stream containing the job logs

        Raises:
            NotImplementedError: Log streaming is not yet fully implemented
        """
        # Get the job ID if not specified
        if job_id is None:
            jobs = self.get_jobs()
            if not jobs:
                raise RuntimeError("No jobs found on this cluster")
            # Get the most recent job (last in list)
            job_id = jobs[-1].id

        # Get logs from the client
        logs = self._client.get_logs(job_id)

        if logs is None:
            logs = f"No logs available for job {job_id}"

        # Return as a StringIO stream
        return io.StringIO(logs)
