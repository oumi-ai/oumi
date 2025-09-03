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

from typing import Any, Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, JobState, JobStatus
from oumi.launcher.clients.sky_client import SkyClient, SkyLogStream


class SkyCluster(BaseCluster):
    """A cluster implementation backed by Sky Pilot."""

    def __init__(self, name: str, client: SkyClient) -> None:
        """Initializes a new instance of the SkyCluster class."""
        # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
        import sky.exceptions

        self._sky_exceptions = sky.exceptions
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two SkyClusters are equal."""
        if not isinstance(other, SkyCluster):
            return False
        return self.name() == other.name()

    def _get_job_state(self, sky_job: dict) -> JobState:
        """Gets the JobState from a sky job."""
        # See sky job states here:
        # https://skypilot.readthedocs.io/en/latest/reference/cli.html#sky-jobs-queue
        status = str(sky_job["status"])
        failed_states = {
            "JobStatus.FAILED",
            "JobStatus.FAILED_SETUP",
            "JobStatus.FAILED_NO_RESOURCE",
            "JobStatus.FAILED_CONTROLLER",
        }
        if status == "JobStatus.SUCCEEDED":
            return JobState.SUCCEEDED
        elif status == "JobStatus.CANCELLED":
            return JobState.CANCELLED
        elif status == "JobStatus.RUNNING":
            return JobState.RUNNING
        elif status in failed_states:
            return JobState.FAILED
        return JobState.PENDING

    def _convert_sky_job_to_status(self, sky_job: dict) -> JobStatus:
        """Converts a sky job to a JobStatus."""
        required_fields = ["job_id", "job_name", "status"]
        for field in required_fields:
            if field not in sky_job:
                raise ValueError(f"Missing required field: {field}")
        state = self._get_job_state(sky_job)
        return JobStatus(
            id=str(sky_job["job_id"]),
            name=str(sky_job["job_name"]),
            status=str(sky_job["status"]),
            cluster=self.name(),
            metadata="",
            done=state == JobState.SUCCEEDED
            or state == JobState.FAILED
            or state == JobState.CANCELLED,
            state=state,
        )

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
        try:
            return [
                self._convert_sky_job_to_status(job)
                for job in self._client.queue(self.name())
            ]
        except self._sky_exceptions.ClusterNotUpError:
            return []

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the specified job on this cluster."""
        self._client.cancel(self.name(), job_id)
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} not found.")
        return job

    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster."""
        job_id = self._client.exec(job, self.name())
        job_status = self.get_job(job_id)
        if job_status is None:
            raise RuntimeError(f"Job {job_id} not found after submission.")
        return job_status

    def stop(self) -> None:
        """Stops the current cluster."""
        self._client.stop(self.name())

    def down(self) -> None:
        """Tears down the current cluster."""
        self._client.down(self.name())

    def get_logs_stream(
        self, cluster_name: str, job_id: Optional[str] = None
    ) -> SkyLogStream:
        """Gets a stream that tails the logs of the target job.

        Args:
            cluster_name: The name of the cluster the job was run in.
            job_id: The ID of the job to tail the logs of.
        """
        return self._client.get_logs_stream(cluster_name, job_id)
