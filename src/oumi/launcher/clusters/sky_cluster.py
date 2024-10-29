from typing import Any, Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, JobStatus
from oumi.launcher.clients.sky_client import SkyClient


class SkyCluster(BaseCluster):
    """A cluster implementation backed by Sky Pilot."""

    def __init__(self, name: str, client: SkyClient) -> None:
        """Initializes a new instance of the SkyCluster class."""
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two SkyClusters are equal."""
        if not isinstance(other, SkyCluster):
            return False
        return self.name() == other.name()

    def _convert_sky_job_to_status(self, sky_job: dict) -> JobStatus:
        """Converts a sky job to a JobStatus."""
        required_fields = ["job_id", "job_name", "status"]
        for field in required_fields:
            if field not in sky_job:
                raise ValueError(f"Missing required field: {field}")
        return JobStatus(
            id=str(sky_job["job_id"]),
            name=str(sky_job["job_name"]),
            status=str(sky_job["status"]),
            cluster=self.name(),
            metadata="",
            # See sky job states here:
            # https://skypilot.readthedocs.io/en/latest/reference/cli.html#sky-jobs-queue
            done=str(sky_job["status"])
            not in [
                "JobStatus.PENDING",
                "JobStatus.SUBMITTED",
                "JobStatus.STARTING",
                "JobStatus.RUNNING",
                "JobStatus.RECOVERING",
                "JobStatus.CANCELLING",
            ],
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
        return [
            self._convert_sky_job_to_status(job)
            for job in self._client.queue(self.name())
        ]

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

    def down(self) -> None:
        """Tears down the current cluster."""
        self._client.down(self.name())
