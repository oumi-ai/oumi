from typing import List, Optional

import sky

import lema.launcher.utils.sky_utils as sky_utils
from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.core.types.configs import JobConfig


class SkyCluster(BaseCluster):
    """A cluster implementation backed by Sky Pilot."""

    def __init__(self, name: str) -> None:
        """Initializes a new instance of the SkyCluster class."""
        self._name = name

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
        )

    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the job on this cluster."""
        for job in self.get_jobs():
            if job.id == job_id:
                return job
        return None

    def get_jobs(self) -> List[JobStatus]:
        """List the jobs on this cluster."""
        return [self._convert_sky_job_to_status(job) for job in sky.queue(self.name())]

    def stop_job(self, job_id: str) -> JobStatus:
        """Stop the specified job on this cluster."""
        sky.cancel(self.name(), int(job_id))
        job = self.get_job(job_id)
        if job is None:
            raise ValueError(f"Job {job_id} not found.")
        return job

    def run_job(self, job: JobConfig) -> JobStatus:
        """Run the specified job on this cluster."""
        sky_job = sky_utils.convert_job_to_task(job)
        job_id, _ = sky.exec(sky_job, self.name())
        if job_id is None:
            raise ValueError("Failed to submit job.")
        job_status = self.get_job(str(job_id))
        if job_status is None:
            raise ValueError(f"Job {job_id} not found after submission.")
        return job_status

    def down(self) -> None:
        """Tears down the current cluster."""
        sky.down(self.name())
