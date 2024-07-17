from typing import List

from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.core.types.configs import JobConfig


class SkyCluster(BaseCluster):
    """A cluster implementation backed by Sky Pilot."""

    def __init__(self, name: str) -> None:
        """Initializes a new instance of the SkyCluster class."""
        self._name = name

    @property
    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self) -> JobStatus:
        """Gets the job's on this cluster."""
        raise NotImplementedError

    def get_jobs(self) -> List[JobStatus]:
        """List the jobs on this cluster."""
        raise NotImplementedError

    def stop_job(self, job_id: str) -> JobStatus:
        """Stop the specified job on this cluster."""
        # sky.cancel(job)
        raise NotImplementedError

    def run_job(self, job: JobConfig) -> JobStatus:
        """Run the specified job on this cluster."""
        raise NotImplementedError

    def down(self) -> None:
        """Tears down the current cluster."""
        raise NotImplementedError
