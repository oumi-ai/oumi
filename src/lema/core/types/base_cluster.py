from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from lema.core.types.configs import JobConfig


@dataclass
class JobStatus:
    """Dataclass to hold the status of a job."""

    # The display name of the job.
    name: str

    # The unique identifier of the job on the cluster
    id: str

    # The status of the job.
    status: str

    # The cluster to which the job belongs.
    cluster: str

    # Miscellaneous metadata about the job.
    metadata: str


class BaseCluster(ABC):
    """Base class for a compute cluster (job queue)."""

    @abstractmethod
    def name(self) -> str:
        """Gets the name of the cluster."""
        raise NotImplementedError

    @abstractmethod
    def get_job(self) -> JobStatus:
        """Gets the jobs on this cluster if it exists, else returns None."""
        raise NotImplementedError

    @abstractmethod
    def get_jobs(self) -> List[JobStatus]:
        """Lists the jobs on this cluster."""
        raise NotImplementedError

    @abstractmethod
    def stop_job(self, job: str) -> JobStatus:
        """Stops the specified job on this cluster."""
        raise NotImplementedError

    @abstractmethod
    def run_job(self, job: JobConfig) -> Optional[JobStatus]:
        """Runs the specified job on this cluster."""
        raise NotImplementedError

    @abstractmethod
    def down(self) -> None:
        """Tears down the current cluster."""
        raise NotImplementedError
