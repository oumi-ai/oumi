from abc import ABC, abstractmethod
from typing import List, Optional

from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.core.types.configs.job_config import JobConfig


class BaseCloud(ABC):
    """Base class for resource pool capable of creating clusters."""

    @abstractmethod
    def up_cluster(self, job: JobConfig, name: Optional[str]) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        raise NotImplementedError

    @abstractmethod
    def get_cluster(self, name: str) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    def list_clusters(self) -> List[BaseCluster]:
        """Lists the active clusters on this cloud."""
        raise NotImplementedError
