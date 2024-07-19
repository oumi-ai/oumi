from abc import ABC, abstractmethod
from typing import List, Optional

from lema.core.types.base_cluster import BaseCluster
from lema.core.types.configs import JobConfig


class BaseCloud(ABC):
    """Base class for resource pool capable of creating clusters."""

    @abstractmethod
    def up_cluster(self, job: JobConfig, name: Optional[str]) -> BaseCluster:
        """Creates a cluster and starts the provided Job."""
        raise NotImplementedError

    @abstractmethod
    def get_cluster(self, name: str) -> BaseCluster:
        """Gets the cluster with the specified name."""
        raise NotImplementedError

    @abstractmethod
    def list_clusters(self) -> List[BaseCluster]:
        """List the active clusters on this cloud."""
        raise NotImplementedError
