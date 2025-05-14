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

from abc import ABC, abstractmethod
from typing import Optional

from oumi.core.configs.job_config import JobConfig
from oumi.core.launcher.base_cluster import BaseCluster, JobStatus


class BaseCloud(ABC):
    """Base class for resource pool capable of creating clusters."""

    @abstractmethod
    def up_cluster(self, job: JobConfig, name: Optional[str], **kwargs) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        raise NotImplementedError

    @abstractmethod
    def get_cluster(self, name: str) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    def list_clusters(self) -> list[BaseCluster]:
        """Lists the active clusters on this cloud."""
        raise NotImplementedError
