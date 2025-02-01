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
from dataclasses import dataclass

from oumi.core.configs import JobConfig


@dataclass
class JobStatus:
    """Dataclass to hold the status of a job."""

    #: The display name of the job.
    name: str

    #: The unique identifier of the job on the cluster
    id: str

    #: The status of the job.
    status: str

    #: The cluster to which the job belongs.
    cluster: str

    #: Miscellaneous metadata about the job.
    metadata: str

    #: A flag indicating whether the job is done.
    #: True only if the job is in a terminal state (e.g. completed, failed, or
    #: canceled).
    done: bool


class BaseCluster(ABC):
    """Base class for a compute cluster (job queue)."""

    @abstractmethod
    def name(self) -> str:
        """Gets the name of the cluster."""
        raise NotImplementedError

    @abstractmethod
    def get_job(self, job_id: str) -> JobStatus:
        """Gets the job on this cluster if it exists, else returns None."""
        raise NotImplementedError

    @abstractmethod
    def get_jobs(self) -> list[JobStatus]:
        """Lists the jobs on this cluster."""
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the specified job on this cluster."""
        raise NotImplementedError

    @abstractmethod
    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops the current cluster."""
        raise NotImplementedError

    @abstractmethod
    def down(self) -> None:
        """Tears down the current cluster."""
        raise NotImplementedError
