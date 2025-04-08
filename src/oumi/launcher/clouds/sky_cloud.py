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

from typing import Optional, TypeVar

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import register_cloud_builder
from oumi.launcher.clients.sky_client import SkyClient
from oumi.launcher.clusters.sky_cluster import SkyCluster

T = TypeVar("T")


class SkyCloud(BaseCloud):
    """A resource pool capable of creating clusters using Sky Pilot."""

    @property
    def _client(self) -> SkyClient:
        """Returns the SkyClient instance."""
        # Instantiating a SkyClient imports sky.
        # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
        if not self._sky_client:
            self._sky_client = SkyClient()
        return self._sky_client

    def __init__(self, cloud_name: str):
        """Initializes a new instance of the SkyCloud class."""
        self._cloud_name = cloud_name
        self._sky_client: Optional[SkyClient] = None

    def _get_clusters_by_class(self, cloud_class: type[T]) -> list[BaseCluster]:
        """Gets the appropriate clusters of type T."""
        # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
        import sky

        return [
            SkyCluster(cluster["name"], self._client)
            for cluster in self._client.status()
            if (
                isinstance(cluster["handle"].launched_resources.cloud, cloud_class)
                and cluster["status"] == sky.ClusterStatus.UP
            )
        ]

    def up_cluster(self, job: JobConfig, name: Optional[str], **kwargs) -> JobStatus:
        """Creates a cluster and starts the provided Job."""
        job_status = self._client.launch(job, name, **kwargs)
        cluster = self.get_cluster(job_status.cluster)
        if not cluster:
            raise RuntimeError(f"Cluster {job_status.cluster} not found.")
        return cluster.get_job(job_status.id)

    def get_cluster(self, name) -> Optional[BaseCluster]:
        """Gets the cluster with the specified name, or None if not found."""
        clusters = self.list_clusters()
        for cluster in clusters:
            if cluster.name() == name:
                return cluster
        return None

    def list_clusters(self) -> list[BaseCluster]:
        """Lists the active clusters on this cloud."""
        # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
        import sky

        if self._cloud_name == SkyClient.SupportedClouds.GCP.value:
            return self._get_clusters_by_class(sky.clouds.GCP)
        elif self._cloud_name == SkyClient.SupportedClouds.RUNPOD.value:
            return self._get_clusters_by_class(sky.clouds.RunPod)
        elif self._cloud_name == SkyClient.SupportedClouds.LAMBDA.value:
            return self._get_clusters_by_class(sky.clouds.Lambda)
        elif self._cloud_name == SkyClient.SupportedClouds.AWS.value:
            return self._get_clusters_by_class(sky.clouds.AWS)
        elif self._cloud_name == SkyClient.SupportedClouds.AZURE.value:
            return self._get_clusters_by_class(sky.clouds.Azure)
        raise ValueError(f"Unsupported cloud: {self._cloud_name}")


@register_cloud_builder("runpod")
def runpod_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for runpod."""
    return SkyCloud(SkyClient.SupportedClouds.RUNPOD.value)


@register_cloud_builder("gcp")
def gcp_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Google Cloud Platform."""
    return SkyCloud(SkyClient.SupportedClouds.GCP.value)


@register_cloud_builder("lambda")
def lambda_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Lambda."""
    return SkyCloud(SkyClient.SupportedClouds.LAMBDA.value)


@register_cloud_builder("aws")
def aws_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for AWS."""
    return SkyCloud(SkyClient.SupportedClouds.AWS.value)


@register_cloud_builder("azure")
def azure_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Azure."""
    return SkyCloud(SkyClient.SupportedClouds.AZURE.value)
