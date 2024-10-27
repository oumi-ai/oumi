from typing import Optional, TypeVar

import sky

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCloud, BaseCluster, JobStatus
from oumi.core.registry import register_cloud_builder
from oumi.launcher.clients.sky_client import SkyClient
from oumi.launcher.clusters.sky_cluster import SkyCluster

T = TypeVar("T")


class SkyCloud(BaseCloud):
    """A resource pool capable of creating clusters using Sky Pilot."""

    def __init__(self, cloud_name: str, client: SkyClient):
        """Initializes a new instance of the SkyCloud class."""
        self._cloud_name = cloud_name
        self._client = client

    def _get_clusters_by_class(self, cloud_class: type[T]) -> list[BaseCluster]:
        """Gets the appropriate clusters of type T."""
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
    return SkyCloud(SkyClient.SupportedClouds.RUNPOD.value, SkyClient())


@register_cloud_builder("gcp")
def gcp_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Google Cloud Platform."""
    return SkyCloud(SkyClient.SupportedClouds.GCP.value, SkyClient())


@register_cloud_builder("lambda")
def lambda_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Lambda."""
    return SkyCloud(SkyClient.SupportedClouds.LAMBDA.value, SkyClient())


@register_cloud_builder("aws")
def aws_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for AWS."""
    return SkyCloud(SkyClient.SupportedClouds.AWS.value, SkyClient())


@register_cloud_builder("azure")
def azure_cloud_builder() -> SkyCloud:
    """Builds a SkyCloud instance for Azure."""
    return SkyCloud(SkyClient.SupportedClouds.AZURE.value, SkyClient())
