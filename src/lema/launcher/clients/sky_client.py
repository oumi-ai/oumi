from enum import Enum
from typing import Any, Dict, List, Optional

import sky
import sky.data

from lema.core.types.base_cluster import JobStatus
from lema.core.types.configs import JobConfig


def _get_sky_cloud_from_job(job: JobConfig) -> sky.clouds.Cloud:
    """Returns the sky.Cloud object from the JobConfig."""
    if job.resources.cloud == SkyClient.SupportedClouds.GCP.value:
        return sky.clouds.GCP()
    elif job.resources.cloud == SkyClient.SupportedClouds.RUNPOD.value:
        return sky.clouds.RunPod()
    elif job.resources.cloud == SkyClient.SupportedClouds.LAMBDA.value:
        return sky.clouds.Lambda()
    raise ValueError(f"Unsupported cloud: {job.resources.cloud}")


def _get_sky_storage_mounts_from_job(job: JobConfig) -> Dict[str, sky.data.Storage]:
    """Returns the sky.StorageMount objects from the JobConfig."""
    sky_mounts = {}
    for k, v in job.storage_mounts.items():
        storage_mount = sky.data.Storage(
            source=v.source,
        )
        sky_mounts[k] = storage_mount
    return sky_mounts


def _convert_job_to_task(job: JobConfig) -> sky.Task:
    """Converts a JobConfig to a sky.Task."""
    sky_cloud = _get_sky_cloud_from_job(job)
    resources = sky.Resources(
        cloud=sky_cloud,
        instance_type=job.resources.instance_type,
        cpus=job.resources.cpus,
        memory=job.resources.memory,
        accelerators=job.resources.accelerators,
        use_spot=job.resources.use_spot,
        region=job.resources.region,
        zone=job.resources.zone,
        disk_size=job.resources.disk_size,
        disk_tier=job.resources.disk_tier,
    )
    sky_task = sky.Task(
        name=job.name,
        setup=job.setup,
        run=job.run,
        envs=job.envs,
        workdir=job.working_dir,
        num_nodes=job.num_nodes,
    )
    sky_task.set_file_mounts(job.file_mounts)
    sky_task.set_storage_mounts(_get_sky_storage_mounts_from_job(job))
    sky_task.set_resources(resources)
    return sky_task


class SkyClient:
    """A wrapped client for communicating with Sky Pilot."""

    class SupportedClouds(Enum):
        """Enum representing the supported clouds."""

        GCP = "gcp"
        RUNPOD = "runpod"
        LAMBDA = "lambda"

    def launch(self, job: JobConfig, cluster_name: Optional[str] = None) -> JobStatus:
        """Creates a cluster and starts the provided Job.

        Args:
            job: The job to execute on the cluster.
            cluster_name: The name of the cluster to create.

        Returns:
            A JobStatus with only `id` and `cluster` populated.
        """
        job_id, resource_handle = sky.launch(
            _convert_job_to_task(job), cluster_name=cluster_name, detach_run=True
        )
        if job_id is None or resource_handle is None:
            raise RuntimeError("Failed to launch job.")
        return JobStatus(
            name="",
            id=str(job_id),
            cluster=resource_handle.cluster_name,
            status="",
            metadata="",
            done=False,
        )

    def status(self) -> List[Dict[str, Any]]:
        """Gets a list of cluster statuses.

        Returns:
            A list of dictionaries, each containing the status of a cluster.
        """
        return sky.status()

    def queue(self, cluster_name: str) -> List[dict]:
        """Gets the job queue of a cluster.

        Args:
            cluster_name: The name of the cluster to get the queue for.

        Returns:
            A list of dictionaries, each containing the metadata of a cluster.
        """
        return sky.queue(cluster_name)

    def cancel(self, cluster_name: str, job_id: str) -> None:
        """Gets the job queue of a cluster.

        Args:
            cluster_name: The name of the cluster to cancel the job on.
            job_id: The ID of the job to cancel.
        """
        sky.cancel(cluster_name, int(job_id))

    def exec(self, job: JobConfig, cluster_name: str) -> str:
        """Executes the specified job on the target cluster.

        Args:
            job: The job to execute.
            cluster_name: The name of the cluster to execute the job on.

        Returns:
            The ID of the job that was created.
        """
        job_id, _ = sky.exec(_convert_job_to_task(job), cluster_name, detach_run=True)
        if job_id is None:
            raise RuntimeError("Failed to submit job.")
        return str(job_id)

    def down(self, cluster_name: str) -> None:
        """Tears down the target cluster.

        Args:
            cluster_name: The name of the cluster to tear down.
        """
        sky.down(cluster_name)
