from typing import Dict

import sky
import sky.data

from lema.core.types.configs import JobConfig
from lema.core.types.params.node_params import SupportedCloud


def get_sky_cloud_from_job(job: JobConfig) -> sky.clouds.Cloud:
    """Returns the sky.Cloud object from the JobConfig."""
    if job.resources.cloud == SupportedCloud.GCP:
        return sky.clouds.GCP()
    elif job.resources.cloud == SupportedCloud.RUNPOD:
        return sky.clouds.RunPod()
    raise ValueError(f"Unsupported cloud: {job.resources.cloud}")


def get_sky_strorage_mounts_from_job(job: JobConfig) -> Dict[str, sky.data.Storage]:
    """Returns the sky.StorageMount objects from the JobConfig."""
    sky_mounts = {}
    for k, v in job.storage_mounts.items():
        storage_mount = sky.data.Storage(
            source=v.source,
        )
        sky_mounts[k] = storage_mount
    return sky_mounts


def convert_job_to_task(job: JobConfig) -> sky.Task:
    """Converts a JobConfig to a sky.Task."""
    sky_cloud = get_sky_cloud_from_job(job)
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
        disk_tier=job.resources.disk_tier.value,
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
    sky_task.set_storage_mounts(get_sky_strorage_mounts_from_job(job))
    sky_task.set_resources(resources)
    return sky_task
