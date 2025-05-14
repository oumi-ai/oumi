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

import os
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import JobStatus
from oumi.utils.logging import logger
from oumi.utils.str_utils import try_str_to_bool

if TYPE_CHECKING:
    import sky
    import sky.data


def _get_sky_cloud_from_job(job: JobConfig) -> "sky.clouds.Cloud":
    """Returns the sky.Cloud object from the JobConfig."""
    # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
    import sky

    if job.resources.cloud == SkyClient.SupportedClouds.GCP.value:
        return sky.clouds.GCP()
    elif job.resources.cloud == SkyClient.SupportedClouds.RUNPOD.value:
        return sky.clouds.RunPod()
    elif job.resources.cloud == SkyClient.SupportedClouds.LAMBDA.value:
        return sky.clouds.Lambda()
    elif job.resources.cloud == SkyClient.SupportedClouds.AWS.value:
        return sky.clouds.AWS()
    elif job.resources.cloud == SkyClient.SupportedClouds.AZURE.value:
        return sky.clouds.Azure()
    raise ValueError(f"Unsupported cloud: {job.resources.cloud}")


def _get_sky_storage_mounts_from_job(job: JobConfig) -> dict[str, "sky.data.Storage"]:
    """Returns the sky.StorageMount objects from the JobConfig."""
    # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
    import sky.data

    sky_mounts = {}
    for k, v in job.storage_mounts.items():
        storage_mount = sky.data.Storage(
            source=v.source,
        )
        sky_mounts[k] = storage_mount
    return sky_mounts


def _get_use_spot_vm_override() -> Optional[bool]:
    """Determines whether to override `use_spot_vm` setting based on OUMI_USE_SPOT_VM.

    Fetches the override value from the OUMI_USE_SPOT_VM environment variable
    if specified.

    Returns:
        The override value if specified, or `None`.
    """
    _ENV_VAR_NAME = "OUMI_USE_SPOT_VM"
    s = os.environ.get(_ENV_VAR_NAME, "")
    mode = s.lower().replace("-", "").replace("_", "").strip()
    if not mode or mode in ("config",):
        return None
    bool_result = try_str_to_bool(mode)
    if bool_result is not None:
        return bool_result
    if mode in ("spot", "preemptible", "preemptable"):
        return True
    elif mode in ("nonspot", "nonpreemptible", "nonpreemptable"):
        return False
    raise ValueError(f"{_ENV_VAR_NAME} has unsupported value: '{s}'.")


def _convert_job_to_task(job: JobConfig) -> "sky.Task":
    """Converts a JobConfig to a sky.Task."""
    # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
    import sky

    sky_cloud = _get_sky_cloud_from_job(job)
    use_spot_vm = _get_use_spot_vm_override()
    if use_spot_vm is None:
        use_spot_vm = job.resources.use_spot
    elif use_spot_vm != job.resources.use_spot:
        logger.info(f"Set use_spot={use_spot_vm} based on 'OUMI_USE_SPOT_VM' override.")

    resources = sky.Resources(
        cloud=sky_cloud,
        instance_type=job.resources.instance_type,
        cpus=job.resources.cpus,
        memory=job.resources.memory,
        accelerators=job.resources.accelerators,
        use_spot=use_spot_vm,
        region=job.resources.region,
        zone=job.resources.zone,
        disk_size=job.resources.disk_size,
        disk_tier=job.resources.disk_tier,
        image_id=job.resources.image_id,
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

        AWS = "aws"
        AZURE = "azure"
        GCP = "gcp"
        RUNPOD = "runpod"
        LAMBDA = "lambda"

    def __init__(self):
        """Initializes a new instance of the SkyClient class."""
        # Delay sky import: https://github.com/oumi-ai/oumi/issues/1605
        import sky

        self._sky_lib = sky

    def launch(
        self, job: JobConfig, cluster_name: Optional[str] = None, **kwargs
    ) -> JobStatus:
        """Creates a cluster and starts the provided Job.

        Args:
            job: The job to execute on the cluster.
            cluster_name: The name of the cluster to create.
            kwargs: Additional arguments to pass to the Sky Pilot client.

        Returns:
            A JobStatus with only `id` and `cluster` populated.
        """
        sky_cloud = _get_sky_cloud_from_job(job)
        sky_task = _convert_job_to_task(job)

        # Set autostop if supported by the cloud, defaulting to 60 minutes if not
        # specified by the user. Currently, Lambda and RunPod do not support autostop.
        idle_minutes_to_autostop = None
        try:
            sky_resources = next(iter(sky_task.resources))
            # This will raise an exception if the cloud does not support stopping.
            sky_cloud.check_features_are_supported(
                sky_resources,
                requested_features={
                    self._sky_lib.clouds.CloudImplementationFeatures.STOP
                },
            )
            autostop_kw = "idle_minutes_to_autostop"
            # Default to 60 minutes.
            idle_minutes_to_autostop = 60
            if autostop_kw in kwargs:
                idle_minutes_to_autostop = kwargs.get(autostop_kw)
                logger.info(f"Setting autostop to {idle_minutes_to_autostop} minutes.")
            else:
                logger.info(
                    "No idle_minutes_to_autostop provided. "
                    f"Defaulting to {idle_minutes_to_autostop} minutes."
                )
        except Exception:
            logger.info(
                f"{sky_cloud._REPR} does not support stopping clusters. "
                "Will not set autostop."
            )

        job_id, resource_handle = self._sky_lib.launch(
            sky_task,
            cluster_name=cluster_name,
            detach_run=True,
            idle_minutes_to_autostop=idle_minutes_to_autostop,
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

    def status(self) -> list[dict[str, Any]]:
        """Gets a list of cluster statuses.

        Returns:
            A list of dictionaries, each containing the status of a cluster.
        """
        return self._sky_lib.status()

    def queue(self, cluster_name: str) -> list[dict]:
        """Gets the job queue of a cluster.

        Args:
            cluster_name: The name of the cluster to get the queue for.

        Returns:
            A list of dictionaries, each containing the metadata of a cluster.
        """
        return self._sky_lib.queue(cluster_name)

    def cancel(self, cluster_name: str, job_id: str) -> None:
        """Gets the job queue of a cluster.

        Args:
            cluster_name: The name of the cluster to cancel the job on.
            job_id: The ID of the job to cancel.
        """
        self._sky_lib.cancel(cluster_name, int(job_id))

    def exec(self, job: JobConfig, cluster_name: str) -> str:
        """Executes the specified job on the target cluster.

        Args:
            job: The job to execute.
            cluster_name: The name of the cluster to execute the job on.

        Returns:
            The ID of the job that was created.
        """
        job_id, _ = self._sky_lib.exec(
            _convert_job_to_task(job), cluster_name, detach_run=True
        )
        if job_id is None:
            raise RuntimeError("Failed to submit job.")
        return str(job_id)

    def stop(self, cluster_name: str) -> None:
        """Stops the target cluster.

        Args:
            cluster_name: The name of the cluster to stop.
        """
        self._sky_lib.stop(cluster_name)

    def down(self, cluster_name: str) -> None:
        """Tears down the target cluster.

        Args:
            cluster_name: The name of the cluster to tear down.
        """
        self._sky_lib.down(cluster_name)
