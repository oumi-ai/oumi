from functools import reduce
from typing import Any, List, Optional

from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.core.types.configs import JobConfig
from lema.launcher.clients.polaris_client import PolarisClient
from lema.utils.logging import logger


def _create_job_script(job: JobConfig) -> str:
    """Creates a job script for the specified job.

    Args:
        job: The job to create a script for.

    Returns:
        The script as a string.
    """
    run_lines = job.run.strip().split("\n")
    # Find the last PBS instruction line. Return -1 if not found.
    last_pbs = (
        reduce(
            lambda acc, val: val[0] if val[1].startswith("#PBS") else acc,
            enumerate(run_lines),
            -1,
        )
        + 1
    )
    # Inject environment variables into the script after PBS instructions.
    env_lines = [f"export {key}={value}" for key, value in job.envs.items()]
    # Pad the environment variables with newlines.
    env_lines = [""] + env_lines + [""] if env_lines else []
    run_lines = run_lines[:last_pbs] + env_lines + run_lines[last_pbs:]
    # Always start the script with #!/bin/bash.
    script_prefix = "#!/bin/bash"
    if len(run_lines) > 0:
        if not run_lines[0].startswith("script_prefix"):
            run_lines.insert(0, script_prefix)
    # Join each line. Always end the script with a new line.
    return "\n".join(run_lines) + "\n"


def _validate_job_config(job: JobConfig) -> None:
    """Validates the provided job configuration.

    Args:
        job: The job to validate.
    """
    if not job.user:
        raise ValueError("User must be provided for Polaris jobs.")
    if not job.working_dir:
        raise ValueError("Working directory must be provided for Polaris jobs.")
    if not job.run:
        raise ValueError("Run script must be provided for Polaris jobs.")
    if job.num_nodes < 1:
        raise ValueError("Number of nodes must be at least 1.")
    if job.resources.cloud != "polaris":
        raise ValueError(
            f"`Resources.cloud` must be `polaris`. "
            f"Unsupported cloud: {job.resources.cloud}"
        )
    # Warn that other resource parameters are unused for Polaris.
    if job.resources.region:
        logger.warning("Region is unused for Polaris jobs.")
    if job.resources.zone:
        logger.warning("Zone is unused for Polaris jobs.")
    if job.resources.accelerators:
        logger.warning("Accelerators are unused for Polaris jobs.")
    if job.resources.cpus:
        logger.warning("CPUs are unused for Polaris jobs.")
    if job.resources.memory:
        logger.warning("Memory is unused for Polaris jobs.")
    if job.resources.instance_type:
        logger.warning("Instance type is unused for Polaris jobs.")
    if job.resources.disk_size:
        logger.warning("Disk size is unused for Polaris jobs.")
    if job.resources.instance_type:
        logger.warning("Instance type is unused for Polaris jobs.")
    # Warn that storage mounts are currently unsupported.
    if len(job.storage_mounts.items()) > 0:
        logger.warning("Storage mounts are currently unsupported for Polaris jobs.")


class PolarisCluster(BaseCluster):
    """A cluster implementation backed by Polaris."""

    def __init__(self, name: str, client: PolarisClient) -> None:
        """Initializes a new instance of the PolarisCluster class."""
        self._name = name
        self._queue = self._get_queue_from_name()
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two SkyClusters are equal."""
        if not isinstance(other, PolarisCluster):
            return False
        return self.name() == other.name()

    def _get_queue_from_name(self) -> PolarisClient.SupportedQueues:
        """Gets the queue from the provided name."""
        splits = self._name.split(".")
        if len(splits) < 2:
            raise ValueError(
                f"Invalid queue name: {self._name}. "
                "A queue name should be of the form: `queue.user`."
            )
        queue = splits[0].lower()
        if queue == PolarisClient.SupportedQueues.DEBUG.value:
            return PolarisClient.SupportedQueues.DEBUG
        if queue == PolarisClient.SupportedQueues.DEBUG_SCALING.value:
            return PolarisClient.SupportedQueues.DEBUG_SCALING
        if queue == PolarisClient.SupportedQueues.DEMAND.value:
            return PolarisClient.SupportedQueues.DEMAND
        if queue == PolarisClient.SupportedQueues.PREEMPTABLE.value:
            return PolarisClient.SupportedQueues.PREEMPTABLE
        if queue == PolarisClient.SupportedQueues.PROD.value:
            return PolarisClient.SupportedQueues.PROD
        raise ValueError(f"Unsupported queue: {queue}")

    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the jobs on this cluster if it exists, else returns None."""
        for job in self.get_jobs():
            if job.id == job_id:
                return job
        return None

    def get_jobs(self) -> List[JobStatus]:
        """Lists the jobs on this cluster."""
        return self._client.list_jobs(self._queue)

    def stop_job(self, job_id: str) -> JobStatus:
        """Stops the specified job on this cluster."""
        self._client.cancel(job_id, self._queue)
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} not found.")
        return job

    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster.

        For Polaris this method consists of 5 parts:
        1. Copy the working directory to /home/$USER/lema_launcher/$JOB_NAME.
        2. Check if there is a conda installation at /home/$USER/miniconda3/envs/lema.
            If not, install it.
        3. Copy all file mounts.
        4. Create a job script with all env vars, setup, and run commands.
        5. CD into the working directory and submit the job.

        Args:
            job: The job to run.

        Returns:
            The job status.
        """
        job_id = self._client.submit_job(
            "script_path",
            job.num_nodes,
            self._queue,
            job.name,
        )
        job_status = self.get_job(job_id)
        if job_status is None:
            raise RuntimeError(f"Job {job_id} not found after submission.")
        return job_status

    def down(self) -> None:
        """This is a no-op for Polaris clusters."""
        pass
