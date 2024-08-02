import re
import uuid
from functools import reduce
from pathlib import Path
from typing import Any, List, Optional

from lema.core.types.base_cluster import BaseCluster, JobStatus
from lema.core.types.configs import JobConfig
from lema.launcher.clients.polaris_client import PolarisClient
from lema.utils.logging import logger


def _last_pbs_line(script: List[str]) -> int:
    """Finds the last PBS instruction line in the script.

    Args:
        script: The lines of the script.

    Returns:
        The index of the last PBS instruction line. -1 if not found.
    """
    return reduce(
        lambda acc, val: val[0] if val[1].startswith("#PBS") else acc,
        enumerate(script),
        -1,
    )


def _get_logging_directories(script: str) -> List[str]:
    """Gets the logging directories from the script.

    Parses the provided script for commands starting with `#PBS -o`, `#PBS -e`,
    `#PBS -oe`, `#PBS -eo`, or `#PBS -doe`.

    Args:
        script: The script to extract logging directories from.

    Returns:
        A list of logging directories.
    """
    logging_pattern = r"#PBS\s+-[oe|eo|doe|o|e]\s+(.*)"
    logging_dirs = []
    for line in script.split("\n"):
        match = re.match(logging_pattern, line.strip())
        if match:
            logging_dirs.append(match.group(1))
    return logging_dirs


def _create_job_script(job: JobConfig) -> str:
    """Creates a job script for the specified job.

    Args:
        job: The job to create a script for.

    Returns:
        The script as a string.
    """
    setup_lines = [] if not job.setup else job.setup.strip().split("\n")
    run_lines = job.run.strip().split("\n")
    # Find the last PBS instruction line.
    last_run_pbs = _last_pbs_line(run_lines) + 1
    last_setup_pbs = _last_pbs_line(setup_lines) + 1
    # Inject environment variables into the script after PBS instructions.
    env_lines = [f"export {key}={value}" for key, value in job.envs.items()]
    # Pad the environment variables with newlines.
    env_lines = [""] + env_lines + [""] if env_lines else []
    # Generate the job script.
    # The script should have the following structure:
    # 1. PBS instructions from Setup and Run commands (in that order).
    # 2. Environment variables.
    # 3. Setup commands.
    # 4. Run commands.
    output_lines = (
        setup_lines[:last_setup_pbs]
        + run_lines[:last_run_pbs]
        + env_lines
        + setup_lines[last_setup_pbs:]
        + run_lines[last_run_pbs:]
    )
    # Always start the script with #!/bin/bash.
    script_prefix = "#!/bin/bash"
    if len(output_lines) > 0:
        if not output_lines[0].startswith("script_prefix"):
            output_lines.insert(0, script_prefix)
    # Join each line. Always end the script with a new line.
    return "\n".join(output_lines) + "\n"


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
        jobs = self._client.list_jobs(self._queue)
        for job in jobs:
            job.cluster = self._name
        return jobs

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
        _validate_job_config(job)
        job_name = job.name or uuid.uuid1().hex
        user = str(job.user)
        remote_working_dir = Path(f"/home/{user}/lema_launcher/{job_name}")
        # Copy the working directory to Polaris /home/ system.
        self._client.run_commands([f"mkdir -p {remote_working_dir}"])
        self._client.put_r(job.working_dir, str(remote_working_dir))
        # Check if lema is installed in a conda env. If not, install it.
        lema_env_path = Path("/home/$USER/miniconda3/envs/lema")
        should_create_env = True
        try:
            self._client.run_commands([f"test -d {lema_env_path}"])
            should_create_env = False
        except RuntimeError:
            pass
        if should_create_env:
            conda_cmds = [
                "module use /soft/modulefiles",
                "module load conda",
                'echo "Creating LeMa Conda environment... ---------------------------"',
                f"conda create -y python=3.11 --prefix {lema_env_path}",
                f"conda activate {lema_env_path}",
                "pip install flash-attn --no-build-isolation",
            ]
            # Run all commands in the same context.
            self._client.run_commands([" && ".join(conda_cmds)])
            # Install LeMa requirements.
            install_cmds = [
                f"cd {remote_working_dir}",
                "module use /soft/modulefiles",
                "module load conda",
                f"conda activate {lema_env_path}",
                'echo "Installing packages... ---------------------------------------"',
                "pip install -e '.[train]'",
            ]
            self._client.run_commands([" && ".join(install_cmds)])
        # Copy all file mounts.
        for remote_path, local_path in job.file_mounts.items():
            self._client.put_r(local_path, remote_path)
        # Create the job script by merging envs, setup, and run commands.
        job_script = _create_job_script(job)
        script_path = remote_working_dir / "lema_job.sh"
        self._client.put(job_script, str(script_path))
        # Set the proper CHMOD permissions.
        self._client.run_commands([f"chmod a+x {script_path}"])
        # Set up logging directories.
        logging_dirs = _get_logging_directories(job_script)
        loggin_dir_cmds = [f"mkdir -p {log_dir}" for log_dir in logging_dirs]
        if loggin_dir_cmds:
            self._client.run_commands(loggin_dir_cmds)
        # Submit the job.
        job_id = self._client.submit_job(
            str(script_path),
            job.num_nodes,
            self._queue,
            job_name,
        )
        job_status = self.get_job(job_id)
        if job_status is None:
            raise RuntimeError(f"Job {job_id} not found after submission.")
        return job_status

    def down(self) -> None:
        """This is a no-op for Polaris clusters."""
        pass
