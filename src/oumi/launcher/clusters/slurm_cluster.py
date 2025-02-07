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

import uuid
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import BaseCluster, JobStatus
from oumi.launcher.clients.slurm_client import SlurmClient
from oumi.utils.logging import logger


def _format_date(date: datetime) -> str:
    """Formats the provided date as a string.

    Args:
        date: The date to format.

    Returns:
        The formatted date.
    """
    return date.strftime("%Y%m%d_%H%M%S%f")


def _last_sbatch_line(script: list[str]) -> int:
    """Finds the last SBATCH instruction line in the script.

    Args:
        script: The lines of the script.

    Returns:
        The index of the last SBATCH instruction line. -1 if not found.
    """
    return reduce(
        lambda acc, val: val[0] if val[1].startswith("#SBATCH") else acc,
        enumerate(script),
        -1,
    )


def _create_job_script(job: JobConfig) -> str:
    """Creates a job script for the specified job.

    Args:
        job: The job to create a script for.

    Returns:
        The script as a string.
    """
    setup_lines = [] if not job.setup else job.setup.strip().split("\n")
    run_lines = job.run.strip().split("\n")
    # Find the last SBATCH instruction line.
    last_run_sbatch = _last_sbatch_line(run_lines) + 1
    last_setup_sbatch = _last_sbatch_line(setup_lines) + 1
    # Inject environment variables into the script after SBATCH instructions.
    env_lines = [f"export {key}={value}" for key, value in job.envs.items()]
    # Pad the environment variables with newlines.
    env_lines = [""] + env_lines + [""] if env_lines else []
    # Generate the job script.
    # The script should have the following structure:
    # 1. SBATCH instructions from Setup and Run commands (in that order).
    # 2. Environment variables.
    # 3. Setup commands.
    # 4. Run commands.
    output_lines = (
        setup_lines[:last_setup_sbatch]
        + run_lines[:last_run_sbatch]
        + env_lines
        + setup_lines[last_setup_sbatch:]
        + run_lines[last_run_sbatch:]
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
        raise ValueError("User must be provided for Slurm jobs.")
    if not job.working_dir:
        raise ValueError("Working directory must be provided for Slurm jobs.")
    if not job.run:
        raise ValueError("Run script must be provided for Slurm jobs.")
    if job.num_nodes < 1:
        raise ValueError("Number of nodes must be at least 1.")
    if job.resources.cloud != "slurm":
        raise ValueError(
            f"`Resources.cloud` must be `slurm`. "
            f"Unsupported cloud: {job.resources.cloud}"
        )
    # Warn that other resource parameters are unused for Slurm.
    if job.resources.region:
        logger.warning("Region is unused for Slurm jobs.")
    if job.resources.zone:
        logger.warning("Zone is unused for Slurm jobs.")
    if job.resources.accelerators:
        logger.warning("Accelerators are unused for Slurm jobs.")
    if job.resources.cpus:
        logger.warning("CPUs are unused for Slurm jobs.")
    if job.resources.memory:
        logger.warning("Memory is unused for Slurm jobs.")
    if job.resources.instance_type:
        logger.warning("Instance type is unused for Slurm jobs.")
    if job.resources.disk_size:
        logger.warning("Disk size is unused for Slurm jobs.")
    if job.resources.instance_type:
        logger.warning("Instance type is unused for Slurm jobs.")
    # Warn that storage mounts are currently unsupported.
    if len(job.storage_mounts.items()) > 0:
        logger.warning("Storage mounts are currently unsupported for Slurm jobs.")


class SlurmCluster(BaseCluster):
    """A cluster implementation backed by a Slurm scheduler."""

    def __init__(self, name: str, client: SlurmClient) -> None:
        """Initializes a new instance of the SlurmCluster class."""
        self._name = name
        self._client = client

    def __eq__(self, other: Any) -> bool:
        """Checks if two SlurmClusters are equal."""
        if not isinstance(other, SlurmCluster):
            return False
        return self.name() == other.name()

    def name(self) -> str:
        """Gets the name of the cluster."""
        return self._name

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the jobs on this cluster if it exists, else returns None."""
        for job in self.get_jobs():
            if job.id == job_id:
                return job
        return None

    def get_jobs(self) -> list[JobStatus]:
        """Lists the jobs on this cluster."""
        jobs = self._client.list_jobs()
        for job in jobs:
            job.cluster = self._name
        return jobs

    def cancel_job(self, job_id: str) -> JobStatus:
        """Cancels the specified job on this cluster."""
        self._client.cancel(job_id)
        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"Job {job_id} not found.")
        return job

    def run_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster.

        For Slurm this method consists of 5 parts:

        1. Copy the working directory to ~/oumi_launcher/$JOB_NAME.
        2. Check if there is a conda installation at /home/$USER/miniconda3/envs/oumi.
           If not, install it.
        3. Copy all file mounts.
        4. Create a job script with all env vars, setup, and run commands.
        5. CD into the working directory and submit the job.

        Args:
            job: The job to run.

        Returns:
            JobStatus: The job status.
        """
        _validate_job_config(job)
        job_name = job.name or uuid.uuid1().hex
        submission_time = _format_date(datetime.now())
        remote_working_dir = Path(f"~/oumi_launcher/{submission_time}")
        # Copy the working directory to ~/oumi_launcher/...
        self._client.put_recursive(job.working_dir, str(remote_working_dir))
        # Copy all file mounts.
        for remote_path, local_path in job.file_mounts.items():
            self._client.put_recursive(local_path, remote_path)
        # Create the job script by merging envs, setup, and run commands.
        job_script = _create_job_script(job)
        script_path = remote_working_dir / "oumi_job.sh"
        self._client.put(job_script, str(script_path))
        # Set the proper CHMOD permissions.
        self._client.run_commands([f"chmod +x {script_path}"])
        # Submit the job.
        job_id = self._client.submit_job(
            str(script_path),
            str(remote_working_dir),
            job.num_nodes,
            job_name,
        )
        job_status = self.get_job(job_id)
        if job_status is None:
            raise RuntimeError(f"Job {job_id} not found after submission.")
        return job_status

    def stop(self) -> None:
        """This is a no-op for Slurm clusters."""
        pass

    def down(self) -> None:
        """This is a no-op for Slurm clusters."""
        pass
