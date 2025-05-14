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

import functools
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from oumi.core.launcher import JobStatus
from oumi.utils.logging import logger

_CTRL_PATH = "-S ~/.ssh/control-%h-%p-%r"


class _SlurmAuthException(Exception):
    pass


def _check_connection(user: str, slurm_host: str) -> None:
    """Checks if the connection is still open."""
    ssh_cmd = f"ssh {_CTRL_PATH} -O check {user}@{slurm_host}"
    error_msg = ""
    try:
        child = subprocess.run(
            ssh_cmd,
            shell=True,
            capture_output=True,
            timeout=10,
        )
        error_msg = child.stderr.decode("utf-8")
    except subprocess.TimeoutExpired:
        raise _SlurmAuthException("Timeout while checking connection.")
    if child.returncode == 0:
        return
    if error_msg:
        logger.error(f"Error checking connection: {error_msg}")
        error_msg = f" Error: {error_msg}"
    raise _SlurmAuthException("Connection to Slurm host is closed." + error_msg)


def _is_job_done(job_state: str) -> bool:
    """Determines if a job is done based on its state.

    See https://slurm.schedmd.com/job_state_codes.html for more details.

    Args:
        job_state: The state of the job.

    Returns:
        True if the job is done, False otherwise.
    """
    terminal_states = {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "LAUNCH_FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
        "SUSPENDED",
        "STOPPED",
    }
    return job_state in terminal_states


def _split_status_line(
    line: str, column_lengths: list[int], cluster_name: str, metadata: str
) -> JobStatus:
    """Splits a status line into a JobStatus object.

    The expected order of job fields is:
    0. Job ID
    1. Job Name
    2. User
    3. Job State
    4. Job State Reason

    Sample status report:
    ID      NAME   USER     STATE REASON
    ----- ------ ------ --------- ------
    1     my_job   user COMPLETED    0:0
    2       job2          RUNNING

    Args:
        line: The line to split.
        column_lengths: The lengths in chars of each column in the line.
        cluster_name: The name of the cluster the job is running on.
        metadata: Additional metadata to attach to the job status.

    Returns:
        A JobStatus object.
    """
    if len(column_lengths) != 5:
        raise ValueError(
            f"Expected 5 fields, but found {len(column_lengths)}. Invalid line: {line}."
        )
    fields = []
    # Note: We can't use a simple split() here because empty fields are allowed.
    for i in range(len(column_lengths)):
        start = sum(column_lengths[:i]) + i
        end = start + column_lengths[i]
        fields.append(line[start:end].strip())
    return JobStatus(
        id=fields[0],
        name=fields[1],
        # JobState can have additional information. The primary state is the first word.
        status=fields[3].split(" ")[0],
        cluster=cluster_name,
        metadata=metadata,
        done=_is_job_done(fields[3]),
    )


def _compute_duration_debug_str(start_time: float) -> str:
    duration_sec = time.perf_counter() - start_time
    return f"Duration: {duration_sec:.2f} sec"


@dataclass
class SlurmResponse:
    """A response from Slurm."""

    stdout: str
    stderr: str
    exit_code: int


def retry_auth(user_function: Callable) -> Callable:
    """Decorator to ensure auth is fresh before calling a function."""

    @functools.wraps(user_function)
    def wrapper(self, *args, **kwargs):
        self._refresh_creds()
        return user_function(self, *args, **kwargs)

    return wrapper


class SlurmClient:
    """A client for communicating with a Slurm cluster."""

    def __init__(self, user: str, slurm_host: str, cluster_name: str):
        """Initializes a new instance of the SlurmClient class.

        Args:
            user: The user to act as.
            slurm_host: The Slurm Head Node to connect to.
            cluster_name: The name of the cluster this client communicates with.
        """
        self._user = user
        self._slurm_host = slurm_host
        self._cluster_name = cluster_name
        self._refresh_creds()

    def _refresh_creds(self):
        """Refreshes the credentials for the client."""
        try:
            _check_connection(self._user, self._slurm_host)
            # We have fresh credentials, so we return early.
            return
        except _SlurmAuthException:
            logger.warning("No connection found. Establishing a new SSH tunnel...")
        ssh_cmd = (
            f'ssh -f -N -M {_CTRL_PATH} -o "ControlPersist 4h" '
            f"{self._user}@{self._slurm_host}"
        )
        child = subprocess.run(
            ssh_cmd,
            shell=True,
            capture_output=True,
            timeout=180,  # time in seconds
        )
        if child.returncode != 0:
            output = child.stderr.decode("utf-8")
            logger.error(f"Credential error: {output}")
            raise RuntimeError(
                "Failed to refresh Slurm credentials "
                f"for {self._user}@{self._slurm_host}."
            )
        return SlurmResponse(
            stdout=child.stdout.decode("utf-8"),
            stderr=child.stderr.decode("utf-8"),
            exit_code=child.returncode,
        )

    @staticmethod
    def get_active_users(slurm_host: str) -> list[str]:
        """Gets the list of users with an open SSH tunnel to a Slurm cluster.

        Returns:
            A list of users.
        """
        # List all active users with an open SSH tunnel to the Slurm head node.
        command = f"ls ~/.ssh/ | egrep 'control-{slurm_host}-.*-.*'"
        result = subprocess.run(command, shell=True, capture_output=True)
        if result.returncode != 0:
            return []
        # Sample Pattern:
        # control-HOSTNAME-22-taenin
        ssh_tunnel_pattern = rf"control-{slurm_host}-[^-]*-(.*)"
        lines = result.stdout.decode("utf-8").strip().split("\n")
        users = set()
        for line in lines:
            match = re.match(ssh_tunnel_pattern, line.strip())
            if match:
                users.add(match.group(1))
        return list(users)

    def _parse_job_id(self, sbatch_output: str) -> str:
        """Parses the job ID from the result of sbatch.

        From the Slurm MAN page:
        Outputs only the job id number and the cluster name if present.
        The values are separated by a semicolon. Errors will still be displayed.

        Args:
            sbatch_output: The result of sbatch

        Returns:
            The job ID.
        """
        split_job = sbatch_output.strip().split(";")
        if len(split_job) > 2:
            raise ValueError(f"Unexpected output from sbatch: {sbatch_output}")
        return split_job[0]

    @retry_auth
    def run_commands(self, commands: list[str]) -> SlurmResponse:
        """Runs the provided commands in a single SSH command.

        Args:
            commands: The commands to run.
        """
        ssh_cmd = f"ssh {_CTRL_PATH} {self._user}@{self._slurm_host}  << 'EOF'"
        eof_suffix = "EOF"
        new_cmd = "\n".join([ssh_cmd, *commands, eof_suffix])
        start_time: float = time.perf_counter()
        try:
            logger.debug(f"Running commands:\n{new_cmd}")
            child = subprocess.run(
                new_cmd,
                shell=True,
                capture_output=True,
                timeout=180,  # time in seconds
            )
            duration_str = _compute_duration_debug_str(start_time)
            if child.returncode == 0:
                logger.debug(f"Commands successfully finished! {duration_str}")
            else:
                logger.error(
                    f"Commands failed with code: {child.returncode}! {duration_str}"
                )
            return SlurmResponse(
                stdout=child.stdout.decode("utf-8"),
                stderr=child.stderr.decode("utf-8"),
                exit_code=child.returncode,
            )
        except subprocess.TimeoutExpired:
            duration_str = _compute_duration_debug_str(start_time)
            logger.exception(f"Commands timed out ({duration_str})! {new_cmd}")
            return SlurmResponse(
                stdout="",
                stderr=f"Timeout while running command: {new_cmd}",
                exit_code=1,
            )
        except Exception:
            duration_str = _compute_duration_debug_str(start_time)
            logger.exception(f"Command failed ({duration_str})! {new_cmd}")
            raise

    def submit_job(
        self,
        job_path: str,
        working_dir: str,
        node_count: int,
        name: Optional[str],
    ) -> str:
        """Submits the specified job script to Slurm.

        Args:
            job_path: The path to the job script to submit.
            working_dir: The working directory to submit the job from.
            node_count: The number of nodes to use for the job.
            name: The name of the job (optional).

        Returns:
            The ID of the submitted job.
        """
        optional_name_args = ""
        if name:
            optional_name_args = f"--job-name={name}"
        sbatch_cmd = (
            f"sbatch --nodes={node_count} {optional_name_args} --parsable {job_path}"
        )
        result = self.run_commands([f"cd {working_dir}", sbatch_cmd])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to submit job. stderr: {result.stderr}")
        return self._parse_job_id(result.stdout.strip())

    def list_jobs(self) -> list[JobStatus]:
        """Lists all jobs for the current user.

        Returns:
            A list of JobStatus.
        """
        response_format = "JobId%-30,JobName%30,User%30,State%30,Reason%30"
        # Forcibly list all jobs since Jan 1, 2025.
        # Otherwise completed jobs older than ~24 hours may not be listed.
        command = (
            f"sacct --user={self._user} --format='{response_format}' -X "
            "--starttime 2025-01-01"
        )
        result = self.run_commands([command])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to list jobs. stderr: {result.stderr}")
        # Parse STDOUT to retrieve job statuses.
        lines = result.stdout.strip().split("\n")
        jobs = []
        if len(lines) < 2:
            return jobs
        # Look for a line starting in JobID followed by a line starting with "--".
        start_idx = -1
        for idx in range(len(lines) - 1):
            if lines[idx].startswith("JobID") and lines[idx + 1].startswith("--"):
                start_idx = idx
                break
        if start_idx == -1:
            raise RuntimeError(
                f"Failed to parse job list. Unexpected format: {result.stdout}"
            )
        lines = lines[start_idx:]
        # The first two lines are metadata headers.
        # The top line is composed of column titles.
        # The second line is composed of ---- characters, each the length of a column.
        metadata_headers = lines[:2]
        column_lengths = [len(col) for col in lines[1].strip().split(" ")]
        job_lines = lines[2:]
        for line in job_lines:
            job_metadata = "\n".join(metadata_headers + [line])
            status = _split_status_line(
                line, column_lengths, self._cluster_name, job_metadata
            )
            jobs.append(status)
        return jobs

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """Gets the specified job's status.

        Args:
            job_id: The ID of the job to get.

        Returns:
            The job status if found, None otherwise.
        """
        job_list = self.list_jobs()
        for job in job_list:
            if job.id == job_id:
                return job
        return None

    def cancel(self, job_id) -> Optional[JobStatus]:
        """Cancels the specified job.

        Args:
            job_id: The ID of the job to cancel.

        Returns:
            The job status if found, None otherwise.
        """
        command = f"scancel {job_id}"
        result = self.run_commands([command])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to cancel job. stderr: {result.stderr}")
        return self.get_job(job_id)

    @retry_auth
    def put_recursive(self, source: str, destination: str) -> None:
        """Puts the specified file/directory to the remote path using rsync.

        Args:
            source: The local file/directory to write.
            destination: The remote path to write the file/directory to.
        """
        if Path(source).is_dir():
            self.run_commands([f"mkdir -p {destination}"])
        tests_dir = Path(source) / "tests"
        git_ignore = Path(source) / ".gitignore"
        rsync_cmd_list = [f'rsync -e "ssh {_CTRL_PATH}" -avz --delete ']
        if git_ignore.is_file():
            rsync_cmd_list.append(f"--exclude-from {str(git_ignore)} ")
        if tests_dir.is_dir():
            rsync_cmd_list.append(f"--exclude {str(tests_dir)} ")
        rsync_cmd_list.append(f"{source} ")
        rsync_cmd_list.append(f"{self._user}@{self._slurm_host}:{destination}")
        rsync_cmd = "".join(rsync_cmd_list)
        logger.info(f"Running rsync command: {rsync_cmd} ...")
        try:
            child = subprocess.run(
                rsync_cmd,
                shell=True,
                capture_output=True,
                timeout=300,
            )
            logger.info(f"Rsync command completed with exit code: {child.returncode}")
            if child.returncode != 0:
                parsed_stderr = child.stderr.decode("utf-8") if child.stderr else ""
                raise RuntimeError(f"Rsync failed. stderr: {parsed_stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout while running rsync command.")

    def put(self, file_contents: str, destination: str) -> None:
        """Puts the specified file contents to the remote path.

        Args:
            file_contents: The contents of the file to write.
            destination: The remote path to write the file to.
        """
        destination_path = Path(destination)
        parent_dir = destination_path.parent
        dir_cmd = f"mkdir -p {parent_dir}"
        create_cmd = f"touch {destination}"
        write_command = f'cat <<"SCRIPTFILETAG" > {destination}'
        file_suffix = "SCRIPTFILETAG"
        cmds = [dir_cmd, create_cmd, write_command, file_contents, file_suffix]
        result = self.run_commands(cmds)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to write file. stderr: {result.stderr}")
