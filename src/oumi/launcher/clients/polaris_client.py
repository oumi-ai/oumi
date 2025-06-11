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
from enum import Enum
from getpass import getpass
from pathlib import Path
from typing import Optional

import pexpect

from oumi.core.launcher import JobStatus
from oumi.utils.logging import logger

_CTRL_PATH = "-S ~/.ssh/control-%h-%p-%r"


class _PolarisAuthException(Exception):
    pass


def _check_connection(user: str):
    """Checks if the connection is still open."""
    ssh_cmd = f"ssh {_CTRL_PATH} -O check {user}@polaris.alcf.anl.gov"
    try:
        child = subprocess.run(
            ssh_cmd,
            shell=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        raise _PolarisAuthException("Timeout while checking connection.")
    if child.returncode == 0:
        return
    raise _PolarisAuthException("Connection to Polaris is closed.")


@dataclass
class PolarisResponse:
    """A response from Polaris."""

    stdout: str
    stderr: str
    exit_code: int


def retry_auth(user_function):
    """Decorator to ensure auth is fresh before calling a function."""

    @functools.wraps(user_function)
    def wrapper(self, *args, **kwargs):
        self._refresh_creds()
        return user_function(self, *args, **kwargs)

    return wrapper


class PolarisClient:
    """A client for communicating with Polaris at ALCF."""

    class SupportedQueues(Enum):
        """Enum representing the supported queues on Polaris.

        For more details, see:
        https://docs.alcf.anl.gov/polaris/running-jobs/#queues
        """

        # The demand queue can only be used with explicit permission from ALCF.
        # Do not use this queue unless you have been granted permission.
        DEMAND = "demand"
        DEBUG = "debug"
        DEBUG_SCALING = "debug-scaling"
        PREEMPTABLE = "preemptable"
        PROD = "prod"

    _FINISHED_STATUS = "F"

    _PROD_QUEUES = {
        "small",
        "medium",
        "large",
        "backfill-small",
        "backfill-medium",
        "backfill-large",
    }

    def __init__(self, user: str):
        """Initializes a new instance of the PolarisClient class.

        Args:
            user: The user to act as.
        """
        self._user = user
        self._refresh_creds()

    def _split_status_line(self, line: str, metadata: str) -> JobStatus:
        """Splits a status line into a JobStatus object.

        The expected order of job fields is:
        0. Job ID
        1. User
        2. Queue
        3. Job Name
        4. Session ID
        5. Node Count
        6. Tasks
        7. Required Memory
        8. Required Time
        9. Status
        10. Ellapsed Time

        Args:
            line: The line to split.
            metadata: Additional metadata to attach to the job status.

        Returns:
            A JobStatus object.
        """
        fields = re.sub(" +", " ", line.strip()).split(" ")
        if len(fields) != 11:
            raise ValueError(
                f"Invalid status line: {line}. "
                f"Expected 11 fields, but found {len(fields)}."
            )
        return JobStatus(
            id=self._get_short_job_id(fields[0]),
            name=fields[3],
            status=fields[9],
            cluster=fields[2],
            metadata=metadata,
            done=fields[9] == self._FINISHED_STATUS,
        )

    def _get_short_job_id(self, job_id: str) -> str:
        """Gets the short form of the job ID.

        Polaris Job IDs should be of the form:
        `2037042.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`
        where the shortened ID is `2037042`.

        Args:
            job_id: The job ID to shorten.

        Returns:
            The short form of the job ID.
        """
        if "." not in job_id:
            return job_id
        return job_id.split(".")[0]

    def _refresh_creds(self):
        """Refreshes the credentials for the client."""
        try:
            _check_connection(self._user)
            # We have fresh credentials, so we return early.
            return
        except _PolarisAuthException:
            logger.warning("No connection found. Establishing a new SSH tunnel...")
        ssh_cmd = (
            f'ssh -f -N -M {_CTRL_PATH} -o "ControlPersist 4h" '
            f"{self._user}@polaris.alcf.anl.gov"
        )
        child = pexpect.spawn(ssh_cmd)
        child.expect("Password:")
        child.sendline(getpass(prompt=f"Polaris passcode for {self._user}: "))
        child.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=10)
        output = child.before
        child.close()
        exit_code = child.exitstatus
        if exit_code != 0:
            logger.error(f"Credential error: {output}")
            raise RuntimeError("Failed to refresh Polaris credentials.")

    @staticmethod
    def get_active_users() -> list[str]:
        """Gets the list of users with an open SSH tunnel to Polaris.

        Returns:
            A list of users.
        """
        # List all active users with an open SSH tunnel to Polaris.
        command = "ls ~/.ssh/ | egrep 'control-polaris.alcf.anl.gov-.*-.*'"
        result = subprocess.run(command, shell=True, capture_output=True)
        if result.returncode != 0:
            return []
        ssh_tunnel_pattern = r"control-polaris.alcf.anl.gov-[^-]*-(.*)"
        lines = result.stdout.decode("utf-8").strip().split("\n")
        users = set()
        for line in lines:
            match = re.match(ssh_tunnel_pattern, line.strip())
            if match:
                users.add(match.group(1))
        return list(users)

    def _compute_duration_debug_str(self, start_time: float) -> str:
        duration_sec = time.perf_counter() - start_time
        return f"Duration: {duration_sec:.2f} sec"

    @retry_auth
    def run_commands(self, commands: list[str]) -> PolarisResponse:
        """Runs the provided commands in a single SSH command.

        Args:
            commands: The commands to run.
        """
        ssh_cmd = f"ssh {_CTRL_PATH} {self._user}@polaris.alcf.anl.gov  << 'EOF'"
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
            duration_str = self._compute_duration_debug_str(start_time)
            if child.returncode == 0:
                logger.debug(f"Commands successfully finished! {duration_str}")
            else:
                logger.error(
                    f"Commands failed with code: {child.returncode}! {duration_str}"
                )
            return PolarisResponse(
                stdout=child.stdout.decode("utf-8"),
                stderr=child.stderr.decode("utf-8"),
                exit_code=child.returncode,
            )
        except subprocess.TimeoutExpired:
            duration_str = self._compute_duration_debug_str(start_time)
            logger.exception(f"Commands timed out ({duration_str})! {new_cmd}")
            return PolarisResponse(
                stdout="",
                stderr=f"Timeout while running command: {new_cmd}",
                exit_code=1,
            )
        except Exception:
            duration_str = self._compute_duration_debug_str(start_time)
            logger.exception(f"Command failed ({duration_str})! {new_cmd}")
            raise

    def submit_job(
        self,
        job_path: str,
        working_dir: str,
        node_count: int,
        queue: SupportedQueues,
        name: Optional[str],
    ) -> str:
        """Submits the specified job script to Polaris.

        Args:
            job_path: The path to the job script to submit.
            working_dir: The working directory to submit the job from.
            node_count: The number of nodes to use for the job.
            queue: The name of the queue to submit the job to.
            name: The name of the job (optional).

        Returns:
            The ID of the submitted job.
        """
        optional_name_args = ""
        if name:
            optional_name_args = f"-N {name}"
        qsub_cmd = (
            f"qsub -l select={node_count}:system=polaris -q {queue.value}"
            f" {optional_name_args} {job_path}"
        )
        result = self.run_commands([f"cd {working_dir}", qsub_cmd])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to submit job. stderr: {result.stderr}")
        return self._get_short_job_id(result.stdout.strip())

    def list_jobs(self, queue: SupportedQueues) -> list[JobStatus]:
        """Lists a list of job statuses for the given queue.

        Returns:
            A list of dictionaries, each containing the status of a cluster.
        """
        command = f"qstat -s -x -w -u {self._user}"
        result = self.run_commands([command])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to list jobs. stderr: {result.stderr}")
        # Parse STDOUT to retrieve job statuses.
        lines = result.stdout.strip().split("\n")
        jobs = []
        # Non-empty responses should have at least 4 lines.
        if len(lines) < 4:
            return jobs
        metadata_header = lines[1:4]
        job_lines = lines[4:]
        line_number = 0
        while line_number < len(job_lines) - 1:
            line = job_lines[line_number]
            # Every second line is metadata.
            metadata_line = job_lines[line_number + 1]
            job_metadata = "\n".join(metadata_header + [line, metadata_line])
            status = self._split_status_line(line, job_metadata)
            if status.cluster == queue.value:
                jobs.append(status)
            elif (
                queue == self.SupportedQueues.PROD
                and status.cluster in self._PROD_QUEUES
            ):
                jobs.append(status)
            line_number += 2
        if line_number != len(job_lines):
            raise RuntimeError("At least one job status was not parsed.")
        return jobs

    def get_job(self, job_id: str, queue: SupportedQueues) -> Optional[JobStatus]:
        """Gets the specified job's status.

        Args:
            job_id: The ID of the job to get.
            queue: The name of the queue to search.

        Returns:
            The job status if found, None otherwise.
        """
        job_list = self.list_jobs(queue)
        for job in job_list:
            if job.id == job_id:
                return job
        return None

    def cancel(self, job_id, queue: SupportedQueues) -> Optional[JobStatus]:
        """Cancels the specified job.

        Args:
            job_id: The ID of the job to cancel.
            queue: The name of the queue to search.

        Returns:
            The job status if found, None otherwise.
        """
        command = f"qdel {job_id}"
        result = self.run_commands([command])
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to cancel job. stderr: {result.stderr}")
        return self.get_job(job_id, queue)

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
        rsync_cmd_list.append(f"{self._user}@polaris.alcf.anl.gov:{destination}")
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
