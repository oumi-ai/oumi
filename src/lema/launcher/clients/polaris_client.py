import functools
import io
import re
from enum import Enum
from getpass import getpass
from typing import List, Optional

from asyncssh.sftp import SFTPNoConnection
from fabric import Connection
from paramiko.ssh_exception import BadAuthenticationType
from sshfs import SSHFileSystem

from lema.core.types.base_cluster import JobStatus
from lema.utils.logging import logger


def retry_auth(user_function):
    """Decorator to retry a function if the connection is closed."""

    @functools.wraps(user_function)
    def wrapper(self, *args, **kwargs):
        try:
            return user_function(self, *args, **kwargs)
        except (EOFError, BadAuthenticationType):
            logger.warning("Connection closed. Reconnecting...")
            self._connection = self._refresh_creds(close_connection=True)
            return user_function(self, *args, **kwargs)

    return wrapper


def retry_fs(user_function):
    """Decorator to retry a function if the filesystem is closed."""

    @functools.wraps(user_function)
    def wrapper(self, *args, **kwargs):
        try:
            return user_function(self, *args, **kwargs)
        except SFTPNoConnection:
            logger.warning("Connection closed. Reconnecting...")
            self._fs = self._refresh_fs()
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

    _CD_PATTERN = r"cd\s+(.*?)($|\s)"

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
        self._connection = self._refresh_creds()
        self._fs = self._refresh_fs()

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

    def _refresh_fs(self) -> SSHFileSystem:
        """Refreshes the remote filesystem."""
        return SSHFileSystem(
            "polaris.alcf.anl.gov",
            username=self._user,
            password=getpass(
                prompt="Mounting Polaris filesystem...\n"
                "This requires a new set of Polaris credentials.\n"
                "**You must refresh your passcode before proceeding!**\n"
                f"Polaris passcode for {self._user}: "
            ),
        )

    def _refresh_creds(self, close_connection=False) -> Connection:
        """Refreshes the credentials for the client."""
        if close_connection:
            self._connection.close()
        connection = Connection(
            "polaris.alcf.anl.gov",
            user=self._user,
            connect_kwargs={
                "password": getpass(prompt=f"Polaris passcode for {self._user}: ")
            },
        )
        connection.open()
        return connection

    @retry_auth
    def run_commands(self, commands: List[str]) -> None:
        """Runs the provided commands using recursive context setting.

        Due to an implementation detail in Fabric, `cd` commands are not preserved
        unless run in the same context as the previous command. To this end, this
        function detects `cd` commands and generates a new context for them. Following
        commands are invoked in the same context via recursion.

        Args:
            commands: The commands to run.
        """
        if len(commands) == 0:
            return
        command = commands[0]
        remaining_commands = commands[1:]
        match = re.search(self._CD_PATTERN, command)
        if match:
            location = match.group(1)
            with self._connection.cd(location):
                return self.run_commands(remaining_commands)
        result = self._connection.run(command, warn=True)
        if not result:
            raise RuntimeError(
                f"Failed to run command: {command} . stderr: {result.stderr}"
            )
        return self.run_commands(remaining_commands)

    @retry_auth
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
        with self._connection.cd(working_dir):
            result = self._connection.run(
                f"qsub -l select={node_count}:system=polaris "
                f"-q {queue.value} {optional_name_args} {job_path}",
                warn=True,
            )
            if not result:
                raise RuntimeError(f"Failed to submit job. stderr: {result.stderr}")
            return self._get_short_job_id(result.stdout.strip())

    @retry_auth
    def list_jobs(self, queue: SupportedQueues) -> List[JobStatus]:
        """Lists a list of job statuses for the given queue.

        Returns:
            A list of dictionaries, each containing the status of a cluster.
        """
        command = f"qstat -s -x -w -u {self._user}"
        result = self._connection.run(command, warn=True)
        if not result:
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

    @retry_auth
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

    @retry_auth
    def cancel(self, job_id, queue: SupportedQueues) -> Optional[JobStatus]:
        """Cancels the specified job.

        Args:
            job_id: The ID of the job to cancel.
            queue: The name of the queue to search.

        Returns:
            The job status if found, None otherwise.
        """
        command = f"qdel {job_id}"
        result = self._connection.run(command, warn=True)
        if not result:
            raise RuntimeError(f"Failed to cancel job. stderr: {result.stderr}")
        return self.get_job(job_id, queue)

    @retry_fs
    def put_recursive(self, source: str, destination: str) -> None:
        """Puts the specified file/directory to the remote path, recursively.

        Args:
            source: The local file/directory to write.
            destination: The remote path to write the file/directory to.
        """
        if self._fs is None:
            self._fs = self._refresh_fs()
        self._fs.put(source, destination, recursive=True)
        # Ensure all copied files are executable as `put` does not propagate
        # permissions.
        self._connection.run(f"chmod -R +x {destination}", warn=True)

    @retry_auth
    def put(self, file_contents: str, destination: str) -> None:
        """Puts the specified file contents to the remote path.

        Args:
            file_contents: The contents of the file to write.
            destination: The remote path to write the file to.
        """
        self._connection.put(io.StringIO(file_contents), destination)
