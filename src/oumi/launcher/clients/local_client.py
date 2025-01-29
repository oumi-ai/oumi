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
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from subprocess import PIPE, Popen
from threading import Lock, Thread
from typing import Optional

from oumi.core.configs import JobConfig
from oumi.core.launcher import JobStatus


@dataclass
class _LocalJob:
    """A class representing a job running locally."""

    status: JobStatus
    config: JobConfig
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class _JobState(Enum):
    """An enumeration of the possible states of a job."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class LocalClient:
    """A client for running jobs locally in a subprocess."""

    # The maximum number of characters to read from the subprocess's stdout and stderr.
    _MAX_BUFFER_SIZE = 1024
    # The environment variable used to specify the logging directory.
    _OUMI_LOGGING_DIR = "OUMI_LOGGING_DIR"

    def __init__(self):
        """Initializes a new instance of the LocalClient class."""
        self._mutex = Lock()
        self._next_job_id = 0
        # A mapping of job IDs to their respective job configurations.
        self._jobs = {}
        self._running_process = None
        self._worker = Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _update_job_status(self, job_id: str, status: _JobState) -> None:
        """Updates the status of the job. Assumes the mutex is already acquired."""
        if job_id not in self._jobs:
            return
        self._jobs[job_id].status.status = status.value
        is_done = status in (_JobState.COMPLETED, _JobState.FAILED, _JobState.CANCELED)
        self._jobs[job_id].status.done = is_done

    def _worker_run_job(self) -> Optional[_LocalJob]:
        """Kicks off and returns a new job. Assumes the mutex is already acquired."""
        job = self._get_next_job()
        if job is None:
            return None
        env_copy = os.environ.copy()
        env_copy.update(job.config.envs)
        # Check if the user has specified a logging directory.
        if self._OUMI_LOGGING_DIR in env_copy:
            logging_dir = Path(env_copy[self._OUMI_LOGGING_DIR])
            logging_dir.mkdir(parents=True, exist_ok=True)
            dt = datetime.now()
            log_format = f"{dt:%Y_%m_%d_%H_%M_%S}_{dt.microsecond // 1000:03d}"
            job.stderr = str(logging_dir / f"{log_format}_{job.status.id}.stderr")
            job.stdout = str(logging_dir / f"{log_format}_{job.status.id}.stdout")
        # Always change to the working directory before running the job.
        working_dir_cmd = f"cd {job.config.working_dir}"
        setup_cmds = job.config.setup or ""
        cmds = "\n".join([working_dir_cmd, setup_cmds, job.config.run])
        # Start the job but don't block.
        stderr_logs = open(job.stderr, "w") if job.stderr else PIPE
        stdout_logs = open(job.stdout, "w") if job.stdout else PIPE
        self._running_process = Popen(
            cmds,
            shell=True,
            env=env_copy,
            stdout=stdout_logs,
            stderr=stderr_logs,
        )
        self._update_job_status(job.status.id, _JobState.RUNNING)
        return job

    def _worker_handle_running_job(self, job: _LocalJob) -> None:
        """Polls and handles the specified job. Acquires the mutex."""
        # Return immediately if no job is running.
        if self._running_process is None:
            return
        # Wait for the job to finish. No need to grab the mutex here.
        if self._running_process.wait() == 0:
            # Job was successful.
            finish_time = datetime.fromtimestamp(time.time()).isoformat()
            with self._mutex:
                self._jobs[
                    job.status.id
                ].status.metadata = f"Job finished at {finish_time} ."
                self._update_job_status(job.status.id, _JobState.COMPLETED)
                if job.stdout is not None:
                    self._jobs[
                        job.status.id
                    ].status.metadata += f" Logs available at: {job.stdout}"
        else:
            # Job failed.
            with self._mutex:
                self._update_job_status(job.status.id, _JobState.FAILED)
                if job.stderr is not None:
                    self._jobs[
                        job.status.id
                    ].status.metadata = f"Error logs available at: {job.stderr}"
                else:
                    error_metadata = ""
                    if self._running_process.stderr is not None:
                        for line in self._running_process.stderr:
                            error_metadata += str(line)
                    # Only keep the last _MAX_BUFFER_SIZE characters.
                    error_metadata = error_metadata[-self._MAX_BUFFER_SIZE :]
                    self._jobs[job.status.id].status.metadata = error_metadata

    def _worker_loop(self):
        """The main worker loop that runs jobs."""
        while True:
            with self._mutex:
                # Run the next job if it exists.
                job = self._worker_run_job()
            # No job to run, sleep for a bit.
            if job is None:
                time.sleep(5)
                continue
            # Wait for the job to finish.
            self._worker_handle_running_job(job)
            # Clear the running process.
            with self._mutex:
                self._running_process = None

    def _generate_next_job_id(self) -> str:
        """Gets the next job ID."""
        job_id = self._next_job_id
        self._next_job_id += 1
        return str(job_id)

    def _get_next_job(self) -> Optional[_LocalJob]:
        """Gets the next QUEUED job from the queue."""
        queued_jobs = [
            job
            for job in self._jobs.values()
            if job.status.status == _JobState.QUEUED.value
        ]
        if len(queued_jobs) == 0:
            return None
        next_job_id = queued_jobs[0].status.id
        for job in queued_jobs:
            if int(job.status.id) < int(next_job_id):
                next_job_id = job.status.id
        return self._jobs[next_job_id]

    def submit_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster."""
        with self._mutex:
            job_id = self._generate_next_job_id()
            name = job.name if job.name else job_id
            status = JobStatus(
                name=name,
                id=job_id,
                status=_JobState.QUEUED.value,
                cluster="",
                metadata="",
                done=False,
            )
            self._jobs[job_id] = _LocalJob(status=status, config=job)
            return status

    def list_jobs(self) -> list[JobStatus]:
        """Returns a list of job statuses."""
        with self._mutex:
            return [job.status for job in self._jobs.values()]

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
            queue: The name of the queue to search.

        Returns:
            The job status if found, None otherwise.
        """
        with self._mutex:
            if job_id not in self._jobs:
                return None
            job = self._jobs[job_id]
            if job.status.status == _JobState.RUNNING.value:
                if self._running_process is not None:
                    self._running_process.terminate()
                self._update_job_status(job_id, _JobState.CANCELED)
            elif job.status.status == _JobState.QUEUED.value:
                self._update_job_status(job_id, _JobState.CANCELED)
            return job.status
