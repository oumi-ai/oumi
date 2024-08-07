import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import reduce
from subprocess import PIPE, Popen
from threading import Lock, Thread
from typing import List, Optional

from lema.core.types.base_cluster import JobStatus
from lema.core.types.configs import JobConfig


@dataclass
class _LocalJob:
    """A class representing a job running locally."""

    status: JobStatus
    config: JobConfig


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

    def __init__(self):
        """Initializes a new instance of the LocalClient class."""
        self._mutex = Lock()
        self._next_job_id = 0
        # A mapping of job IDs to their respective job configurations.
        self._jobs = {}
        self._running_process = None
        self._worker = Thread(target=self._worker_loop)

    def _worker_loop(self):
        """The main worker loop that runs jobs."""
        while True:
            with self._mutex:
                # Safe because we're in the job mutex.
                job = self._get_next_job()
                if job is not None:
                    env_copy = os.environ.copy()
                    env_copy.update(job.config.envs)
                    # Always change to the working directory before running the job.
                    working_dir_cmd = f"cd {job.config.working_dir}"
                    setup_cmds = job.config.setup or ""
                    cmds = "\n".join([working_dir_cmd, setup_cmds, job.config.run])
                    # Start the job but don't block.
                    self._running_process = Popen(
                        cmds,
                        shell=True,
                        env=env_copy,
                        stdout=PIPE,
                        stderr=PIPE,
                    )
            if job is None:
                time.sleep(5)
                continue
            if self._running_process is not None:
                # Wait for the job to finish. No need to grab the mutex here.
                if self._running_process.wait() == 0:
                    # Job was successful.
                    finish_time = datetime.fromtimestamp(time.time()).isoformat()
                    with self._mutex:
                        self._jobs[
                            job.status.id
                        ].status.status = _JobState.COMPLETED.value
                        self._jobs[
                            job.status.id
                        ].status.metadata = f"Job finished at ${finish_time}"
                else:
                    # Job failed.
                    with self._mutex:
                        self._jobs[job.status.id].status.status = _JobState.FAILED.value
                        error_metadata = ""
                        if self._running_process.stderr is not None:
                            for line in self._running_process.stderr:
                                error_metadata += str(line)
                        # Only keep the last _MAX_BUFFER_SIZE characters.
                        error_metadata = error_metadata[-self._MAX_BUFFER_SIZE :]
                        self._jobs[job.status.id].status.metadata = error_metadata
            with self._mutex:
                self._running_process = None

    def _get_next_job_id(self) -> str:
        """Gets the next job ID."""
        job_id = self._next_job_id
        self._next_job_id += 1
        return str(job_id)

    def _get_next_job(self) -> Optional[_LocalJob]:
        """Gets the next job from the queue and marks it as RUNNING."""
        queued_jobs = [
            job
            for job in self._jobs.values()
            if job.status.status == _JobState.QUEUED.value
        ]
        if len(queued_jobs) == 0:
            return None
        next_job_id = str(
            reduce(
                lambda acc, val: min(acc, int(val.status.id)),
                queued_jobs,
                -1,
            )
        )
        next_job = self._jobs[next_job_id]
        next_job.status.status = _JobState.RUNNING.value
        return next_job

    def _queue_job(self, job: _LocalJob) -> None:
        self._jobs[job.status.id] = job

    def submit_job(self, job: JobConfig) -> JobStatus:
        """Runs the specified job on this cluster."""
        with self._mutex:
            job_id = self._get_next_job_id()
            name = job.name if job.name else job_id
            status = JobStatus(
                name=name,
                id=job_id,
                status=_JobState.QUEUED.value,
                cluster="local",
                metadata="",
            )
            self._queue_job(_LocalJob(status=status, config=job))
            return status

    def list_jobs(self) -> List[JobStatus]:
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
                job.status.status = _JobState.CANCELED.value
            elif job.status.status == _JobState.QUEUED.value:
                job.status.status = _JobState.CANCELED.value
