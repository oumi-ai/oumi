"""Job class for executing oumi verbs within workflows."""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from oumi.workflow.config import JobConfig
from oumi.workflow.progress_parser import get_parser_registry

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job in the workflow."""

    PENDING = "pending"  # Not yet started
    WAITING = "waiting"  # Waiting for dependencies
    QUEUED = "queued"  # Ready to run, waiting for resources
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user
    TIMEOUT = "timeout"  # Exceeded timeout


@dataclass
class JobMetrics:
    """Metrics collected during job execution."""

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Progress tracking
    current_step: int = 0
    total_steps: Optional[int] = None
    progress_percent: float = 0.0

    # Training metrics (if applicable)
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[int] = None

    # System metrics
    gpu_id: Optional[int] = None
    gpu_memory_used: Optional[float] = None
    gpu_utilization: Optional[float] = None

    # Output location
    output_dir: Optional[str] = None
    log_file: Optional[str] = None

    # Custom metrics
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Result of job execution."""

    status: JobStatus
    metrics: JobMetrics
    error: Optional[str] = None
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    artifacts: dict[str, Path] = field(default_factory=dict)
    # Example artifacts:
    # {"checkpoint": Path("/path/to/ckpt"), "metrics": Path("/metrics.json")}


class Job:
    """Represents a single job (oumi verb execution) in a workflow."""

    def __init__(self, config: JobConfig, workflow_output_dir: Optional[Path] = None):
        """Initialize job.

        Args:
            config: Job configuration
            workflow_output_dir: Base output directory for workflow
        """
        self.config = config
        self.workflow_output_dir = workflow_output_dir

        # State
        self.status = JobStatus.PENDING
        self.metrics = JobMetrics()
        self.result: Optional[JobResult] = None

        # Execution
        self._process: Optional[asyncio.subprocess.Process] = None
        self._task: Optional[asyncio.Task] = None
        self._retry_count = 0

        # Resource assignment
        self.assigned_gpu: Optional[int] = None
        self.assigned_remote: Optional[str] = None

    @property
    def id(self) -> str:
        """Unique identifier for this job."""
        return self.config.name

    @property
    def is_ready(self) -> bool:
        """Check if job is ready to run (all dependencies met)."""
        return self.status in (JobStatus.QUEUED, JobStatus.PENDING)

    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING

    @property
    def is_complete(self) -> bool:
        """Check if job has completed (success or failure)."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        )

    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return (
            self.status == JobStatus.FAILED
            and self._retry_count < self.config.max_retries
        )

    async def run(
        self,
        env: Optional[dict[str, str]] = None,
        on_progress: Optional[callable] = None,
    ) -> JobResult:
        """Execute the job.

        Args:
            env: Additional environment variables
            on_progress: Callback for progress updates (receives JobMetrics)

        Returns:
            JobResult with execution details

        Raises:
            asyncio.TimeoutError: If job exceeds timeout
            RuntimeError: If job is not in correct state to run
        """
        if not self.is_ready:
            raise RuntimeError(
                f"Job {self.id} is not ready to run (status: {self.status})"
            )

        self.status = JobStatus.RUNNING
        self.metrics.start_time = time.time()

        try:
            # Build command
            cmd = self._build_command()
            logger.info(f"Starting job {self.id}: {' '.join(cmd)}")

            # Prepare environment
            job_env = os.environ.copy()
            if env:
                job_env.update(env)
            if self.config.env:
                job_env.update(self.config.env)

            # Set GPU if assigned
            if self.assigned_gpu is not None:
                job_env["CUDA_VISIBLE_DEVICES"] = str(self.assigned_gpu)
                self.metrics.gpu_id = self.assigned_gpu

            # Set working directory
            workdir = self.config.workdir or (
                str(self.workflow_output_dir) if self.workflow_output_dir else None
            )

            # Create output directory for job
            if self.workflow_output_dir:
                job_output_dir = self.workflow_output_dir / self.config.name
                job_output_dir.mkdir(parents=True, exist_ok=True)
                self.metrics.output_dir = str(job_output_dir)

                # Set up log file
                log_file = job_output_dir / "job.log"
                self.metrics.log_file = str(log_file)

            # Execute command
            if self.config.timeout:
                result = await asyncio.wait_for(
                    self._execute_command(cmd, job_env, workdir, on_progress),
                    timeout=self.config.timeout,
                )
            else:
                result = await self._execute_command(cmd, job_env, workdir, on_progress)

            self.result = result
            return result

        except asyncio.TimeoutError:
            logger.error(f"Job {self.id} exceeded timeout of {self.config.timeout}s")
            self.status = JobStatus.TIMEOUT
            self.result = JobResult(
                status=JobStatus.TIMEOUT,
                metrics=self.metrics,
                error=f"Job exceeded timeout of {self.config.timeout}s",
            )
            if self._process:
                try:
                    self._process.terminate()
                    await asyncio.sleep(2)
                    if self._process.returncode is None:
                        self._process.kill()
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")
            return self.result

        except Exception as e:
            logger.exception(f"Job {self.id} failed with exception")
            self.status = JobStatus.FAILED
            self.result = JobResult(
                status=JobStatus.FAILED,
                metrics=self.metrics,
                error=str(e),
            )
            return self.result

        finally:
            self.metrics.end_time = time.time()
            if self.metrics.start_time:
                self.metrics.duration = self.metrics.end_time - self.metrics.start_time

    async def _execute_command(
        self,
        cmd: list[str],
        env: dict[str, str],
        workdir: Optional[str],
        on_progress: Optional[callable],
    ) -> JobResult:
        """Execute command and capture output.

        Args:
            cmd: Command to execute
            env: Environment variables
            workdir: Working directory
            on_progress: Progress callback

        Returns:
            JobResult with execution details
        """
        stdout_lines = []
        stderr_lines = []

        try:
            # Start process
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=workdir,
            )

            # Read output asynchronously
            async def read_stream(stream, lines_buffer, is_stderr=False):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8", errors="replace").rstrip()
                    lines_buffer.append(line_str)

                    # Write to log file if available
                    if self.metrics.log_file:
                        with open(self.metrics.log_file, "a") as f:
                            prefix = "[STDERR] " if is_stderr else ""
                            f.write(f"{prefix}{line_str}\n")

                    # Try to parse progress from output
                    if on_progress and not is_stderr:
                        registry = get_parser_registry()
                        if registry.parse_line(
                            self.config.verb, line_str, self.metrics
                        ):
                            on_progress(self.metrics)

            # Read stdout and stderr concurrently
            await asyncio.gather(
                read_stream(self._process.stdout, stdout_lines),
                read_stream(self._process.stderr, stderr_lines, is_stderr=True),
            )

            # Wait for process to complete
            return_code = await self._process.wait()

            # Determine status
            if return_code == 0:
                self.status = JobStatus.COMPLETED
                status = JobStatus.COMPLETED
            else:
                self.status = JobStatus.FAILED
                status = JobStatus.FAILED

            # Discover artifacts after job completion
            artifacts = self._discover_artifacts()

            return JobResult(
                status=status,
                metrics=self.metrics,
                return_code=return_code,
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
                error="\n".join(stderr_lines) if return_code != 0 else None,
                artifacts=artifacts,
            )

        except Exception as e:
            logger.exception(f"Error executing command: {e}")
            self.status = JobStatus.FAILED
            return JobResult(
                status=JobStatus.FAILED,
                metrics=self.metrics,
                error=str(e),
                stdout="\n".join(stdout_lines),
                stderr="\n".join(stderr_lines),
            )

    def _build_command(self) -> list[str]:
        """Build command to execute the oumi verb.

        Returns:
            Command as list of strings
        """
        # Get python executable
        python_exe = sys.executable

        # Build base command: python -m oumi <verb>
        cmd = [python_exe, "-m", "oumi", self.config.verb]

        # Add config file
        cmd.extend(["--config", self.config.config])

        # Add additional arguments
        if self.config.args:
            cmd.extend(self.config.args)

        return cmd

    def _discover_artifacts(self) -> dict[str, Path]:
        """Discover artifacts produced by the job.

        Looks for common output files in the job output directory.

        Returns:
            Dict mapping artifact name to path
        """
        artifacts = {}

        if not self.metrics.output_dir:
            return artifacts

        output_dir = Path(self.metrics.output_dir)
        if not output_dir.exists():
            return artifacts

        # Common artifact patterns to look for
        patterns = {
            "checkpoint": ["checkpoint-*", "*.ckpt", "*.pth", "model.safetensors"],
            "config": ["config.json", "*.yaml", "*.yml"],
            "metrics": ["metrics.json", "results.json", "eval_results.json"],
            "logs": ["trainer_log.json", "training.log"],
            "tokenizer": ["tokenizer_config.json", "tokenizer.json"],
        }

        for artifact_type, globs in patterns.items():
            for glob_pattern in globs:
                matches = list(output_dir.rglob(glob_pattern))
                if matches:
                    # Use the most recent file if multiple matches
                    most_recent = max(matches, key=lambda p: p.stat().st_mtime)
                    artifacts[artifact_type] = most_recent
                    break  # Only one artifact per type

        return artifacts

    async def cancel(self) -> None:
        """Cancel the running job."""
        if not self.is_running:
            logger.warning(f"Cannot cancel job {self.id} - not running")
            return

        logger.info(f"Cancelling job {self.id}")
        self.status = JobStatus.CANCELLED

        if self._process:
            try:
                self._process.terminate()
                await asyncio.sleep(2)
                if self._process.returncode is None:
                    self._process.kill()
            except Exception as e:
                logger.warning(f"Error cancelling job {self.id}: {e}")

    def reset(self) -> None:
        """Reset job state for retry."""
        self.status = JobStatus.PENDING
        self.metrics = JobMetrics()
        self.result = None
        self._process = None
        self._task = None
        self._retry_count += 1

    def __repr__(self) -> str:
        """String representation."""
        return f"Job(name={self.config.name}, verb={self.config.verb}, status={self.status})"
