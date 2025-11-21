"""Workflow orchestration and execution."""

import logging
import time
from pathlib import Path
from typing import Optional

from oumi.workflow.config import WorkflowConfig
from oumi.workflow.executor import WorkflowExecutor
from oumi.workflow.job import Job, JobStatus
from oumi.workflow.resource_manager import ResourceManager
from oumi.workflow.state import (
    JobState,
    WorkflowState,
    WorkflowStateManager,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


class Workflow:
    """Represents a workflow - a collection of jobs with dependencies."""

    def __init__(
        self,
        config: WorkflowConfig,
        state_manager: Optional[WorkflowStateManager] = None,
    ):
        """Initialize workflow.

        Args:
            config: Workflow configuration
            state_manager: State manager for persistence (default: use global instance)
        """
        self.config = config
        self.jobs: dict[str, Job] = {}
        self.state_manager = state_manager or WorkflowStateManager()
        self.created_at = time.time()
        self._setup_jobs()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid workflow configuration:\n" + "\n".join(errors))

    def _setup_jobs(self) -> None:
        """Create Job instances from config."""
        # Create output directory if specified
        output_dir = None
        if self.config.output_dir:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Create jobs
        for job_config in self.config.jobs:
            job = Job(job_config, workflow_output_dir=output_dir)
            self.jobs[job.id] = job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job instance or None
        """
        return self.jobs.get(job_id)

    def get_ready_jobs(self) -> list[Job]:
        """Get jobs that are ready to run (dependencies satisfied).

        Returns:
            List of jobs ready to execute
        """
        ready = []
        for job in self.jobs.values():
            if job.status == JobStatus.PENDING:
                # Check if all dependencies are completed
                deps_satisfied = all(
                    self.jobs[dep_id].is_successful
                    for dep_id in job.config.depends_on
                    if dep_id in self.jobs
                )
                if deps_satisfied:
                    job.status = JobStatus.QUEUED
                    ready.append(job)

        return ready

    def get_running_jobs(self) -> list[Job]:
        """Get jobs currently running.

        Returns:
            List of running jobs
        """
        return [job for job in self.jobs.values() if job.is_running]

    def get_completed_jobs(self) -> list[Job]:
        """Get jobs that have completed.

        Returns:
            List of completed jobs
        """
        return [job for job in self.jobs.values() if job.is_complete]

    def get_failed_jobs(self) -> list[Job]:
        """Get jobs that failed.

        Returns:
            List of failed jobs
        """
        return [
            job
            for job in self.jobs.values()
            if job.status in (JobStatus.FAILED, JobStatus.TIMEOUT)
        ]

    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete (all jobs finished).

        Returns:
            True if all jobs are complete
        """
        return all(job.is_complete for job in self.jobs.values())

    @property
    def is_successful(self) -> bool:
        """Check if workflow completed successfully (all jobs successful).

        Returns:
            True if all jobs completed successfully
        """
        return all(job.is_successful for job in self.jobs.values())

    @property
    def progress(self) -> float:
        """Get overall workflow progress percentage.

        Returns:
            Progress percentage (0-100)
        """
        if not self.jobs:
            return 0.0

        completed = len(self.get_completed_jobs())
        return (completed / len(self.jobs)) * 100

    def estimate_cost(
        self, estimated_hours_per_job: Optional[dict[str, float]] = None
    ) -> float:
        """Estimate total cost of workflow in USD.

        Args:
            estimated_hours_per_job: Dict mapping job ID to estimated hours
                If None, assumes 1 hour per job

        Returns:
            Estimated cost in USD, or 0 if cost_per_gpu_hour not configured
        """
        if not self.config.resources.cost_per_gpu_hour:
            return 0.0

        total_hours = 0.0
        for job in self.jobs.values():
            hours = 1.0  # Default estimate
            if estimated_hours_per_job and job.id in estimated_hours_per_job:
                hours = estimated_hours_per_job[job.id]
            total_hours += hours

        return total_hours * self.config.resources.cost_per_gpu_hour

    def get_actual_cost(self) -> float:
        """Get actual cost based on completed jobs.

        Returns:
            Actual cost in USD, or 0 if cost_per_gpu_hour not configured
        """
        if not self.config.resources.cost_per_gpu_hour:
            return 0.0

        total_hours = 0.0
        for job in self.jobs.values():
            if job.metrics.duration:
                hours = job.metrics.duration / 3600.0  # Convert seconds to hours
                total_hours += hours

        return total_hours * self.config.resources.cost_per_gpu_hour

    def save_state(self, config_path: str = "") -> None:
        """Save workflow state to database.

        Args:
            config_path: Path to config file (for tracking)
        """
        # Convert workflow status
        if self.is_complete:
            if self.is_successful:
                status = WorkflowStatus.COMPLETED
            else:
                status = WorkflowStatus.FAILED
        elif any(job.is_running for job in self.jobs.values()):
            status = WorkflowStatus.RUNNING
        else:
            status = WorkflowStatus.PENDING

        # Convert job states
        job_states = {}
        for job_id, job in self.jobs.items():
            job_state = JobState(
                job_id=job_id,
                status=job.status,
                assigned_gpu=job.assigned_gpu,
                assigned_remote=job.assigned_remote,
                retry_count=job._retry_count,
                start_time=job.metrics.start_time,
                end_time=job.metrics.end_time,
                duration=job.metrics.duration,
                current_step=job.metrics.current_step,
                total_steps=job.metrics.total_steps,
                progress_percent=job.metrics.progress_percent,
                loss=job.metrics.loss,
                learning_rate=job.metrics.learning_rate,
                epoch=job.metrics.epoch,
                artifacts={
                    k: str(v)
                    for k, v in (job.result.artifacts if job.result else {}).items()
                },
                output_dir=job.metrics.output_dir,
                log_file=job.metrics.log_file,
                error=job.result.error if job.result else None,
                return_code=job.result.return_code if job.result else None,
            )
            job_states[job_id] = job_state

        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=self.config.id,
            name=self.config.name,
            config_path=config_path,
            status=status,
            created_at=self.created_at,
            started_at=min(
                (
                    j.metrics.start_time
                    for j in self.jobs.values()
                    if j.metrics.start_time
                ),
                default=None,
            ),
            completed_at=max(
                (j.metrics.end_time for j in self.jobs.values() if j.metrics.end_time),
                default=None,
            )
            if self.is_complete
            else None,
            jobs=job_states,
        )

        # Save to database
        self.state_manager.save_state(workflow_state)
        logger.info(f"Saved workflow state for {self.config.id}")

    @classmethod
    def load_from_state(
        cls,
        workflow_id: str,
        state_manager: Optional[WorkflowStateManager] = None,
    ) -> Optional["Workflow"]:
        """Load workflow from saved state.

        Args:
            workflow_id: Workflow ID to load
            state_manager: State manager to use

        Returns:
            Workflow instance or None if not found
        """
        state_mgr = state_manager or WorkflowStateManager()
        state = state_mgr.load_state(workflow_id)

        if not state:
            return None

        # Load config from path
        from oumi.workflow.config import WorkflowConfig

        try:
            config = WorkflowConfig.from_yaml(state.config_path)
            # Restore workflow ID
            config.id = state.workflow_id
        except Exception as e:
            logger.error(f"Failed to load config from {state.config_path}: {e}")
            return None

        # Create workflow
        workflow = cls(config, state_manager=state_mgr)
        workflow.created_at = state.created_at

        # Restore job states
        for job_id, job_state in state.jobs.items():
            if job_id in workflow.jobs:
                job = workflow.jobs[job_id]
                job.status = job_state.status
                job.assigned_gpu = job_state.assigned_gpu
                job.assigned_remote = job_state.assigned_remote
                job._retry_count = job_state.retry_count

                # Restore metrics
                job.metrics.start_time = job_state.start_time
                job.metrics.end_time = job_state.end_time
                job.metrics.duration = job_state.duration
                job.metrics.current_step = job_state.current_step
                job.metrics.total_steps = job_state.total_steps
                job.metrics.progress_percent = job_state.progress_percent
                job.metrics.loss = job_state.loss
                job.metrics.learning_rate = job_state.learning_rate
                job.metrics.epoch = job_state.epoch
                job.metrics.output_dir = job_state.output_dir
                job.metrics.log_file = job_state.log_file

        logger.info(f"Loaded workflow state for {workflow_id}")
        return workflow

    async def run(
        self,
        resource_manager: Optional[ResourceManager] = None,
        on_job_update: Optional[callable] = None,
    ) -> bool:
        """Execute the workflow.

        Args:
            resource_manager: Resource manager for job allocation
            on_job_update: Callback when job status/metrics update

        Returns:
            True if workflow completed successfully, False otherwise
        """
        # Create resource manager if not provided
        if resource_manager is None:
            resource_manager = ResourceManager(
                local_gpus=self.config.resources.gpus or None,
                max_parallel=self.config.resources.max_parallel,
                discover_gpus=not self.config.resources.gpus,
            )

        # Register remote resources
        for remote in self.config.resources.remote:
            resource_manager.register_remote(remote.name, remote.max_jobs)

        # Create executor
        executor = WorkflowExecutor(
            workflow=self,
            resource_manager=resource_manager,
            on_job_update=on_job_update,
        )

        # Execute workflow
        try:
            success = await executor.execute()
            return success
        except Exception as e:
            logger.exception(f"Workflow execution failed: {e}")
            return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Workflow(name={self.config.name}, "
            f"jobs={len(self.jobs)}, "
            f"progress={self.progress:.1f}%)"
        )
