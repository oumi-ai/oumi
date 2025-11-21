"""Workflow executor for managing job execution."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from oumi.workflow.job import JobStatus

if TYPE_CHECKING:
    from oumi.workflow.resource_manager import ResourceManager
    from oumi.workflow.workflow import Workflow

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Executes workflows by managing job scheduling and execution."""

    def __init__(
        self,
        workflow: "Workflow",
        resource_manager: "ResourceManager",
        on_job_update: Optional[callable] = None,
    ):
        """Initialize executor.

        Args:
            workflow: Workflow to execute
            resource_manager: Resource manager for job allocation
            on_job_update: Callback for job updates (receives Job)
        """
        self.workflow = workflow
        self.resource_manager = resource_manager
        self.on_job_update = on_job_update

        # Execution state
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._should_stop = False

    async def execute(self) -> bool:
        """Execute the workflow.

        Returns:
            True if workflow completed successfully, False otherwise
        """
        logger.info(f"Starting workflow: {self.workflow.config.name}")

        try:
            # Start scheduler and monitor tasks
            scheduler_task = asyncio.create_task(self._scheduler())
            monitor_task = asyncio.create_task(self._monitor())

            # Wait for workflow to complete or timeout
            if self.workflow.config.timeout:
                await asyncio.wait_for(
                    self._wait_for_completion(),
                    timeout=self.workflow.config.timeout,
                )
            else:
                await self._wait_for_completion()

            # Stop scheduler and monitor
            self._should_stop = True
            scheduler_task.cancel()
            monitor_task.cancel()

            # Wait for tasks to finish
            await asyncio.gather(scheduler_task, monitor_task, return_exceptions=True)

            # Check results
            if self.workflow.is_successful:
                logger.info(
                    f"Workflow {self.workflow.config.name} completed successfully"
                )
                return True
            else:
                failed_jobs = self.workflow.get_failed_jobs()
                logger.error(
                    f"Workflow {self.workflow.config.name} failed. "
                    f"Failed jobs: {[j.id for j in failed_jobs]}"
                )
                return False

        except asyncio.TimeoutError:
            logger.error(
                f"Workflow {self.workflow.config.name} exceeded timeout of "
                f"{self.workflow.config.timeout}s"
            )
            # Cancel all running jobs
            await self._cancel_all_jobs()
            return False

        except Exception as e:
            logger.exception(f"Workflow execution error: {e}")
            await self._cancel_all_jobs()
            return False

    async def _scheduler(self) -> None:
        """Schedule and start jobs as resources become available."""
        logger.info("Scheduler started")

        while not self._should_stop:
            try:
                # Get jobs ready to run
                ready_jobs = self.workflow.get_ready_jobs()

                for job in ready_jobs:
                    # Try to acquire resource for job
                    resource = await self.resource_manager.acquire(
                        requirements=job.config.resources,
                        job_id=job.id,
                        timeout=1.0,  # Short timeout to avoid blocking
                    )

                    if resource:
                        # Assign resource to job
                        if resource.gpu_index is not None:
                            job.assigned_gpu = resource.gpu_index
                        if resource.remote_name:
                            job.assigned_remote = resource.remote_name

                        # Start job
                        task = asyncio.create_task(self._run_job(job, resource))
                        self._running_tasks[job.id] = task
                        logger.info(f"Started job {job.id} on resource {resource.id}")

                        # Notify callback
                        if self.on_job_update:
                            self.on_job_update(job)

                # Update GPU metrics periodically
                await self.resource_manager.update_gpu_metrics()

                # Sleep before next scheduling iteration
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.exception(f"Error in scheduler: {e}")
                await asyncio.sleep(1.0)

        logger.info("Scheduler stopped")

    async def _run_job(self, job, resource) -> None:
        """Run a job and handle completion.

        Args:
            job: Job to run
            resource: Assigned resource
        """
        try:
            # Execute job with progress callback
            def on_progress(metrics):
                if self.on_job_update:
                    self.on_job_update(job)

            result = await job.run(
                env=self.workflow.config.env,
                on_progress=on_progress,
            )

            # Handle result
            if result.status == JobStatus.FAILED and job.can_retry():
                logger.info(
                    f"Job {job.id} failed, retrying "
                    f"({job._retry_count}/{job.config.max_retries})"
                )
                await asyncio.sleep(job.config.retry_delay)
                job.reset()

            # Notify callback
            if self.on_job_update:
                self.on_job_update(job)

        except Exception as e:
            logger.exception(f"Error running job {job.id}: {e}")
            job.status = JobStatus.FAILED

        finally:
            # Release resource
            await self.resource_manager.release(resource, job.id)

            # Remove from running tasks
            if job.id in self._running_tasks:
                del self._running_tasks[job.id]

    async def _monitor(self) -> None:
        """Monitor workflow progress and log updates."""
        logger.info("Monitor started")

        last_progress = -1
        last_save_time = 0.0
        save_interval = 30.0  # Save state every 30 seconds

        while not self._should_stop:
            try:
                # Log progress updates
                progress = self.workflow.progress
                if progress != last_progress:
                    running = len(self.workflow.get_running_jobs())
                    completed = len(self.workflow.get_completed_jobs())
                    logger.info(
                        f"Workflow progress: {progress:.1f}% "
                        f"(running: {running}, completed: {completed}/{len(self.workflow.jobs)})"
                    )
                    last_progress = progress

                # Periodically save workflow state
                import time

                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    try:
                        self.workflow.save_state()
                        last_save_time = current_time
                    except Exception as e:
                        logger.warning(f"Failed to save workflow state: {e}")

                await asyncio.sleep(5.0)

            except Exception as e:
                logger.exception(f"Error in monitor: {e}")
                await asyncio.sleep(5.0)

        logger.info("Monitor stopped")

    async def _wait_for_completion(self) -> None:
        """Wait for workflow to complete."""
        while not self.workflow.is_complete and not self._should_stop:
            await asyncio.sleep(0.5)

    async def _cancel_all_jobs(self) -> None:
        """Cancel all running jobs."""
        logger.info("Cancelling all running jobs")

        cancel_tasks = []
        for job in self.workflow.get_running_jobs():
            cancel_tasks.append(job.cancel())

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job.

        Args:
            job_id: ID of job to cancel

        Returns:
            True if job was cancelled
        """
        job = self.workflow.get_job(job_id)
        if not job:
            logger.warning(f"Job {job_id} not found")
            return False

        if not job.is_running:
            logger.warning(f"Job {job_id} is not running")
            return False

        await job.cancel()

        # Notify callback
        if self.on_job_update:
            self.on_job_update(job)

        return True

    def get_status_summary(self) -> dict:
        """Get summary of workflow status.

        Returns:
            Dict with status information
        """
        return {
            "workflow_name": self.workflow.config.name,
            "total_jobs": len(self.workflow.jobs),
            "pending": sum(
                1 for j in self.workflow.jobs.values() if j.status == JobStatus.PENDING
            ),
            "queued": sum(
                1 for j in self.workflow.jobs.values() if j.status == JobStatus.QUEUED
            ),
            "running": len(self.workflow.get_running_jobs()),
            "completed": len(
                [
                    j
                    for j in self.workflow.jobs.values()
                    if j.status == JobStatus.COMPLETED
                ]
            ),
            "failed": len(self.workflow.get_failed_jobs()),
            "cancelled": sum(
                1
                for j in self.workflow.jobs.values()
                if j.status == JobStatus.CANCELLED
            ),
            "progress": self.workflow.progress,
            "is_complete": self.workflow.is_complete,
            "is_successful": self.workflow.is_successful,
            "available_resources": self.resource_manager.available_count,
            "resources_in_use": self.resource_manager.in_use_count,
        }
