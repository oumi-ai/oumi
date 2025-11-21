"""Custom widgets for workflow TUI."""

from typing import TYPE_CHECKING

from rich.progress import Progress, TaskID
from rich.table import Table
from textual.message import Message
from textual.widgets import Static

from oumi.workflow.job import JobStatus

if TYPE_CHECKING:
    from oumi.workflow.job import Job
    from oumi.workflow.workflow import Workflow


class JobCard(Static):
    """Widget displaying a single job's status and progress."""

    class Selected(Message):
        """Message sent when job card is selected."""

        def __init__(self, job_id: str):
            self.job_id = job_id
            super().__init__()

    def __init__(self, job: "Job", **kwargs):
        """Initialize job card.

        Args:
            job: Job to display
            **kwargs: Additional widget kwargs
        """
        super().__init__(**kwargs)
        self.job = job
        self._progress_bar: Progress | None = None
        self._progress_task: TaskID | None = None

    def on_mount(self) -> None:
        """Handle mount event."""
        self.update_from_job()

    def update_from_job(self) -> None:
        """Update display from current job state."""
        # Build status indicator
        status_colors = {
            JobStatus.PENDING: "yellow",
            JobStatus.WAITING: "yellow",
            JobStatus.QUEUED: "cyan",
            JobStatus.RUNNING: "green",
            JobStatus.COMPLETED: "blue",
            JobStatus.FAILED: "red",
            JobStatus.CANCELLED: "gray",
            JobStatus.TIMEOUT: "red",
        }
        color = status_colors.get(self.job.status, "white")

        # Create table for job info
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        # Job name and status
        table.add_row(
            "Job",
            f"[{color}]{self.job.config.name}[/{color}]",
        )
        table.add_row(
            "Status",
            f"[{color}]{self.job.status.value.upper()}[/{color}]",
        )
        table.add_row("Verb", self.job.config.verb)

        # Resource info
        if self.job.assigned_gpu is not None:
            table.add_row("GPU", f"GPU:{self.job.assigned_gpu}")
        elif self.job.assigned_remote:
            table.add_row("Resource", self.job.assigned_remote)

        # Progress info
        if self.job.is_running and self.job.metrics.total_steps:
            progress_text = (
                f"{self.job.metrics.current_step}/{self.job.metrics.total_steps} "
                f"({self.job.metrics.progress_percent:.1f}%)"
            )
            table.add_row("Progress", progress_text)

        # Metrics
        if self.job.metrics.loss is not None:
            table.add_row("Loss", f"{self.job.metrics.loss:.4f}")

        if self.job.metrics.learning_rate is not None:
            table.add_row("LR", f"{self.job.metrics.learning_rate:.2e}")

        if self.job.metrics.duration is not None:
            duration_min = self.job.metrics.duration / 60
            table.add_row("Duration", f"{duration_min:.1f}m")

        # Error info
        if self.job.result and self.job.result.error:
            error_preview = (
                self.job.result.error[:50] + "..."
                if len(self.job.result.error) > 50
                else self.job.result.error
            )
            table.add_row("Error", f"[red]{error_preview}[/red]")

        # Update widget content
        self.update(table)

        # Update CSS class based on status
        self.remove_class("running", "failed", "completed")
        if self.job.status == JobStatus.RUNNING:
            self.add_class("running")
        elif self.job.status in (JobStatus.FAILED, JobStatus.TIMEOUT):
            self.add_class("failed")
        elif self.job.status == JobStatus.COMPLETED:
            self.add_class("completed")

    async def on_click(self) -> None:
        """Handle click event."""
        self.post_message(self.Selected(self.job.id))


class StatusBar(Static):
    """Status bar showing overall workflow progress."""

    def __init__(self, workflow: "Workflow", **kwargs):
        """Initialize status bar.

        Args:
            workflow: Workflow to monitor
            **kwargs: Additional widget kwargs
        """
        super().__init__(**kwargs)
        self.workflow = workflow

    def on_mount(self) -> None:
        """Handle mount event."""
        self.update_from_workflow()

    def update_from_workflow(self) -> None:
        """Update display from workflow state."""
        total = len(self.workflow.jobs)
        pending = sum(
            1 for j in self.workflow.jobs.values() if j.status == JobStatus.PENDING
        )
        queued = sum(
            1 for j in self.workflow.jobs.values() if j.status == JobStatus.QUEUED
        )
        running = len(self.workflow.get_running_jobs())
        completed = len(
            [j for j in self.workflow.jobs.values() if j.status == JobStatus.COMPLETED]
        )
        failed = len(self.workflow.get_failed_jobs())

        status_text = (
            f"Total: {total} | "
            f"Pending: {pending} | "
            f"Queued: {queued} | "
            f"Running: [green]{running}[/green] | "
            f"Completed: [blue]{completed}[/blue] | "
            f"Failed: [red]{failed}[/red] | "
            f"Progress: {self.workflow.progress:.1f}%"
        )

        self.update(status_text)


class MetricsPanel(Static):
    """Panel showing aggregated workflow metrics."""

    def __init__(self, workflow: "Workflow", **kwargs):
        """Initialize metrics panel.

        Args:
            workflow: Workflow to monitor
            **kwargs: Additional widget kwargs
        """
        super().__init__(**kwargs)
        self.workflow = workflow

    def on_mount(self) -> None:
        """Handle mount event."""
        self.update_metrics()

    def update_metrics(self) -> None:
        """Update metrics display."""
        table = Table(title="Workflow Metrics", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        # Overall progress
        table.add_row("Overall Progress", f"{self.workflow.progress:.1f}%")

        # Job counts by status
        running_jobs = self.workflow.get_running_jobs()
        completed_jobs = self.workflow.get_completed_jobs()
        failed_jobs = self.workflow.get_failed_jobs()

        table.add_row("Running Jobs", str(len(running_jobs)))
        table.add_row("Completed Jobs", str(len(completed_jobs)))
        table.add_row("Failed Jobs", str(len(failed_jobs)))

        # Average metrics from running jobs
        if running_jobs:
            avg_progress = sum(j.metrics.progress_percent for j in running_jobs) / len(
                running_jobs
            )
            table.add_row("Avg Job Progress", f"{avg_progress:.1f}%")

            losses = [
                j.metrics.loss for j in running_jobs if j.metrics.loss is not None
            ]
            if losses:
                avg_loss = sum(losses) / len(losses)
                table.add_row("Avg Loss", f"{avg_loss:.4f}")

        # Total duration
        completed_durations = [
            j.metrics.duration for j in completed_jobs if j.metrics.duration is not None
        ]
        if completed_durations:
            total_duration = sum(completed_durations)
            table.add_row("Total Completed Time", f"{total_duration / 60:.1f}m")

        self.update(table)


class LogViewer(Static):
    """Widget for displaying job logs."""

    def __init__(self, **kwargs):
        """Initialize log viewer.

        Args:
            **kwargs: Additional widget kwargs
        """
        super().__init__(**kwargs)
        self._log_lines: list[str] = []
        self._max_lines = 1000

    def add_log_line(self, line: str) -> None:
        """Add a log line.

        Args:
            line: Log line to add
        """
        self._log_lines.append(line)
        if len(self._log_lines) > self._max_lines:
            self._log_lines = self._log_lines[-self._max_lines :]

        self._update_display()

    def clear_logs(self) -> None:
        """Clear all logs."""
        self._log_lines = []
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current logs."""
        # Show last 50 lines
        visible_lines = (
            self._log_lines[-50:] if len(self._log_lines) > 50 else self._log_lines
        )
        self.update("\n".join(visible_lines))
