"""Main Textual application for workflow TUI."""

import asyncio
import logging
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Label, Static

from oumi.workflow.job import Job
from oumi.workflow.tui.widgets import JobCard, MetricsPanel, StatusBar
from oumi.workflow.workflow import Workflow

logger = logging.getLogger(__name__)


class WorkflowApp(App):
    """Textual TUI application for workflow monitoring and management."""

    CSS = """
    Screen {
        background: $surface;
    }

    #header {
        dock: top;
        height: 3;
        background: $primary;
        color: $text;
        padding: 1;
    }

    #main-container {
        height: 100%;
    }

    #left-panel {
        width: 60%;
        border-right: solid $primary;
    }

    #right-panel {
        width: 40%;
    }

    #jobs-list {
        height: 100%;
        padding: 1;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        background: $primary-darken-2;
        color: $text;
    }

    .job-card {
        margin-bottom: 1;
        padding: 1;
        border: solid $primary;
        background: $surface-lighten-1;
    }

    .job-card.running {
        border: solid $success;
    }

    .job-card.failed {
        border: solid $error;
    }

    .job-card.completed {
        border: solid $success-darken-1;
    }

    .metrics-panel {
        height: 40%;
        padding: 1;
        border: solid $primary;
        margin: 1;
    }

    .log-viewer {
        height: 60%;
        padding: 1;
        border: solid $primary;
        margin: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "cancel_job", "Cancel Job"),
        ("r", "refresh", "Refresh"),
        ("j", "scroll_down", "Down"),
        ("k", "scroll_up", "Up"),
    ]

    def __init__(
        self,
        workflow: Workflow,
        **kwargs,
    ):
        """Initialize TUI app.

        Args:
            workflow: Workflow to monitor
            **kwargs: Additional Textual app kwargs
        """
        super().__init__(**kwargs)
        self.workflow = workflow
        self._job_cards: dict[str, JobCard] = {}
        self._update_task: Optional[asyncio.Task] = None
        self._workflow_task: Optional[asyncio.Task] = None
        self._selected_job_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header()

        # Main container with two panels
        with Horizontal(id="main-container"):
            # Left panel: Job list
            with Vertical(id="left-panel"):
                yield Label(
                    f"Workflow: {self.workflow.config.name}",
                    id="workflow-title",
                )
                with VerticalScroll(id="jobs-list"):
                    # Job cards will be added dynamically
                    for job in self.workflow.jobs.values():
                        card = JobCard(job)
                        self._job_cards[job.id] = card
                        yield card

            # Right panel: Metrics and logs
            with Vertical(id="right-panel"):
                yield MetricsPanel(
                    workflow=self.workflow,
                    classes="metrics-panel",
                )
                yield Static(
                    "Select a job to view logs",
                    classes="log-viewer",
                    id="log-viewer",
                )

        yield StatusBar(workflow=self.workflow, id="status-bar")
        yield Footer()

    async def on_mount(self) -> None:
        """Handle app mount event."""
        # Start periodic UI updates
        self._update_task = asyncio.create_task(self._update_loop())

        # Start workflow execution in background
        self._workflow_task = asyncio.create_task(self._run_workflow())

    async def on_unmount(self) -> None:
        """Handle app unmount event."""
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        # Wait for workflow to finish or cancel it
        if self._workflow_task and not self._workflow_task.done():
            logger.info("Workflow still running, cancelling...")
            self._workflow_task.cancel()
            try:
                await self._workflow_task
            except asyncio.CancelledError:
                pass

    async def _run_workflow(self) -> None:
        """Run the workflow in background."""
        try:

            def on_job_update(job: Job):
                """Callback when job updates."""
                # Update job card in UI
                if job.id in self._job_cards:
                    # Schedule UI update on main thread
                    self.call_later(self._update_job_card, job.id)

            success = await self.workflow.run(on_job_update=on_job_update)

            if success:
                logger.info("Workflow completed successfully")
            else:
                logger.error("Workflow failed")

        except Exception as e:
            logger.exception(f"Error running workflow: {e}")

    def _update_job_card(self, job_id: str) -> None:
        """Update a job card in the UI.

        Args:
            job_id: Job ID to update
        """
        if job_id in self._job_cards:
            card = self._job_cards[job_id]
            card.update_from_job()

    async def _update_loop(self) -> None:
        """Periodically update the UI."""
        while True:
            try:
                # Update all job cards
                for job_id, card in self._job_cards.items():
                    card.update_from_job()

                # Update status bar
                status_bar = self.query_one("#status-bar", StatusBar)
                status_bar.update_from_workflow()

                # Update metrics panel
                metrics_panel = self.query_one(MetricsPanel)
                metrics_panel.update_metrics()

                await asyncio.sleep(1.0)

            except Exception as e:
                logger.exception(f"Error in update loop: {e}")
                await asyncio.sleep(1.0)

    @on(JobCard.Selected)
    def on_job_selected(self, event: JobCard.Selected) -> None:
        """Handle job card selection.

        Args:
            event: Selection event
        """
        self._selected_job_id = event.job_id

        # Update log viewer
        job = self.workflow.get_job(event.job_id)
        if job and job.metrics.log_file:
            self._update_log_viewer(job.metrics.log_file)

    def _update_log_viewer(self, log_file: str) -> None:
        """Update log viewer with job logs.

        Args:
            log_file: Path to log file
        """
        try:
            with open(log_file) as f:
                # Read last 100 lines
                lines = f.readlines()
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                log_content = "".join(recent_lines)

            log_viewer = self.query_one("#log-viewer", Static)
            log_viewer.update(log_content)

        except Exception as e:
            logger.debug(f"Error reading log file: {e}")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_cancel_job(self) -> None:
        """Cancel the selected job."""
        if self._selected_job_id:
            job = self.workflow.get_job(self._selected_job_id)
            if job and job.is_running:
                asyncio.create_task(job.cancel())
                self.notify(f"Cancelled job: {self._selected_job_id}")

    def action_refresh(self) -> None:
        """Refresh the UI."""
        for card in self._job_cards.values():
            card.update_from_job()

    def action_scroll_down(self) -> None:
        """Scroll job list down."""
        jobs_list = self.query_one("#jobs-list", VerticalScroll)
        jobs_list.scroll_down()

    def action_scroll_up(self) -> None:
        """Scroll job list up."""
        jobs_list = self.query_one("#jobs-list", VerticalScroll)
        jobs_list.scroll_up()


async def run_workflow_tui(workflow: Workflow) -> bool:
    """Run workflow with TUI.

    Args:
        workflow: Workflow to execute

    Returns:
        True if workflow completed successfully
    """
    app = WorkflowApp(workflow=workflow)
    await app.run_async()
    return workflow.is_successful
