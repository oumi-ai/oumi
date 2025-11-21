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

"""CLI commands for workflow management."""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import typer

from oumi.cli.cli_utils import CONSOLE
from oumi.utils.logging import logger
from oumi.workflow.config import WorkflowConfig
from oumi.workflow.workflow import Workflow

# Check if textual is available
try:
    from oumi.workflow.tui.app import run_workflow_tui

    HAS_TEXTUAL = True
except ImportError:
    HAS_TEXTUAL = False
    logger.warning("Textual not installed. TUI will not be available.")


# Global variable to track shutdown request
_shutdown_requested = False


def setup_signal_handlers(workflow: Workflow) -> None:
    """Setup signal handlers for graceful shutdown.

    Args:
        workflow: Workflow to shutdown gracefully
    """
    global _shutdown_requested

    def shutdown_handler(signum, frame):
        """Handle shutdown signals."""
        global _shutdown_requested
        if _shutdown_requested:
            CONSOLE.print("\n[red]Force shutdown - terminating immediately[/red]")
            sys.exit(1)

        _shutdown_requested = True
        CONSOLE.print(
            "\n[yellow]Shutdown requested - cleaning up (Ctrl+C again to force)[/yellow]"
        )

        # Cancel all running jobs
        loop = asyncio.get_event_loop()
        if loop.is_running():
            for job in workflow.get_running_jobs():
                loop.create_task(job.cancel())

    # Register signal handlers
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)


def run(
    config: str = typer.Argument(
        ...,
        help="Path to workflow configuration YAML file",
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="Disable TUI and run in headless mode",
    ),
    gpus: Optional[str] = typer.Option(
        None,
        "--gpus",
        help="Comma-separated list of GPU indices to use (e.g., '0,1,2')",
    ),
    max_parallel: Optional[int] = typer.Option(
        None,
        "--max-parallel",
        help="Maximum number of parallel jobs",
    ),
) -> None:
    """Execute a workflow from a configuration file.

    Example:
        oumi workflow run my_workflow.yaml

        oumi workflow run my_workflow.yaml --gpus 0,1,2 --max-parallel 3

        oumi workflow run my_workflow.yaml --no-tui
    """
    try:
        # Load workflow config
        config_path = Path(config)
        if not config_path.exists():
            CONSOLE.print(f"[red]Error: Workflow config not found: {config}[/red]")
            sys.exit(1)

        CONSOLE.print(f"[green]Loading workflow configuration from {config}[/green]")
        workflow_config = WorkflowConfig.from_yaml(config_path)

        # Override GPU config if provided
        if gpus:
            gpu_list = [int(g.strip()) for g in gpus.split(",")]
            workflow_config.resources.gpus = gpu_list
            CONSOLE.print(f"[yellow]Using GPUs: {gpu_list}[/yellow]")

        # Override max parallel if provided
        if max_parallel:
            workflow_config.resources.max_parallel = max_parallel
            CONSOLE.print(f"[yellow]Max parallel jobs: {max_parallel}[/yellow]")

        # Override TUI setting if flag provided
        if no_tui:
            workflow_config.tui = False

        # Validate workflow
        errors = workflow_config.validate()
        if errors:
            CONSOLE.print("[red]Workflow configuration errors:[/red]")
            for error in errors:
                CONSOLE.print(f"  - {error}")
            sys.exit(1)

        # Create workflow
        workflow = Workflow(workflow_config)

        # Save initial state
        workflow.save_state(config_path=str(config_path.absolute()))

        CONSOLE.print(f"[green]Workflow loaded: {workflow_config.name}[/green]")
        CONSOLE.print(f"  Jobs: {len(workflow.jobs)}")
        CONSOLE.print(f"  Resources: {len(workflow_config.resources.gpus)} GPUs")

        # Show cost estimate if configured
        if workflow_config.resources.cost_per_gpu_hour:
            estimated_cost = workflow.estimate_cost()
            CONSOLE.print(
                f"  [yellow]Estimated cost: ${estimated_cost:.2f} "
                f"(@ ${workflow_config.resources.cost_per_gpu_hour:.2f}/GPU-hour)[/yellow]"
            )

        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(workflow)

        # Run workflow
        if workflow_config.tui and HAS_TEXTUAL:
            # Run with TUI
            CONSOLE.print("[green]Starting workflow with TUI...[/green]")
            success = asyncio.run(run_workflow_tui(workflow))
        else:
            # Run headless
            if workflow_config.tui and not HAS_TEXTUAL:
                CONSOLE.print(
                    "[yellow]Warning: Textual not installed, running in headless mode[/yellow]"
                )

            CONSOLE.print("[green]Starting workflow (headless mode)...[/green]")
            success = asyncio.run(workflow.run())

        # Save final workflow state
        workflow.save_state(config_path=str(config_path.absolute()))

        # Report results
        if success:
            CONSOLE.print("[green]Workflow completed successfully![/green]")
            # Show actual cost if configured
            if workflow_config.resources.cost_per_gpu_hour:
                actual_cost = workflow.get_actual_cost()
                CONSOLE.print(f"[yellow]Actual cost: ${actual_cost:.2f}[/yellow]")
            CONSOLE.print(f"\n[dim]Workflow ID: {workflow_config.id}[/dim]")
            sys.exit(0)
        else:
            CONSOLE.print("[red]Workflow failed![/red]")
            failed_jobs = workflow.get_failed_jobs()
            if failed_jobs:
                CONSOLE.print("[red]Failed jobs:[/red]")
                for job in failed_jobs:
                    CONSOLE.print(f"  - {job.id}: {job.status}")
                    if job.result and job.result.error:
                        CONSOLE.print(f"    Error: {job.result.error[:200]}")
            CONSOLE.print(f"\n[dim]Workflow ID: {workflow_config.id}[/dim]")
            sys.exit(1)

    except Exception as e:
        CONSOLE.print(f"[red]Error executing workflow: {e}[/red]")
        logger.exception("Workflow execution error")
        sys.exit(1)


def validate(
    config: str = typer.Argument(
        ...,
        help="Path to workflow configuration YAML file",
    ),
) -> None:
    """Validate a workflow configuration file.

    Example:
        oumi workflow validate my_workflow.yaml
    """
    try:
        # Load workflow config
        config_path = Path(config)
        if not config_path.exists():
            CONSOLE.print(f"[red]Error: Workflow config not found: {config}[/red]")
            sys.exit(1)

        CONSOLE.print(f"[green]Validating workflow configuration: {config}[/green]")
        workflow_config = WorkflowConfig.from_yaml(config_path)

        # Validate
        errors = workflow_config.validate()

        if errors:
            CONSOLE.print("[red]Validation failed:[/red]")
            for error in errors:
                CONSOLE.print(f"  - {error}")
            sys.exit(1)
        else:
            CONSOLE.print("[green]Workflow configuration is valid![/green]")
            CONSOLE.print(f"  Name: {workflow_config.name}")
            CONSOLE.print(f"  Jobs: {len(workflow_config.jobs)}")
            for job in workflow_config.jobs:
                deps = (
                    f" (depends on: {', '.join(job.depends_on)})"
                    if job.depends_on
                    else ""
                )
                CONSOLE.print(f"    - {job.name}: {job.verb}{deps}")
            sys.exit(0)

    except Exception as e:
        CONSOLE.print(f"[red]Error validating workflow: {e}[/red]")
        logger.exception("Workflow validation error")
        sys.exit(1)


def status(
    workflow_id: Optional[str] = typer.Argument(
        None,
        help="Workflow ID to show status for (if not provided, lists all workflows)",
    ),
    running_only: bool = typer.Option(
        False,
        "--running",
        help="Show only running workflows",
    ),
) -> None:
    """Show status of workflows.

    Example:
        oumi workflow status

        oumi workflow status --running

        oumi workflow status <workflow-id>
    """
    from oumi.workflow.state import WorkflowStateManager, WorkflowStatus

    try:
        state_manager = WorkflowStateManager()

        if workflow_id:
            # Show specific workflow
            state = state_manager.load_state(workflow_id)
            if not state:
                CONSOLE.print(f"[red]Workflow not found: {workflow_id}[/red]")
                sys.exit(1)

            CONSOLE.print(f"\n[bold]Workflow: {state.name}[/bold]")
            CONSOLE.print(f"  ID: {state.workflow_id}")
            CONSOLE.print(f"  Status: {state.status.value}")
            CONSOLE.print(f"  Config: {state.config_path}")
            CONSOLE.print(f"  Created: {state.created_at}")

            if state.started_at:
                from datetime import datetime

                started = datetime.fromtimestamp(state.started_at)
                CONSOLE.print(f"  Started: {started}")

            if state.completed_at:
                from datetime import datetime

                completed = datetime.fromtimestamp(state.completed_at)
                CONSOLE.print(f"  Completed: {completed}")

            if state.duration:
                hours = state.duration / 3600
                CONSOLE.print(f"  Duration: {hours:.2f} hours")

            CONSOLE.print(f"\n  Jobs: {len(state.jobs)}")
            CONSOLE.print(f"  Progress: {state.progress_percent:.1f}%")

            # Show job statuses
            CONSOLE.print("\n  Job Status:")
            from oumi.workflow.job import JobStatus

            for job_id, job_state in state.jobs.items():
                status_color = {
                    JobStatus.COMPLETED: "green",
                    JobStatus.RUNNING: "yellow",
                    JobStatus.FAILED: "red",
                    JobStatus.PENDING: "white",
                }.get(job_state.status, "white")

                CONSOLE.print(
                    f"    - {job_id}: [{status_color}]{job_state.status.value}[/{status_color}]"
                )
                if job_state.progress_percent > 0:
                    CONSOLE.print(f"      Progress: {job_state.progress_percent:.1f}%")

        else:
            # List workflows
            if running_only:
                workflows = state_manager.get_running_workflows()
            else:
                workflows = state_manager.list_workflows(limit=50)

            if not workflows:
                CONSOLE.print("[yellow]No workflows found[/yellow]")
                return

            CONSOLE.print(f"\n[bold]Workflows ({len(workflows)}):[/bold]\n")

            for state in workflows:
                from datetime import datetime

                created = datetime.fromtimestamp(state.created_at)
                status_color = {
                    WorkflowStatus.COMPLETED: "green",
                    WorkflowStatus.RUNNING: "yellow",
                    WorkflowStatus.FAILED: "red",
                    WorkflowStatus.PENDING: "white",
                }.get(state.status, "white")

                CONSOLE.print(
                    f"  [{status_color}]{state.status.value:10}[/{status_color}] "
                    f"{state.name:30}\n"
                    f"  [dim]ID: {state.workflow_id}[/dim] "
                    f"({state.progress_percent:.0f}%) {created.strftime('%Y-%m-%d %H:%M')}"
                )

            CONSOLE.print(
                "\n[dim]Use 'oumi workflow status <id>' to see details[/dim]"
                "\n[dim]Use 'oumi workflow logs <id>' to view logs[/dim]"
                "\n[dim]Use 'oumi workflow errors <id>' to check for errors[/dim]"
            )

    except Exception as e:
        CONSOLE.print(f"[red]Error getting workflow status: {e}[/red]")
        logger.exception("Error getting workflow status")
        sys.exit(1)


def cancel(
    workflow_id: str = typer.Argument(
        ...,
        help="ID of workflow to cancel",
    ),
) -> None:
    """Cancel a running workflow.

    Example:
        oumi workflow cancel <workflow_id>

    Note: This marks the workflow as cancelled in the database.
    If the workflow is running in another process, you'll need to
    manually stop it (Ctrl+C in the terminal).
    """
    from oumi.workflow.state import WorkflowStateManager, WorkflowStatus

    try:
        state_manager = WorkflowStateManager()

        # Load workflow state
        state = state_manager.load_state(workflow_id)
        if not state:
            CONSOLE.print(f"[red]Workflow not found: {workflow_id}[/red]")
            sys.exit(1)

        if state.status not in (WorkflowStatus.RUNNING, WorkflowStatus.PENDING):
            CONSOLE.print(
                f"[yellow]Workflow is not running (status: {state.status.value})[/yellow]"
            )
            sys.exit(1)

        # Mark as cancelled
        state.status = WorkflowStatus.CANCELLED
        import time

        state.completed_at = time.time()
        state_manager.save_state(state)

        CONSOLE.print(
            f"[yellow]Workflow {workflow_id[:8]}... marked as cancelled[/yellow]"
        )
        CONSOLE.print(
            "[yellow]Note: If the workflow is running in another process, "
            "you'll need to manually stop it (Ctrl+C)[/yellow]"
        )

    except Exception as e:
        CONSOLE.print(f"[red]Error cancelling workflow: {e}[/red]")
        logger.exception("Error cancelling workflow")
        sys.exit(1)


def logs(
    workflow_id: str = typer.Argument(
        ...,
        help="Workflow ID or name to show logs for",
    ),
    job_name: Optional[str] = typer.Argument(
        None,
        help="Specific job name to show logs for (if not provided, shows all jobs)",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Follow log output in real-time (like tail -f)",
    ),
    lines: int = typer.Option(
        50,
        "--lines",
        "-n",
        help="Number of lines to show (default: 50, use 0 for all)",
    ),
    errors_only: bool = typer.Option(
        False,
        "--errors-only",
        "-e",
        help="Show only lines with errors/exceptions",
    ),
) -> None:
    """Show logs for a workflow or specific job.

    Examples:
        # Show last 50 lines from all jobs
        oumi workflow logs <workflow-id>

        # Show logs for specific job
        oumi workflow logs <workflow-id> train-model

        # Show all lines (no limit)
        oumi workflow logs <workflow-id> train-model --lines 0

        # Follow logs in real-time
        oumi workflow logs <workflow-id> train-model --follow

        # Show only errors
        oumi workflow logs <workflow-id> --errors-only
    """
    from oumi.workflow.state import WorkflowStateManager

    try:
        state_manager = WorkflowStateManager()

        # Try to load by ID first
        state = state_manager.load_state(workflow_id)

        # If not found, try to find by name
        if not state:
            workflows = state_manager.list_workflows(limit=100)
            matching = [w for w in workflows if w.name == workflow_id]
            if matching:
                state = matching[0]

        if not state:
            CONSOLE.print(f"[red]Workflow not found: {workflow_id}[/red]")
            sys.exit(1)

        # Determine which jobs to show logs for
        if job_name:
            if job_name not in state.jobs:
                CONSOLE.print(f"[red]Job not found: {job_name}[/red]")
                CONSOLE.print(f"Available jobs: {', '.join(state.jobs.keys())}")
                sys.exit(1)
            jobs_to_show = [job_name]
        else:
            jobs_to_show = list(state.jobs.keys())

        # Show logs for each job
        for job_id in jobs_to_show:
            job_state = state.jobs[job_id]

            if not job_state.log_file:
                CONSOLE.print(f"[yellow]No log file for job: {job_id}[/yellow]")
                continue

            log_path = Path(job_state.log_file)
            if not log_path.exists():
                CONSOLE.print(f"[yellow]Log file not found: {log_path}[/yellow]")
                continue

            # Show header
            CONSOLE.print(f"\n[bold]{'=' * 80}[/bold]")
            CONSOLE.print(f"[bold cyan]Job: {job_id}[/bold cyan]")
            CONSOLE.print(f"[dim]Status: {job_state.status.value}[/dim]")
            CONSOLE.print(f"[dim]Log: {log_path}[/dim]")
            CONSOLE.print(f"[bold]{'=' * 80}[/bold]\n")

            # Follow mode
            if follow:
                CONSOLE.print("[yellow]Following log (Ctrl+C to stop)...[/yellow]\n")
                import subprocess

                try:
                    subprocess.run(["tail", "-f", str(log_path)])
                except KeyboardInterrupt:
                    CONSOLE.print("\n[yellow]Stopped following log[/yellow]")
                return

            # Read and display log
            with open(log_path) as f:
                log_lines = f.readlines()

            # Filter errors only
            if errors_only:
                error_keywords = ["error", "exception", "traceback", "failed", "errno"]
                filtered_lines = []
                for i, line in enumerate(log_lines):
                    if any(keyword in line.lower() for keyword in error_keywords):
                        # Include context (5 lines before and after)
                        start = max(0, i - 5)
                        end = min(len(log_lines), i + 6)
                        filtered_lines.extend(log_lines[start:end])
                        filtered_lines.append("-" * 80 + "\n")
                log_lines = filtered_lines

            # Apply line limit
            if lines > 0:
                log_lines = log_lines[-lines:]

            # Print lines
            for line in log_lines:
                line = line.rstrip()
                # Color code errors
                if any(
                    keyword in line.lower()
                    for keyword in ["error", "exception", "traceback"]
                ):
                    CONSOLE.print(f"[red]{line}[/red]")
                elif "warning" in line.lower():
                    CONSOLE.print(f"[yellow]{line}[/yellow]")
                elif any(
                    keyword in line.lower()
                    for keyword in ["success", "completed", "finished"]
                ):
                    CONSOLE.print(f"[green]{line}[/green]")
                else:
                    CONSOLE.print(line)

            if len(jobs_to_show) > 1:
                CONSOLE.print("")

    except Exception as e:
        CONSOLE.print(f"[red]Error showing logs: {e}[/red]")
        logger.exception("Error showing logs")
        sys.exit(1)


def errors(
    workflow_id: str = typer.Argument(
        ...,
        help="Workflow ID or name to show errors for",
    ),
) -> None:
    """Show all errors from a workflow.

    This is a shortcut for: oumi workflow logs <id> --errors-only

    Example:
        oumi workflow errors <workflow-id>
    """
    from oumi.workflow.state import WorkflowStateManager

    try:
        state_manager = WorkflowStateManager()

        # Try to load by ID first
        state = state_manager.load_state(workflow_id)

        # If not found, try to find by name
        if not state:
            workflows = state_manager.list_workflows(limit=100)
            matching = [w for w in workflows if w.name == workflow_id]
            if matching:
                state = matching[0]

        if not state:
            CONSOLE.print(f"[red]Workflow not found: {workflow_id}[/red]")
            sys.exit(1)

        CONSOLE.print(f"\n[bold]Errors in workflow: {state.name}[/bold]")
        CONSOLE.print(f"[dim]Workflow ID: {state.workflow_id}[/dim]\n")

        found_errors = False

        # Check each job for errors
        for job_id, job_state in state.jobs.items():
            if job_state.error:
                found_errors = True
                CONSOLE.print(f"[bold red]❌ Job: {job_id}[/bold red]")
                CONSOLE.print(f"[red]{job_state.error}[/red]")
                CONSOLE.print("")

            # Also check log file for errors
            if job_state.log_file:
                log_path = Path(job_state.log_file)
                if log_path.exists():
                    with open(log_path) as f:
                        log_content = f.read()

                    # Look for error patterns
                    error_patterns = [
                        "Error:",
                        "Exception:",
                        "Traceback",
                        "FileNotFoundError",
                        "RuntimeError",
                        "ValueError",
                        "FAILED",
                    ]

                    errors_in_log = []
                    for line in log_content.split("\n"):
                        if any(pattern in line for pattern in error_patterns):
                            errors_in_log.append(line)

                    if errors_in_log and not job_state.error:
                        found_errors = True
                        CONSOLE.print(f"[bold yellow]⚠️  Job: {job_id}[/bold yellow]")
                        CONSOLE.print("[yellow]Errors found in log:[/yellow]")
                        for err_line in errors_in_log[:10]:  # Show first 10 error lines
                            CONSOLE.print(f"[yellow]{err_line}[/yellow]")
                        if len(errors_in_log) > 10:
                            CONSOLE.print(
                                f"[dim]... and {len(errors_in_log) - 10} more error lines[/dim]"
                            )
                        CONSOLE.print("")

        if not found_errors:
            CONSOLE.print("[green]✅ No errors found in workflow![/green]")
        else:
            CONSOLE.print("\n[dim]To view full logs:[/dim]")
            CONSOLE.print(f"[dim]  oumi workflow logs {state.workflow_id}[/dim]")

    except Exception as e:
        CONSOLE.print(f"[red]Error showing errors: {e}[/red]")
        logger.exception("Error showing errors")
        sys.exit(1)
