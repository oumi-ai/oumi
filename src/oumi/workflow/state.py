"""Workflow state persistence and tracking."""

import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from oumi.workflow.job import JobStatus


class WorkflowStatus(str, Enum):
    """Overall workflow status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class JobState:
    """Persisted state for a single job."""

    job_id: str
    status: JobStatus
    assigned_gpu: Optional[int] = None
    assigned_remote: Optional[str] = None
    retry_count: int = 0

    # Metrics
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    current_step: int = 0
    total_steps: Optional[int] = None
    progress_percent: float = 0.0
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[int] = None

    # Artifacts
    artifacts: dict[str, str] = field(default_factory=dict)

    # Output
    output_dir: Optional[str] = None
    log_file: Optional[str] = None

    # Result
    error: Optional[str] = None
    return_code: Optional[int] = None


@dataclass
class WorkflowState:
    """Persisted state for an entire workflow."""

    workflow_id: str
    name: str
    config_path: str
    status: WorkflowStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    jobs: dict[str, JobState] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get workflow duration in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    @property
    def progress_percent(self) -> float:
        """Get overall progress percentage."""
        if not self.jobs:
            return 0.0
        completed = sum(
            1
            for j in self.jobs.values()
            if j.status
            in (
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.TIMEOUT,
            )
        )
        return (completed / len(self.jobs)) * 100


class WorkflowStateManager:
    """Manages workflow state persistence using SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize state manager.

        Args:
            db_path: Path to SQLite database (default: ~/.oumi/workflows.db)
        """
        if db_path is None:
            db_path = Path.home() / ".oumi" / "workflows.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    state_json TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON workflows(created_at DESC)
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_status
                ON workflows(status)
            """
            )
            conn.commit()

    def save_state(self, state: WorkflowState) -> None:
        """Save workflow state to database.

        Args:
            state: Workflow state to save
        """
        state_json = json.dumps(self._state_to_dict(state))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workflows
                (workflow_id, name, config_path, status, created_at, started_at, completed_at, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    state.workflow_id,
                    state.name,
                    state.config_path,
                    state.status.value,
                    state.created_at,
                    state.started_at,
                    state.completed_at,
                    state_json,
                ),
            )
            conn.commit()

    def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state from database.

        Args:
            workflow_id: Workflow ID to load

        Returns:
            WorkflowState or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT state_json FROM workflows
                WHERE workflow_id = ?
            """,
                (workflow_id,),
            )
            row = cursor.fetchone()

        if row:
            state_dict = json.loads(row[0])
            return self._dict_to_state(state_dict)
        return None

    def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
    ) -> list[WorkflowState]:
        """List workflows, optionally filtered by status.

        Args:
            status: Filter by status (None = all)
            limit: Maximum number of workflows to return

        Returns:
            List of workflow states, most recent first
        """
        with sqlite3.connect(self.db_path) as conn:
            if status:
                cursor = conn.execute(
                    """
                    SELECT state_json FROM workflows
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (status.value, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT state_json FROM workflows
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            results = []
            for row in cursor.fetchall():
                state_dict = json.loads(row[0])
                results.append(self._dict_to_state(state_dict))

            return results

    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete workflow state from database.

        Args:
            workflow_id: Workflow ID to delete

        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM workflows
                WHERE workflow_id = ?
            """,
                (workflow_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def cleanup_old_workflows(self, days: int = 30) -> int:
        """Delete workflow states older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of workflows deleted
        """
        cutoff = time.time() - (days * 86400)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM workflows
                WHERE created_at < ? AND status IN ('completed', 'failed', 'cancelled')
            """,
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount

    def get_running_workflows(self) -> list[WorkflowState]:
        """Get all currently running workflows.

        Returns:
            List of running workflow states
        """
        return self.list_workflows(status=WorkflowStatus.RUNNING)

    def _state_to_dict(self, state: WorkflowState) -> dict[str, Any]:
        """Convert WorkflowState to dict for serialization."""
        jobs_dict = {}
        for job_id, job_state in state.jobs.items():
            job_dict = asdict(job_state)
            job_dict["status"] = job_state.status.value
            jobs_dict[job_id] = job_dict

        return {
            "workflow_id": state.workflow_id,
            "name": state.name,
            "config_path": state.config_path,
            "status": state.status.value,
            "created_at": state.created_at,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "jobs": jobs_dict,
        }

    def _dict_to_state(self, data: dict[str, Any]) -> WorkflowState:
        """Convert dict to WorkflowState."""
        jobs = {}
        for job_id, job_dict in data.get("jobs", {}).items():
            job_dict["status"] = JobStatus(job_dict["status"])
            jobs[job_id] = JobState(**job_dict)

        return WorkflowState(
            workflow_id=data["workflow_id"],
            name=data["name"],
            config_path=data["config_path"],
            status=WorkflowStatus(data["status"]),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            jobs=jobs,
        )
