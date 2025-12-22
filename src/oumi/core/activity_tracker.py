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

"""Activity tracking for Oumi CLI commands."""

from __future__ import annotations

import functools
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from oumi.utils.io_utils import load_json, save_json
from oumi.utils.logging import logger

OUMI_DIR = Path(os.environ.get("OUMI_DIR", "~/.oumi")).expanduser()
ACTIVITY_FILE = OUMI_DIR / "activity.json"

ActivityStatus = Literal["completed", "failed", "cancelled", "running"]


@dataclass
class Activity:
    """Represents a single CLI command invocation."""

    id: str
    command: str
    timestamp: str  # ISO 8601 format
    status: ActivityStatus
    duration_seconds: float | None = None
    subcommand: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Activity:
        """Create Activity from dictionary."""
        return cls(
            id=data["id"],
            command=data["command"],
            timestamp=data["timestamp"],
            status=data.get("status", "completed"),
            duration_seconds=data.get("duration_seconds"),
            subcommand=data.get("subcommand"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ActivityStore:
    """Container for activity data with version tracking."""

    version: int = 1
    activities: list[Activity] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "activities": [a.to_dict() for a in self.activities],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActivityStore:
        """Create ActivityStore from dictionary."""
        return cls(
            version=data.get("version", 1),
            activities=[Activity.from_dict(a) for a in data.get("activities", [])],
        )


class ActivityTracker:
    """Tracks CLI command usage across all Oumi commands.

    Usage:
        tracker = ActivityTracker()
        activity_id = tracker.start_activity("train", metadata={"model": "llama"})
        try:
            # ... do work ...
            tracker.complete_activity(activity_id, "completed", {"loss": 0.5})
        except Exception:
            tracker.complete_activity(activity_id, "failed")
            raise
    """

    def __init__(self, storage_path: Path | None = None):
        """Initialize the activity tracker.

        Args:
            storage_path: Path to the activity JSON file. Defaults to ~/.oumi/activity.json
        """
        self._storage_path = storage_path or ACTIVITY_FILE
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: ActivityStore | None = None
        self._start_times: dict[str, float] = {}

    def _is_disabled(self) -> bool:
        """Check if activity tracking is disabled."""
        return os.environ.get("OUMI_DISABLE_STATS", "").lower() in ("1", "true", "yes")

    def _load_store(self) -> ActivityStore:
        """Load the activity store from disk."""
        if self._store is not None:
            return self._store

        try:
            data = load_json(self._storage_path)
            self._store = ActivityStore.from_dict(data)
        except FileNotFoundError:
            self._store = ActivityStore()
        except Exception as e:
            logger.warning(f"Failed to load activity store: {e}")
            self._store = ActivityStore()

        return self._store

    def _save_store(self) -> None:
        """Save the activity store to disk."""
        if self._store is None:
            return

        try:
            save_json(self._store.to_dict(), self._storage_path)
        except Exception as e:
            logger.warning(f"Failed to save activity store: {e}")

    def start_activity(
        self,
        command: str,
        subcommand: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record start of a command execution.

        Args:
            command: The CLI command name (e.g., "train", "infer")
            subcommand: Optional subcommand (e.g., "dataset" for "judge dataset")
            metadata: Optional metadata about the command

        Returns:
            activity_id: Unique identifier for this activity
        """
        if self._is_disabled():
            return ""

        activity_id = uuid.uuid4().hex[:8]
        self._start_times[activity_id] = time.time()

        activity = Activity(
            id=activity_id,
            command=command,
            subcommand=subcommand,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="running",
            metadata=metadata or {},
        )

        store = self._load_store()
        store.activities.append(activity)
        self._save_store()

        logger.debug(f"Started activity {activity_id} for command '{command}'")
        return activity_id

    def complete_activity(
        self,
        activity_id: str,
        status: ActivityStatus = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record completion of a command execution.

        Args:
            activity_id: ID returned from start_activity
            status: Final status of the command
            metadata: Additional metadata to merge (e.g., final_loss)
        """
        if self._is_disabled() or not activity_id:
            return

        store = self._load_store()

        # Find the activity
        for activity in store.activities:
            if activity.id == activity_id:
                activity.status = status

                # Calculate duration
                start_time = self._start_times.pop(activity_id, None)
                if start_time is not None:
                    activity.duration_seconds = time.time() - start_time

                # Merge metadata
                if metadata:
                    activity.metadata.update(metadata)

                self._save_store()
                logger.debug(
                    f"Completed activity {activity_id} with status '{status}'"
                )
                return

        logger.warning(f"Activity {activity_id} not found")

    def get_activities(
        self,
        command: str | None = None,
        since: datetime | None = None,
        status: ActivityStatus | None = None,
        limit: int | None = None,
    ) -> list[Activity]:
        """Query activities with optional filters.

        Args:
            command: Filter by command name
            since: Filter to activities after this datetime
            status: Filter by status
            limit: Maximum number of activities to return (most recent first)

        Returns:
            List of matching activities
        """
        store = self._load_store()
        activities = store.activities

        if command:
            activities = [a for a in activities if a.command == command]

        if since:
            since_iso = since.isoformat()
            activities = [a for a in activities if a.timestamp >= since_iso]

        if status:
            activities = [a for a in activities if a.status == status]

        # Sort by timestamp descending (most recent first)
        activities = sorted(activities, key=lambda a: a.timestamp, reverse=True)

        if limit:
            activities = activities[:limit]

        return activities

    def get_activity_counts(self) -> dict[str, int]:
        """Get count of activities per command.

        Returns:
            Dictionary mapping command names to activity counts
        """
        store = self._load_store()
        counts: dict[str, int] = {}
        for activity in store.activities:
            counts[activity.command] = counts.get(activity.command, 0) + 1
        return counts

    def clear(self) -> int:
        """Clear all activity history.

        Returns:
            Number of activities cleared
        """
        store = self._load_store()
        count = len(store.activities)
        self._store = ActivityStore()
        self._save_store()
        return count


def track_activity(
    command: str,
    subcommand: str | None = None,
    metadata_extractor: Callable[..., dict[str, Any]] | None = None,
) -> Callable:
    """Decorator to automatically track activity for a CLI command.

    Args:
        command: The command name to track
        subcommand: Optional subcommand name
        metadata_extractor: Optional function to extract metadata from command args

    Returns:
        Decorated function

    Example:
        @track_activity("train")
        def train(ctx: typer.Context, config: str):
            ...

        @track_activity("train", metadata_extractor=lambda config: {"config": config})
        def train(config: str):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = ActivityTracker()

            # Extract metadata if extractor provided
            metadata = {}
            if metadata_extractor:
                try:
                    metadata = metadata_extractor(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Failed to extract metadata: {e}")

            activity_id = tracker.start_activity(command, subcommand, metadata)

            try:
                result = func(*args, **kwargs)
                tracker.complete_activity(activity_id, "completed")
                return result
            except KeyboardInterrupt:
                tracker.complete_activity(activity_id, "cancelled")
                raise
            except Exception:
                tracker.complete_activity(activity_id, "failed")
                raise

        return wrapper

    return decorator
