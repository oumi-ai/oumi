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

"""Checkpointing support for resuming interrupted optimization runs."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from oumi.utils.logging import logger


@dataclass
class OptimizationCheckpoint:
    """Checkpoint data for resuming optimization."""

    checkpoint_version: str = "1.0"
    """Version of checkpoint format."""

    timestamp: float = 0.0
    """When the checkpoint was created."""

    optimizer_name: str = ""
    """Name of the optimizer (mipro, gepa, bootstrap)."""

    current_trial: int = 0
    """Current trial number."""

    total_trials: int = 0
    """Total number of trials."""

    best_score: float = 0.0
    """Best score achieved so far."""

    best_prompt: str = ""
    """Best prompt found so far."""

    best_demos: list[dict[str, Any]] | None = None
    """Best few-shot demonstrations found so far."""

    training_history: list[dict[str, Any]] | None = None
    """History of trials and scores."""

    completed: bool = False
    """Whether optimization has completed."""

    metadata: dict[str, Any] | None = None
    """Additional optimizer-specific metadata."""

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.best_demos is None:
            self.best_demos = []
        if self.training_history is None:
            self.training_history = []
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the checkpoint.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationCheckpoint":
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            OptimizationCheckpoint instance.
        """
        return cls(**data)

    def save(self, checkpoint_path: Path) -> None:
        """Save checkpoint to disk.

        Args:
            checkpoint_path: Path to save the checkpoint file.
        """
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to temporary file first, then rename (atomic on most filesystems)
        temp_path = checkpoint_path.with_suffix(".tmp")

        with open(temp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Atomic rename
        temp_path.replace(checkpoint_path)

        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    @classmethod
    def load(cls, checkpoint_path: Path) -> Optional["OptimizationCheckpoint"]:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            OptimizationCheckpoint instance, or None if file doesn't exist or is
            invalid.
        """
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)

            checkpoint = cls.from_dict(data)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(
                f"  Progress: {checkpoint.current_trial}/"
                f"{checkpoint.total_trials} trials"
            )
            logger.info(f"  Best score: {checkpoint.best_score:.4f}")

            return checkpoint

        except Exception as e:
            logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return None


class CheckpointManager:
    """Manager for optimization checkpoints."""

    def __init__(self, output_dir: str, optimizer_name: str):
        """Initialize checkpoint manager.

        Args:
            output_dir: Directory to save checkpoints.
            optimizer_name: Name of the optimizer.
        """
        self.output_dir = Path(output_dir)
        self.optimizer_name = optimizer_name
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.last_checkpoint_time = 0.0
        self.checkpoint_interval = 300.0  # Save every 5 minutes

    def should_save_checkpoint(self) -> bool:
        """Check if enough time has passed to save a checkpoint.

        Returns:
            True if a checkpoint should be saved.
        """
        current_time = time.time()
        return (current_time - self.last_checkpoint_time) >= self.checkpoint_interval

    def save_checkpoint(
        self,
        current_trial: int,
        total_trials: int,
        best_score: float,
        best_prompt: str,
        best_demos: list[dict[str, Any]],
        training_history: list[dict[str, Any]],
        completed: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a checkpoint.

        Args:
            current_trial: Current trial number.
            total_trials: Total number of trials.
            best_score: Best score achieved so far.
            best_prompt: Best prompt found so far.
            best_demos: Best few-shot demonstrations found so far.
            training_history: History of trials and scores.
            completed: Whether optimization has completed.
            metadata: Additional optimizer-specific metadata.
        """
        checkpoint = OptimizationCheckpoint(
            optimizer_name=self.optimizer_name,
            current_trial=current_trial,
            total_trials=total_trials,
            best_score=best_score,
            best_prompt=best_prompt,
            best_demos=best_demos or [],
            training_history=training_history or [],
            completed=completed,
            metadata=metadata or {},
        )

        checkpoint.save(self.checkpoint_path)
        self.last_checkpoint_time = time.time()

        if completed:
            logger.info("Saved final checkpoint (optimization completed)")
        else:
            logger.debug(f"Saved checkpoint at trial {current_trial}/{total_trials}")

    def load_checkpoint(self) -> OptimizationCheckpoint | None:
        """Load the latest checkpoint.

        Returns:
            OptimizationCheckpoint instance, or None if no valid checkpoint exists.
        """
        return OptimizationCheckpoint.load(self.checkpoint_path)

    def clear_checkpoint(self) -> None:
        """Delete the checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.debug(f"Deleted checkpoint file: {self.checkpoint_path}")

    def checkpoint_exists(self) -> bool:
        """Check if a checkpoint file exists.

        Returns:
            True if a checkpoint file exists.
        """
        return self.checkpoint_path.exists()


def can_resume_from_checkpoint(
    checkpoint: OptimizationCheckpoint | None,
    optimizer_name: str,
    num_trials: int,
) -> tuple[bool, str | None]:
    """Check if optimization can be resumed from a checkpoint.

    Args:
        checkpoint: The checkpoint to check.
        optimizer_name: Name of the current optimizer.
        num_trials: Number of trials in the current configuration.

    Returns:
        Tuple of (can_resume, reason_if_not).
    """
    if checkpoint is None:
        return False, "No checkpoint found"

    if checkpoint.completed:
        return False, "Previous optimization already completed"

    if checkpoint.optimizer_name != optimizer_name:
        return (
            False,
            f"Checkpoint is for '{checkpoint.optimizer_name}' optimizer, "
            f"but current optimizer is '{optimizer_name}'",
        )

    if checkpoint.total_trials != num_trials:
        return (
            False,
            f"Checkpoint has {checkpoint.total_trials} trials, "
            f"but current config has {num_trials} trials",
        )

    if checkpoint.current_trial >= checkpoint.total_trials:
        return False, "All trials already completed"

    return True, None


def print_checkpoint_summary(checkpoint: OptimizationCheckpoint) -> None:
    """Print a summary of the checkpoint.

    Args:
        checkpoint: The checkpoint to summarize.
    """
    logger.info("\n" + "=" * 80)
    logger.info("RESUMING FROM CHECKPOINT")
    logger.info("=" * 80)
    logger.info(f"Optimizer: {checkpoint.optimizer_name}")
    logger.info(
        f"Progress: {checkpoint.current_trial}/{checkpoint.total_trials} "
        f"trials completed"
    )
    logger.info(
        f"Remaining: {checkpoint.total_trials - checkpoint.current_trial} trials"
    )
    logger.info(f"Best score so far: {checkpoint.best_score:.4f}")

    if checkpoint.best_prompt:
        logger.info(f"Best prompt: {checkpoint.best_prompt[:100]}...")

    if checkpoint.training_history:
        logger.info(f"Training history: {len(checkpoint.training_history)} entries")

    # Calculate time since last checkpoint
    time_since = time.time() - checkpoint.timestamp
    if time_since < 3600:
        logger.info(f"Last checkpoint: {time_since / 60:.1f} minutes ago")
    else:
        logger.info(f"Last checkpoint: {time_since / 3600:.1f} hours ago")

    logger.info("=" * 80 + "\n")
