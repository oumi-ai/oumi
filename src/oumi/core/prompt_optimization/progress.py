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

"""Progress tracking and monitoring for prompt optimization."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from tqdm import tqdm


@dataclass
class OptimizationStats:
    """Statistics tracked during optimization."""

    start_time: float = field(default_factory=time.time)
    """Start time of optimization."""

    end_time: Optional[float] = None
    """End time of optimization."""

    num_examples_processed: int = 0
    """Number of examples processed."""

    num_inference_calls: int = 0
    """Number of inference calls made."""

    num_failed_calls: int = 0
    """Number of failed inference calls."""

    total_input_tokens: int = 0
    """Total input tokens used (if available)."""

    total_output_tokens: int = 0
    """Total output tokens generated (if available)."""

    current_trial: int = 0
    """Current trial number."""

    best_score: float = 0.0
    """Best score achieved so far."""

    metadata: dict[str, Any] = field(default_factory=dict)  # type: ignore[misc]
    """Additional metadata."""

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time

    def get_estimated_total_time(self, total_trials: int) -> Optional[float]:
        """Estimate total time based on current progress.

        Args:
            total_trials: Total number of trials to run.

        Returns:
            Estimated total time in seconds, or None if not enough data.
        """
        if self.current_trial == 0:
            return None

        elapsed = self.get_elapsed_time()
        time_per_trial = elapsed / self.current_trial
        return time_per_trial * total_trials

    def get_remaining_time(self, total_trials: int) -> Optional[float]:
        """Estimate remaining time based on current progress.

        Args:
            total_trials: Total number of trials to run.

        Returns:
            Estimated remaining time in seconds, or None if not enough data.
        """
        if self.current_trial == 0:
            return None

        elapsed = self.get_elapsed_time()
        time_per_trial = elapsed / self.current_trial
        remaining_trials = total_trials - self.current_trial
        return time_per_trial * remaining_trials

    def get_success_rate(self) -> float:
        """Get inference success rate.

        Returns:
            Success rate between 0 and 1.
        """
        if self.num_inference_calls == 0:
            return 0.0
        return (
            self.num_inference_calls - self.num_failed_calls
        ) / self.num_inference_calls

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary.

        Returns:
            Dictionary representation of stats.
        """
        return {
            "elapsed_time_seconds": self.get_elapsed_time(),
            "num_examples_processed": self.num_examples_processed,
            "num_inference_calls": self.num_inference_calls,
            "num_failed_calls": self.num_failed_calls,
            "success_rate": self.get_success_rate(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "current_trial": self.current_trial,
            "best_score": self.best_score,
            "metadata": self.metadata,
        }


class ProgressTracker:
    """Progress tracker for optimization with tqdm integration."""

    def __init__(
        self,
        total_trials: int,
        description: str = "Optimizing",
        disable: bool = False,
    ):
        """Initialize progress tracker.

        Args:
            total_trials: Total number of trials to run.
            description: Description for progress bar.
            disable: Whether to disable progress bar (for non-verbose mode).
        """
        self.total_trials = total_trials
        self.stats = OptimizationStats()
        bar_fmt = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, "
            "{rate_fmt}] Best: {postfix}"
        )
        self.pbar = tqdm(
            total=total_trials,
            desc=description,
            unit="trial",
            disable=disable,
            bar_format=bar_fmt,
        )

    def update(
        self,
        n: int = 1,
        score: Optional[float] = None,
        examples_processed: int = 0,
        inference_calls: int = 0,
        failed_calls: int = 0,
    ) -> None:
        """Update progress.

        Args:
            n: Number of trials to increment.
            score: Current score (updates best if higher).
            examples_processed: Number of examples processed in this update.
            inference_calls: Number of inference calls made in this update.
            failed_calls: Number of failed calls in this update.
        """
        self.stats.current_trial += n
        self.stats.num_examples_processed += examples_processed
        self.stats.num_inference_calls += inference_calls
        self.stats.num_failed_calls += failed_calls

        if score is not None and score > self.stats.best_score:
            self.stats.best_score = score

        # Update progress bar
        self.pbar.update(n)
        self.pbar.set_postfix_str(f"{self.stats.best_score:.4f}")

    def set_description(self, desc: str) -> None:
        """Update progress bar description.

        Args:
            desc: New description.
        """
        self.pbar.set_description(desc)

    def write(self, message: str) -> None:
        """Write a message without disrupting progress bar.

        Args:
            message: Message to write.
        """
        self.pbar.write(message)

    def close(self) -> None:
        """Close progress bar and finalize stats."""
        self.stats.end_time = time.time()
        self.pbar.close()

    def get_stats(self) -> OptimizationStats:
        """Get current optimization statistics.

        Returns:
            Current optimization stats.
        """
        return self.stats

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
