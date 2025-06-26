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

import math
import threading
import time
from collections import deque

from oumi.core.configs.params.remote_params import AdaptiveThroughputParams
from oumi.inference.adaptive_semaphore import AdaptiveSemaphore


class AdaptiveThroughputController:
    """Controls adaptive throughput based on error rates."""

    def __init__(self, config: AdaptiveThroughputParams):
        """Initialize the adaptive throughput controller.

        Args:
            config: Configuration for adaptive throughput control.
        """
        self._config = config
        self._current_concurrency = config.min_concurrency
        self._semaphore = AdaptiveSemaphore(self._current_concurrency)

        # Track request outcomes in a sliding window
        self._outcomes = deque()
        self._outcome_lock = threading.Lock()

        # State tracking
        self._last_adjustment_time = 0
        self._last_warmup_time = 0
        self._in_backoff = False
        self._consecutive_good_windows = 0
        self._consecutive_error_windows = 0
        self._good_windows_required_for_recovery = 2
        self._error_windows_for_additional_backoff = 2

    def record_success(self):
        """Record a successful request."""
        with self._outcome_lock:
            self._outcomes.append(True)

    def record_error(self):
        """Record a failed request."""
        with self._outcome_lock:
            self._outcomes.append(False)

    async def acquire(self):
        """Acquire a permit, potentially adjusting concurrency first."""
        await self._maybe_adjust_concurrency()
        await self._semaphore.acquire()

    def release(self):
        """Release a permit."""
        self._semaphore.release()

    def _get_error_rate(self) -> float:
        """Calculate current error rate based on recent outcomes."""
        with self._outcome_lock:
            if not self._outcomes:
                return 0.0  # No data yet, so no error rate.
            errors = sum(1 for outcome in self._outcomes if not outcome)
            return errors / len(self._outcomes)

    async def _maybe_adjust_concurrency(self) -> None:
        """Adjust concurrency based on current error rate and state."""
        # No data yet, so no information to adjust concurrency.
        if not self._outcomes or len(self._outcomes) < self._config.min_window_size:
            return

        error_rate = self._get_error_rate()
        time_since_last_adjustment = time.time() - self._last_adjustment_time

        # If we haven't reached the update interval, do nothing
        if time_since_last_adjustment < self._config.update_interval:
            return

        # Check if we need to backoff due to high error rate
        if error_rate >= self._config.error_threshold and not self._in_backoff:
            await self._backoff()
            return

        # Check if we can recover from backoff
        if self._in_backoff:
            if error_rate <= self._config.recovery_threshold:
                self._consecutive_good_windows += 1
                self._consecutive_error_windows = 0
                # Require multiple good windows before recovering
                if (
                    self._consecutive_good_windows
                    >= self._good_windows_required_for_recovery
                ):
                    self._end_backoff()
                else:
                    return
            else:
                self._consecutive_error_windows += 1
                self._consecutive_good_windows = 0
                if (
                    self._consecutive_error_windows
                    >= self._error_windows_for_additional_backoff
                ):
                    await self._backoff()
                    return

        # Warmup: gradually increase concurrency if error rate is low
        if (
            error_rate <= self._config.recovery_threshold
            and self._current_concurrency < self._config.max_concurrency
        ):
            await self._warmup()

    def _end_backoff(self):
        self._in_backoff = False
        self._consecutive_good_windows = 0
        self._consecutive_error_windows = 0

    async def _backoff(self):
        """Reduce concurrency due to high error rate."""
        new_concurrency = max(
            self._config.min_concurrency,
            # Round down to nearest integer
            math.floor(self._current_concurrency * self._config.backoff_factor),
        )

        if new_concurrency != self._current_concurrency:
            await self._update_concurrency(new_concurrency)

        # Set backoff state
        self._in_backoff = True

    def _reset_outcomes(self):
        with self._outcome_lock:
            self._outcomes.clear()
            self._consecutive_good_windows = 0
            self._consecutive_error_windows = 0
            self._last_adjustment_time = time.time()

    async def _warmup(self):
        """Increase concurrency during warmup."""
        new_concurrency = min(
            self._config.max_concurrency,
            self._current_concurrency + self._config.concurrency_step,
        )

        if new_concurrency != self._current_concurrency:
            await self._update_concurrency(new_concurrency)

    async def _update_concurrency(self, new_concurrency: int):
        """Update the concurrency limit, preserving existing waiters."""
        self._current_concurrency = new_concurrency
        await self._semaphore.adjust_capacity(new_concurrency)
        self._reset_outcomes()
