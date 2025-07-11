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

import asyncio
import math
import time
from collections import deque

from oumi.core.configs.params.remote_params import AdaptiveConcurrencyParams
from oumi.inference.adaptive_semaphore import AdaptiveSemaphore
from oumi.utils.logging import logger

_MAX_OUTCOMES_WINDOW_SIZE = 1000


class AdaptiveConcurrencyController:
    """Controls concurrency based on error rates.

    The controller functions as a semaphore, but can dynamically adjust the number of
    concurrent requests based on error rates by recording the outcomes of requests.

    The controller will not adjust the concurrency unless the number of recent requests
    is greater than or equal to the min_window_size, and the time since the last
    adjustment is greater than or equal to the min_update_time.

    To avoid making adjustments on stale data, the controller clears the outcomes
    queue whenever the concurrency is adjusted. In addition, the controller waits for
    consecutive good windows before recovering from backoff, and consecutive error
    windows before triggering additional backoff.

    To use the controller, use it as a semaphore while recording request outcomes.

    Example:
    ```python
    controller = AdaptiveConcurrencyController(config)
    async with controller.acquire():
        try:
            # Send async request
            response = await send_async_request()
            controller.record_success()
        except Exception as e:
            controller.record_error()
    ```
    """

    def __init__(self, config: AdaptiveConcurrencyParams):
        """Initialize the adaptive concurrency controller.

        Args:
            config: Configuration for adaptive concurrency control.
        """
        self._config = config
        self._current_concurrency = self._get_initial_concurrency()
        self._semaphore = AdaptiveSemaphore(self._current_concurrency)

        # Track request outcomes in a sliding window
        self._outcomes: deque[bool] = deque(maxlen=_MAX_OUTCOMES_WINDOW_SIZE)
        self._outcome_lock = asyncio.Lock()

        # State tracking
        self._last_adjustment_time = 0
        self._last_warmup_time = 0
        self._in_backoff = False
        self._consecutive_good_windows_since_last_update = 0
        self._consecutive_error_windows_since_last_update = 0
        self._consecutive_good_windows_required_for_recovery = 2
        self._consecutive_error_windows_for_additional_backoff = 2

        self._logged_backoff_warning = False
        self._logged_warmup_warning = False

    async def record_success(self):
        """Record a successful outcome."""
        async with self._outcome_lock:
            self._outcomes.append(True)

    async def record_error(self):
        """Record a failed outcome."""
        async with self._outcome_lock:
            self._outcomes.append(False)

    async def acquire(self):
        """Acquire a permit, then try to adjust concurrency before returning."""
        await self._semaphore.acquire()
        await self._try_adjust_concurrency()

    def release(self):
        """Release a permit."""
        self._semaphore.release()

    async def __aenter__(self):
        """Enter the context manager."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the context manager."""
        self.release()

    def _get_initial_concurrency(self) -> int:
        """Get the initial concurrency."""
        potential_concurrency = math.floor(
            (self._config.max_concurrency - self._config.min_concurrency)
            * self._config.initial_concurrency_factor
            + self._config.min_concurrency
        )
        # Ensure the initial concurrency is at least the min concurrency and no greater
        # than the max concurrency.
        return max(
            self._config.min_concurrency,
            min(potential_concurrency, self._config.max_concurrency),
        )

    async def _get_error_rate(self) -> float:
        """Calculate current error rate based on recent outcomes."""
        async with self._outcome_lock:
            if not self._outcomes:
                return 0.0  # No data yet, so no error rate.
            errors = sum(1 for outcome in self._outcomes if not outcome)
            return errors / len(self._outcomes)

    async def _try_adjust_concurrency(self) -> None:
        """Try to adjust concurrency based on current error rate and state."""
        # No data yet, so no information to adjust concurrency.
        if not self._outcomes or len(self._outcomes) < self._config.min_window_size:
            return

        # If we haven't reached the update time, do nothing
        time_since_last_adjustment = time.time() - self._last_adjustment_time
        if time_since_last_adjustment < self._config.min_update_time:
            return

        # Check if we need to backoff due to high error rate
        error_rate = await self._get_error_rate()
        if error_rate >= self._config.error_threshold and not self._in_backoff:
            await self._try_backoff()
            return

        # Check if we can recover from backoff
        if self._in_backoff:
            if error_rate <= self._config.recovery_threshold:
                self._consecutive_good_windows_since_last_update += 1
                self._consecutive_error_windows_since_last_update = 0
                await self._reset_outcomes()
                # Require multiple good windows before recovering
                if (
                    self._consecutive_good_windows_since_last_update
                    >= self._consecutive_good_windows_required_for_recovery
                ):
                    self._end_backoff()
                    # Don't return because we want to try to warmup
                else:
                    return
            else:
                self._consecutive_error_windows_since_last_update += 1
                self._consecutive_good_windows_since_last_update = 0
                await self._reset_outcomes()
                if (
                    self._consecutive_error_windows_since_last_update
                    >= self._consecutive_error_windows_for_additional_backoff
                ):
                    await self._try_backoff()
                return

        await self._try_warmup()

    def _end_backoff(self):
        """End backoff state."""
        self._in_backoff = False
        self._consecutive_good_windows_since_last_update = 0
        self._consecutive_error_windows_since_last_update = 0

    async def _try_backoff(self):
        """Try to reduce concurrency due to high error rate."""
        new_concurrency = max(
            self._config.min_concurrency,
            # Round down to nearest integer
            math.floor(self._current_concurrency * self._config.backoff_factor),
        )

        if new_concurrency != self._current_concurrency:
            logger.info(
                "Lowering concurrency from %d to %d due to high error rate.",
                self._current_concurrency,
                new_concurrency,
            )
            await self._update_concurrency(new_concurrency)
        else:
            if not self._logged_backoff_warning:
                logger.warning(
                    "Entering backoff state, but concurrency is already at minimum "
                    "value. Consider lowering the min concurrency or increasing the "
                    "time between requests (politeness_policy)."
                )
                self._logged_backoff_warning = True

        # Set backoff state
        self._in_backoff = True

    async def _reset_outcomes(self):
        """Reset the outcomes queue."""
        async with self._outcome_lock:
            self._outcomes.clear()

    async def _clear_adjustment_state(self):
        """Reset adjustment state variables."""
        self._consecutive_good_windows_since_last_update = 0
        self._consecutive_error_windows_since_last_update = 0
        self._last_adjustment_time = time.time()

    async def _try_warmup(self):
        """Try to increase concurrency during warmup."""
        new_concurrency = min(
            self._config.max_concurrency,
            self._current_concurrency + self._config.concurrency_step,
        )

        if new_concurrency != self._current_concurrency:
            await self._update_concurrency(new_concurrency)
        else:
            if not self._logged_warmup_warning:
                logger.warning(
                    "Entering warmup state, but concurrency is already at maximum "
                    "value. Consider raising the max concurrency."
                )
                self._logged_warmup_warning = True

    async def _update_concurrency(self, new_concurrency: int):
        """Update the concurrency limit, preserving existing waiters."""
        self._current_concurrency = new_concurrency
        await self._semaphore.adjust_capacity(new_concurrency)
        await self._reset_outcomes()
        await self._clear_adjustment_state()
