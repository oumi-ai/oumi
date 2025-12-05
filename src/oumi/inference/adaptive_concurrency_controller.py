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
from oumi.inference.adaptive_semaphore import PoliteAdaptiveSemaphore
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

    def __init__(self, config: AdaptiveConcurrencyParams, politeness_policy: float):
        """Initialize the adaptive concurrency controller.

        Args:
            config: Configuration for adaptive concurrency control.
            politeness_policy: The amount of time workers will wait until sending their
                next request, in seconds.
        """
        self._config = config
        self._current_concurrency = self._get_initial_concurrency()
        self._initial_concurrency = self._current_concurrency
        self._initial_politeness_policy = politeness_policy
        self._semaphore = PoliteAdaptiveSemaphore(
            self._current_concurrency, politeness_policy
        )

        # Track dynamic min_update_time with a floor of 1.0 second
        self._min_update_time_floor = 1.0
        self._current_min_update_time = config.min_update_time

        # Track request outcomes in a sliding window for error rate calculation
        self._outcomes: deque[bool] = deque(maxlen=_MAX_OUTCOMES_WINDOW_SIZE)
        self._outcome_lock = asyncio.Lock()

        # Rate limit window tracking (60s windows)
        self._rate_limit_window_start = time.time()
        self._early_window_errors = 0  # Errors in first 45s of window
        self._last_window_reset_time = time.time()

        # RPM tracking for intelligent backoff
        # Track successful requests with timestamps for last 60 seconds
        self._successful_requests: deque[float] = deque()
        # History of successful RPM values at each adjustment point
        # Each entry is (rpm, concurrency, politeness_policy)
        self._rpm_history: list[tuple[float, int, float]] = []

        # Best known operating point for "focusing in" (concurrency + politeness)
        # Track both since they together determine throughput
        self._best_concurrency = self._current_concurrency
        self._best_politeness = 15.0  # Will be updated from semaphore
        self._overshoot_count = 0  # Number of times we've hit errors and backed off

        # Stack of known good concurrency levels for smarter backoff
        # When we backoff, pop from this stack to return to last known good state
        self._good_concurrency_stack: list[int] = []

        # Slow-start exploration: when exploring past best, use +1, +2, +4, +8...
        # Reset to 1 after each backoff for conservative restart
        self._cautious_step_size = 1

        # Burst detection: track recent errors for immediate backoff
        # When rate limits hit, we get bursts of errors (10-15 within milliseconds)
        # Detect and react immediately rather than waiting for adjustment cycle
        self._recent_errors: deque[float] = deque()  # Timestamps of recent errors
        self._burst_window_seconds = 2.0  # Window for detecting error bursts
        self._burst_threshold = 7  # Number of errors in window to trigger backoff
        self._last_burst_backoff_time = 0.0  # Rate limit burst backoffs to avoid spam
        self._burst_backoff_in_progress = False  # Prevent concurrent burst backoffs

        # Dynamic scaling factors that reduce with each overshoot
        self._current_warmup_factor = config.exponential_scaling_factor
        self._current_backoff_factor = config.backoff_factor

        # State tracking
        self._last_adjustment_time = 0  # Set to current time on first permit
        self._last_warmup_time = 0
        self._in_backoff = False
        self._first_permit_granted = False
        self._consecutive_good_windows_since_last_update = 0
        self._consecutive_error_windows_since_last_update = 0
        self._consecutive_good_windows_required_for_recovery = 2
        self._consecutive_error_windows_for_additional_backoff = 2
        self._adjustment_in_progress = False  # Prevent concurrent adjustments

        self._logged_backoff_warning = False
        self._logged_warmup_warning = False

        self._cooldown_lock = asyncio.Lock()

    async def record_success(self):
        """Record a successful outcome."""
        async with self._outcome_lock:
            now = time.time()
            self._outcomes.append(True)

            # Track successful request timestamp for RPM calculation
            self._successful_requests.append(now)
            # Clean old successes outside 60s window
            try:
                now_float = float(now)
                while (
                    self._successful_requests
                    and float(self._successful_requests[0]) < now_float - 60.0
                ):
                    self._successful_requests.popleft()
            except (TypeError, ValueError):
                # Handle mocked time in tests
                pass

            # Check if we've entered a new window
            self._check_and_reset_window(now)

    async def record_error(self):
        """Record a failed outcome."""
        async with self._outcome_lock:
            now = time.time()
            self._outcomes.append(False)

            # Track window position of errors
            self._check_and_reset_window(now)
            position = self._get_window_position(now)

            # Errors in first 36s (60%) of window are strong signals we're over limit
            if position < 0.60:
                self._early_window_errors += 1

            # Burst detection: track recent errors for immediate backoff
            self._recent_errors.append(now)
            # Clean old errors outside burst window
            try:
                now_float = float(now)
                burst_threshold = now_float - self._burst_window_seconds
                while (
                    self._recent_errors
                    and float(self._recent_errors[0]) < burst_threshold
                ):
                    self._recent_errors.popleft()
            except (TypeError, ValueError):
                # Handle mocked time in tests
                pass

            # If we detect a burst, trigger immediate backoff
            if len(self._recent_errors) >= self._burst_threshold:
                # Rate limit: don't backoff more than once per 10 seconds
                # Also check if burst backoff is in progress to prevent cascades
                try:
                    now_float = float(now)
                    last_backoff_float = float(self._last_burst_backoff_time)
                    should_backoff = (
                        now_float - last_backoff_float > 10.0
                        and not self._burst_backoff_in_progress
                    )
                except (TypeError, ValueError):
                    # Handle mocked time in tests - skip burst backoff
                    should_backoff = False

                if should_backoff:
                    logger.warning(
                        "🔥 ERROR BURST DETECTED: %d errors in %.1fs - "
                        "triggering immediate backoff!",
                        len(self._recent_errors),
                        self._burst_window_seconds,
                    )
                    self._last_burst_backoff_time = now
                    self._burst_backoff_in_progress = True
                    # Trigger immediate backoff (outside the lock to avoid deadlock)
                    # We'll schedule it to run after releasing the lock
                    asyncio.create_task(self._immediate_burst_backoff())

    def _check_and_reset_window(self, now: float):
        """Reset window tracking if we've entered a new 60s window."""
        try:
            now_float = float(now)
            start_float = float(self._rate_limit_window_start)
        except (TypeError, ValueError):
            # Handle mocked time in tests
            return

        elapsed_since_start = now_float - start_float

        # If more than 60s has passed, we're in a new window
        if elapsed_since_start >= 60.0:
            # Reset for new window
            self._rate_limit_window_start = now_float - (elapsed_since_start % 60.0)
            self._early_window_errors = 0
            self._last_window_reset_time = now_float

    def _get_window_position(self, now: float) -> float:
        """Get position in current 60s rate limit window (0.0 to 1.0).

        Returns:
            Position in window: 0.0 = start of window, 1.0 = end of window
        """
        try:
            now_float = float(now)
            start_float = float(self._rate_limit_window_start)
        except (TypeError, ValueError):
            # Handle mocked time in tests - return middle of window
            return 0.5

        elapsed = now_float - start_float
        position = (elapsed % 60.0) / 60.0
        return position

    def _has_early_window_errors(self) -> bool:
        """Check if we have errors early in the current window.

        Returns:
            True if we have errors in the first 75% of the current window.
        """
        return self._early_window_errors > 0

    async def acquire(self):
        """Acquire a permit, then try to adjust concurrency before returning."""
        await self._semaphore.acquire()

        # Initialize adjustment timer on first permit to ignore startup delay
        if not self._first_permit_granted:
            self._first_permit_granted = True
            self._last_adjustment_time = time.time()

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

    def _get_current_rpm(self) -> float:
        """Calculate current requests per minute based on recent successes.

        Returns:
            Current successful requests per minute over the last 60 seconds.
        """
        now = time.time()
        # Clean old successes
        try:
            now_float = float(now)
            while (
                self._successful_requests
                and float(self._successful_requests[0]) < now_float - 60.0
            ):
                self._successful_requests.popleft()
        except (TypeError, ValueError):
            # Handle mocked time in tests
            pass

        # Return count of successful requests in last 60 seconds
        return float(len(self._successful_requests))

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

        # Check error rate first for early backoff detection
        error_rate = await self._get_error_rate()

        # If error rate is high, allow immediate backoff even before min_update_time
        # This makes the system reactive to problems
        if error_rate >= self._config.error_threshold and not self._in_backoff:
            async with self._outcome_lock:
                total_outcomes = len(self._outcomes)
                error_count = sum(1 for outcome in self._outcomes if not outcome)
            # Skip periodic backoff if burst backoff is in progress (drain/cooldown)
            if self._burst_backoff_in_progress:
                logger.debug(
                    "Skipping periodic backoff (error_rate=%.1f%%) - "
                    "burst backoff in progress",
                    error_rate * 100,
                )
                return

            logger.warning(
                "Periodic backoff triggered: error_rate=%.1f%% (%d/%d), "
                "threshold=%.1f%%",
                error_rate * 100,
                error_count,
                total_outcomes,
                self._config.error_threshold * 100,
            )
            await self._try_backoff()
            # Apply cooldown to allow server-side token bucket recovery
            await self._apply_cooldown()
            return

        # For recovery and warmup, enforce min_update_time to prevent thrashing
        time_since_last_adjustment = time.time() - self._last_adjustment_time
        if time_since_last_adjustment < self._current_min_update_time:
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
                    # Skip additional backoff if burst backoff is in progress
                    if not self._burst_backoff_in_progress:
                        await self._try_backoff()
                        # Apply cooldown to allow server-side token bucket recovery
                        await self._apply_cooldown()
                    else:
                        logger.debug(
                            "Skipping additional backoff - burst backoff in progress"
                        )
                return

        await self._try_warmup()

    def _end_backoff(self):
        """End backoff state."""
        self._in_backoff = False
        self._consecutive_good_windows_since_last_update = 0
        self._consecutive_error_windows_since_last_update = 0

    async def _apply_cooldown(self):
        """Apply a cooldown period to allow server-side token bucket recovery.

        After backoff, temporarily set capacity to 0, sleep briefly
        to give the server's rate limiting token bucket time to refill,
        then restore the backoff capacity.
        """
        cooldown_seconds = 3.0  # Reduced from 5s for faster recovery

        # Restore capacity to the backoff level
        current_capacity = self._current_concurrency
        await self._semaphore.adjust_capacity(0)
        async with self._cooldown_lock:
            await self._semaphore.pause_for_cooldown(cooldown_seconds)
        await self._semaphore.adjust_capacity(current_capacity)

    async def _immediate_burst_backoff(self):
        """Handle immediate backoff when error burst is detected.

        This is triggered when we detect multiple errors in a very short window
        (e.g., 5+ errors in 2 seconds), indicating we've hit a rate limit.
        Rather than waiting for the regular adjustment cycle, we backoff immediately.
        """
        try:
            logger.info("Executing immediate burst backoff...")
            await self._try_backoff()
        finally:
            # Always clear the flag, even if backoff fails
            self._burst_backoff_in_progress = False

    async def _try_backoff(self):
        """Try to reduce concurrency due to high error rate."""
        # Find highest successful RPM from history
        highest_rpm = 0.0
        best_rpm_config = None
        if self._rpm_history:
            highest_rpm = max(rpm for rpm, _, _ in self._rpm_history)
            # Get the config (concurrency, politeness) for that RPM
            for rpm, concurrency, politeness in self._rpm_history:
                if rpm == highest_rpm:
                    best_rpm_config = (concurrency, politeness)
                    break

        # If we have history, best RPM
        # Only use RPM-based backoff if politeness is meaningful (> 0.1s)
        # Otherwise the formula breaks down (division by zero or near-zero)
        if (
            best_rpm_config is not None
            and highest_rpm > 0
            and best_rpm_config[1] > 0.1  # best_politeness > 0.1
        ):
            # Target 95% of the highest successful RPM to stay safely below limit
            target_rpm = highest_rpm * 0.95
            best_concurrency, best_politeness = best_rpm_config

            # Calculate new concurrency to achieve target RPM
            # Formula: target_rpm = (concurrency * 60) / politeness_policy
            # Solving for concurrency: concurrency = (target_rpm * politeness) / 60
            new_concurrency = math.floor((target_rpm * best_politeness) / 60.0)
            # Ensure we stay within bounds
            new_concurrency = max(
                self._config.min_concurrency,
                min(new_concurrency, self._config.max_concurrency),
            )

            logger.info(
                "RPM-based backoff: targeting %.1f RPM (95%% of best "
                "%.1f). Adjusting concurrency: %d→%d (politeness "
                "unchanged at %.1fs)",
                target_rpm,
                highest_rpm,
                self._current_concurrency,
                new_concurrency,
                best_politeness,
            )

            # Apply the new configuration
            await self._update_concurrency(new_concurrency)

            # Update tracking variables
            self._overshoot_count += 1
            self._current_warmup_factor = max(
                1.1,
                self._config.exponential_scaling_factor * (0.9**self._overshoot_count),
            )
            self._current_backoff_factor = max(
                0.5,
                self._config.backoff_factor * (0.9**self._overshoot_count),
            )
            self._cautious_step_size = 1
            self._in_backoff = True
            return

        # Fallback to original backoff logic if no RPM history
        # Use current overshoot count to apply backoff, then increment for next
        # time. This way first backoff uses original factors, subsequent ones
        # use reduced factors
        decay_factor = 0.9  # 10% reduction per overshoot

        # Calculate what factors WILL BE after this backoff (for logging)
        next_overshoot_count = self._overshoot_count + 1
        next_warmup_factor = max(
            1.1,
            self._config.exponential_scaling_factor
            * (decay_factor**next_overshoot_count),
        )
        next_backoff_factor = max(
            0.5,
            self._config.backoff_factor * (decay_factor**next_overshoot_count),
        )

        # Apply current backoff (using current overshoot count)
        self._current_backoff_factor = max(
            0.5,
            self._config.backoff_factor * (decay_factor**self._overshoot_count),
        )

        # NOW increment and update for next time
        self._overshoot_count += 1
        self._current_warmup_factor = next_warmup_factor

        # Reset slow-start step size to 1 for conservative restart
        self._cautious_step_size = 1

        # Determine target concurrency using the good concurrency stack
        # Pop from stack to get last known good value
        if self._good_concurrency_stack:
            # Pop the top value from stack (last known good that's below current)
            # Keep popping until we find one that's actually below current
            backoff_target = None
            while self._good_concurrency_stack:
                candidate = self._good_concurrency_stack.pop()
                if candidate < self._current_concurrency:
                    backoff_target = candidate
                    break
                # If candidate >= current, discard it and try next

            if backoff_target is not None:
                new_concurrency = max(self._config.min_concurrency, backoff_target)
                logger.info(
                    "Backoff #%d: %d → %d (from stack, depth now: %d). "
                    "Factors: warmup %.2fx→%.2fx, backoff %.2fx→%.2fx",
                    self._overshoot_count,
                    self._current_concurrency,
                    new_concurrency,
                    len(self._good_concurrency_stack),
                    self._config.exponential_scaling_factor
                    * (decay_factor ** (self._overshoot_count - 1)),
                    next_warmup_factor,
                    self._config.backoff_factor
                    * (decay_factor ** (self._overshoot_count - 1)),
                    next_backoff_factor,
                )
            else:
                # Stack empty or all values >= current, fall back to best
                new_concurrency = max(
                    self._config.min_concurrency, self._best_concurrency
                )
                logger.info(
                    "Backoff #%d: %d → %d (to best=%d, stack exhausted). "
                    "Factors: warmup %.2fx→%.2fx, backoff %.2fx→%.2fx",
                    self._overshoot_count,
                    self._current_concurrency,
                    new_concurrency,
                    self._best_concurrency,
                    self._config.exponential_scaling_factor
                    * (decay_factor ** (self._overshoot_count - 1)),
                    next_warmup_factor,
                    self._config.backoff_factor
                    * (decay_factor ** (self._overshoot_count - 1)),
                    next_backoff_factor,
                )
        else:
            # No stack available, fall back to multiplicative backoff
            new_concurrency = max(
                self._config.min_concurrency,
                math.floor(self._current_concurrency * self._current_backoff_factor),
            )
            logger.info(
                "Backoff #%d: %d → %d (%.2fx, no stack). "
                "Factors: warmup %.2fx→%.2fx, backoff %.2fx→%.2fx",
                self._overshoot_count,
                self._current_concurrency,
                new_concurrency,
                self._current_backoff_factor,
                self._config.exponential_scaling_factor
                * (decay_factor ** (self._overshoot_count - 1)),
                next_warmup_factor,
                self._config.backoff_factor
                * (decay_factor ** (self._overshoot_count - 1)),
                next_backoff_factor,
            )

        if new_concurrency != self._current_concurrency:
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
        # Skip if adjustment is in progress to prevent concurrent decisions
        if self._adjustment_in_progress:
            logger.info("Skipping warmup: adjustment already in progress")
            return

        # Update best operating point if currently operating successfully
        # (we reached this warmup attempt with low error rate)
        if self._current_concurrency > self._best_concurrency:
            self._best_concurrency = self._current_concurrency
            try:
                self._best_politeness = float(self._semaphore._politeness_policy)
            except (TypeError, AttributeError):
                pass  # Keep existing politeness if semaphore is mocked

            # Push current concurrency to good stack
            # Avoid duplicates - only push if different from last
            if (
                not self._good_concurrency_stack
                or self._good_concurrency_stack[-1] != self._current_concurrency
            ):
                self._good_concurrency_stack.append(self._current_concurrency)

            logger.info(
                "New best operating point: concurrency=%d, politeness=%.1fs "
                "(stack depth: %d)",
                self._best_concurrency,
                self._best_politeness,
                len(self._good_concurrency_stack),
            )

        # Check rate limit window position - early errors are strong signal to stop
        # Only use this for linear scaling mode (60s politeness)
        # For exponential scaling (15s politeness), bursts happen every 15s
        # so this check doesn't apply - rely on regular error rate threshold
        if not self._config.use_exponential_scaling and self._has_early_window_errors():
            now = time.time()
            position = self._get_window_position(now)
            logger.info(
                "Skipping warmup: detected %d errors early in rate limit "
                "window (position: %.1f%%, concurrency=%d). Over limit.",
                self._early_window_errors,
                position * 100,
                self._current_concurrency,
            )
            await self._clear_adjustment_state()
            return

        # Determine scaling step size based on position relative to best point
        using_slow_start = False  # Track if using slow-start exploration
        if self._config.use_exponential_scaling:
            # Use slow-start (+1, +2, +4, +8...) if we've had ANY overshoots
            # This applies to all warmups after first backoff, not just above best
            using_slow_start = self._overshoot_count > 0

            if using_slow_start:
                # Slow-start exploration: +1, +2, +4, +8...
                # Conservative exploration with exponential growth in step size
                new_concurrency = min(
                    self._config.max_concurrency,
                    self._current_concurrency + self._cautious_step_size,
                )
                logger.info(
                    "Warmup (slow-start +%d): %d → %d (best: %d)",
                    self._cautious_step_size,
                    self._current_concurrency,
                    new_concurrency,
                    self._best_concurrency,
                )
                # Double step size for next time (slow-start exponential growth)
                # This happens after the warmup succeeds
                self._cautious_step_size = min(
                    self._cautious_step_size * 2,
                    self._config.max_concurrency,  # Cap at max to avoid overflow
                )

                # Push current concurrency to good stack during slow-start
                # Since we're here, we've succeeded at this level
                if (
                    not self._good_concurrency_stack
                    or self._good_concurrency_stack[-1] != self._current_concurrency
                ):
                    self._good_concurrency_stack.append(self._current_concurrency)
                    logger.info(
                        "Added concurrency %d to good stack (stack depth: %d)",
                        self._current_concurrency,
                        len(self._good_concurrency_stack),
                    )
            else:
                # Use current (possibly reduced) warmup factor
                new_concurrency = min(
                    self._config.max_concurrency,
                    int(self._current_concurrency * self._current_warmup_factor),
                )
        else:
            # Linear scaling: add the fixed step size
            new_concurrency = min(
                self._config.max_concurrency,
                self._current_concurrency + self._config.concurrency_step,
            )

        if new_concurrency != self._current_concurrency:
            # Only log if not already logged by slow-start above
            if not using_slow_start:
                mode = "exp" if self._config.use_exponential_scaling else "linear"
                factor = (
                    self._current_warmup_factor
                    if self._config.use_exponential_scaling
                    else self._config.concurrency_step
                )
                logger.info(
                    "Warmup (%s): %d → %d (%.2fx, best: %d)",
                    mode,
                    self._current_concurrency,
                    new_concurrency,
                    factor,
                    self._best_concurrency,
                )
            # Set flag to prevent concurrent warmup decisions
            self._adjustment_in_progress = True
            try:
                await self._update_concurrency(new_concurrency)
            finally:
                # Always clear flag, even if adjustment fails
                self._adjustment_in_progress = False
        else:
            if not self._logged_warmup_warning:
                logger.warning(
                    "Entering warmup state, but concurrency is already at maximum "
                    "value. Consider raising the max concurrency."
                )
                self._logged_warmup_warning = True

    async def _update_concurrency(self, new_concurrency: int):
        """Update the concurrency limit, preserving existing waiters."""
        # Record current RPM before adjustment (only if we had successes)
        current_rpm = self._get_current_rpm()
        if current_rpm > 0:
            try:
                current_politeness = float(self._semaphore._politeness_policy)
            except (TypeError, AttributeError):
                current_politeness = self._initial_politeness_policy

            # Only record if this is a successful operating point (low error rate)
            error_rate = await self._get_error_rate()
            if error_rate <= self._config.recovery_threshold:
                self._rpm_history.append(
                    (current_rpm, self._current_concurrency, current_politeness)
                )
                logger.info(
                    "Recorded successful RPM: %.1f (concurrency=%d, "
                    "politeness=%.1fs, history depth=%d)",
                    current_rpm,
                    self._current_concurrency,
                    current_politeness,
                    len(self._rpm_history),
                )

        self._current_concurrency = new_concurrency
        await self._semaphore.adjust_capacity(new_concurrency)

        # Log current politeness for debugging (handle mocked semaphores)
        try:
            current_politeness = float(self._semaphore._politeness_policy)
            max_qpm = (
                (new_concurrency * 60.0 / current_politeness)
                if current_politeness > 0
                else 9999
            )
            logger.info(
                "Updated concurrency to %d "
                "(politeness: %.1fs, theoretical max QPM: %.1f)",
                new_concurrency,
                current_politeness,
                max_qpm,
            )
        except (TypeError, AttributeError):
            # Mocked semaphore in tests
            pass

        await self._reset_outcomes()
        await self._clear_adjustment_state()
