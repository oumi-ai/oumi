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
import logging
import time
from collections import deque

from oumi.core.async_utils import safe_asyncio_run

logger = logging.getLogger(__name__)


class AdaptiveSemaphore:
    """A semaphore that can dynamically adjust capacity.

    Preserves waiters during capacity adjustments.
    """

    def __init__(self, initial_capacity: int):
        """Initialize the adaptive semaphore.

        Args:
            initial_capacity: The initial capacity of the semaphore.
        """
        if initial_capacity <= 0:
            raise ValueError("Initial capacity must be greater than 0.")

        self._max_capacity = initial_capacity
        self._current_capacity = initial_capacity
        self._waiters: deque = deque()
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Enter the context manager."""
        await self.acquire()
        return None

    async def __aexit__(self, exc_type, exc, tb):
        """Exit the context manager."""
        await self._release_async()

    def __repr__(self):
        """Return a string representation of the semaphore."""
        return (
            f"AdaptiveSemaphore(capacity={self._max_capacity}, "
            f"current_count={self._current_capacity})"
        )

    def locked(self):
        """Return True if the semaphore is locked."""
        return self._current_capacity <= 0

    async def acquire(self) -> bool:
        """Acquire a permit."""
        waiter_future = None
        async with self._lock:
            if self.locked():
                waiter_future = asyncio.get_event_loop().create_future()
                self._waiters.append(waiter_future)
            else:
                self._current_capacity -= 1
        if waiter_future is None:
            return True

        try:
            await waiter_future
        except asyncio.CancelledError:
            # Remove cancelled waiter
            await self._remove_waiter_from_queue(waiter_future)
            raise
        finally:
            await self._remove_waiter_from_queue(waiter_future)

        logger.info(
            "ADAPTIVE SEMAPHORE: Acquired permit (current capacity: %d)",
            self._current_capacity,
        )
        return True

    async def _remove_waiter_from_queue(self, waiter: asyncio.Future):
        """Remove a waiter from the queue."""
        async with self._lock:
            try:
                self._waiters.remove(waiter)
            except ValueError:
                pass

    async def _release_async(self):
        """Release a permit."""
        async with self._lock:
            self._current_capacity = min(self._current_capacity + 1, self._max_capacity)

            # Wake up the next waiter if any
            while self._waiters and self._current_capacity > 0:
                waiter = self._waiters.popleft()
                if not waiter.cancelled():
                    self._current_capacity -= 1
                    waiter.set_result(None)
                    break

            logger.info(
                "ADAPTIVE SEMAPHORE: Released permit (current capacity: %d)",
                self._current_capacity,
            )

    def release(self):
        """Release a permit."""
        safe_asyncio_run(self._release_async())

    async def adjust_capacity(self, new_capacity: int):
        """Adjust the semaphore capacity, handling waiters appropriately."""
        if new_capacity < 0:
            raise ValueError("New capacity must be non-negative.")

        async with self._lock:
            capacity_change = new_capacity - self._max_capacity
            self._max_capacity = new_capacity
            self._current_capacity += capacity_change
            # Allow negative capacity to track excess active workers
            # during backoff

            # If we increased capacity, wake up waiters
            if capacity_change > 0:
                woken = 0
                while (
                    self._waiters
                    and self._current_capacity > 0
                    and woken < capacity_change
                ):
                    waiter = self._waiters.popleft()
                    if not waiter.cancelled():
                        self._current_capacity -= 1
                        waiter.set_result(None)
                        woken += 1
                logger.info(
                    "Adjusted capacity: %d → %d (woke %d waiters)",
                    new_capacity - capacity_change,
                    new_capacity,
                    woken,
                )

            # If we decreased capacity below current usage, we don't
            # forcibly revoke permits, but future acquires will be
            # limited by the new capacity


class PoliteAdaptiveSemaphore(AdaptiveSemaphore):
    """A semaphore that enforces rate limiting.

    Based on capacity and politeness.

    This semaphore grants permits at a controlled rate:
    capacity / politeness_policy permits per second. For example, with
    capacity=60 and politeness_policy=60s, the semaphore grants 1 permit
    per second.

    The base AdaptiveSemaphore handles capacity limiting
    (max N concurrent workers), while this class adds rate limiting on top
    (max R grants per second).
    """

    def __init__(self, capacity: int, politeness_policy: float):
        """Initialize the rate-limited adaptive semaphore.

        Args:
            capacity: The maximum number of concurrent tasks.
            politeness_policy: The politeness policy in seconds
                (time window for rate limiting).
        """
        self._politeness_policy = politeness_policy
        super().__init__(initial_capacity=capacity)

        # Track active workers separately from capacity
        # This helps during backoff to know when workers have drained
        self._active_workers = 0
        self._active_lock = asyncio.Lock()

        # Lock to prevent concurrent capacity adjustments
        self._adjustment_lock = asyncio.Lock()

        # Rate limiting state
        self._last_grant_time = 0.0  # Last time we granted a permit
        self._grant_lock = asyncio.Lock()  # Serialize grant timing

        # Event to signal when capacity becomes non-zero
        self._capacity_available = asyncio.Event()
        if capacity > 0:
            self._capacity_available.set()

    def _calculate_min_interval(self) -> float:
        """Calculate minimum time between grants based on current capacity.

        Returns:
            Minimum seconds between permits (politeness_policy / capacity).
        """
        if self._max_capacity == 0:
            return float("inf")  # No permits granted when capacity is 0
        return self._politeness_policy / self._max_capacity

    async def acquire(self):
        """Acquire a permit with rate limiting.

        First acquires from the base semaphore (capacity limiting),
        then enforces rate limiting by ensuring minimum interval between
        grants.
        """
        # First acquire from base semaphore (capacity limiting)
        await super().acquire()

        async with self._grant_lock:
            # Wait for capacity to become non-zero
            await self._capacity_available.wait()

            async with self._adjustment_lock:
                min_interval = self._calculate_min_interval()
            time_since_last = time.time() - self._last_grant_time
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
            self._last_grant_time = time.time()

        # Increment active worker count
        async with self._active_lock:
            self._active_workers += 1

    async def pause_for_cooldown(self, cooldown_seconds: float):
        """Pause for cooldown."""
        logger.info(
            "POLITE ADAPTIVE SEMAPHORE: Pausing for cooldown (%.1fs)",
            cooldown_seconds,
        )
        await asyncio.sleep(cooldown_seconds)

    async def _release_async(self):
        """Release a permit."""
        # Decrement active worker count
        async with self._active_lock:
            self._active_workers = max(0, self._active_workers - 1)
            logger.info(
                "Releasing permit: active_workers=%d, max_capacity=%d, "
                "current_capacity=%d",
                self._active_workers,
                self._max_capacity,
                self._current_capacity,
            )

        await super()._release_async()

        # Log after release
        logger.info(
            "Released permit: max_capacity=%d, current_capacity=%d",
            self._max_capacity,
            self._current_capacity,
        )

    async def adjust_capacity(self, new_capacity: int):
        """Adjust the semaphore capacity.

        With the rate-based approach, capacity adjustments are simple:
        just update the capacity and the rate calculation automatically
        adjusts.
        """
        if new_capacity < 0:
            raise ValueError("New capacity must be non-negative.")

        # Acquire adjustment lock to prevent concurrent capacity adjustments
        async with self._adjustment_lock:
            old_capacity = self._max_capacity
            old_rate = self._calculate_min_interval()

            # Just update the base capacity
            await super().adjust_capacity(new_capacity)

            # Update the capacity available event
            if new_capacity > 0:
                self._capacity_available.set()
            else:
                self._capacity_available.clear()

            new_rate = self._calculate_min_interval()
            logger.info(
                "Rate-limited capacity adjustment: %d → %d "
                "(rate: %.3fs → %.3fs per permit, target QPM: %.1f → %.1f)",
                old_capacity,
                new_capacity,
                old_rate,
                new_rate,
                (60.0 / old_rate) if old_rate > 0 else 0,
                (60.0 / new_rate) if new_rate > 0 else 0,
            )

    async def adjust_politeness_policy(self, new_politeness: float):
        """Adjust the politeness policy.

        Args:
            new_politeness: The new politeness policy in seconds.
        """
        if new_politeness <= 0:
            raise ValueError("Politeness policy must be greater than 0.")

        async with self._adjustment_lock:
            old_politeness = self._politeness_policy
            old_rate = self._calculate_min_interval()

            self._politeness_policy = new_politeness

            new_rate = self._calculate_min_interval()
        logger.info(
            "Politeness adjustment: %.1fs → %.1fs "
            "(rate: %.3fs → %.3fs per permit, target QPM: %.1f → %.1f)",
            old_politeness,
            new_politeness,
            old_rate,
            new_rate,
            (60.0 / old_rate) if old_rate > 0 else 0,
            (60.0 / new_rate) if new_rate > 0 else 0,
        )
