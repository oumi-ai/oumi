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
from collections import deque

from oumi.core.async_utils import safe_asyncio_run


class AdaptiveSemaphore:
    """A semaphore that can dynamically adjust capacity while preserving waiters."""

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
        self.release()

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

    def release(self):
        """Release a permit."""
        safe_asyncio_run(self._release_async())

    async def adjust_capacity(self, new_capacity: int):
        """Adjust the semaphore capacity, handling waiters appropriately."""
        if new_capacity <= 0:
            raise ValueError("New capacity must be greater than 0.")

        async with self._lock:
            capacity_change = new_capacity - self._max_capacity
            self._max_capacity = new_capacity
            self._current_capacity += capacity_change
            self._current_capacity = max(self._current_capacity, 0)

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

            # If we decreased capacity below current usage, we don't forcibly
            # revoke permits, but future acquires will be limited by the new capacity
