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


class AdaptiveSemaphore:
    """A semaphore that can dynamically adjust capacity while preserving waiters."""

    def __init__(self, initial_capacity: int):
        """Initialize the adaptive semaphore.

        Args:
            initial_capacity: The initial capacity of the semaphore.
        """
        self._capacity = initial_capacity
        self._current_count = initial_capacity
        self._waiters: deque = deque()
        self._condition = asyncio.Condition()

    async def acquire(self):
        """Acquire a permit."""
        waiter = None
        async with self._condition:
            if self._current_count <= 0:
                waiter = asyncio.get_event_loop().create_future()
                self._waiters.append(waiter)
            else:
                self._current_count -= 1

        if waiter is None:
            return

        try:
            await waiter
        except asyncio.CancelledError:
            # Remove cancelled waiter
            async with self._condition:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                raise

    def release(self):
        """Release a permit."""

        async def _release():
            async with self._condition:
                self._current_count += 1

            # Wake up the next waiter if any
            while self._waiters and self._current_count > 0:
                waiter = self._waiters.popleft()
                if not waiter.cancelled():
                    self._current_count -= 1
                    waiter.set_result(None)
                    break

        # Schedule the release to run in the event loop
        asyncio.create_task(_release())

    async def adjust_capacity(self, new_capacity: int):
        """Adjust the semaphore capacity, handling waiters appropriately."""
        async with self._condition:
            capacity_change = new_capacity - self._capacity
            self._capacity = new_capacity
            self._current_count += capacity_change

            # If we increased capacity, wake up waiters
            if capacity_change > 0:
                woken = 0
                while (
                    self._waiters
                    and self._current_count > 0
                    and woken < capacity_change
                ):
                    waiter = self._waiters.popleft()
                    if not waiter.cancelled():
                        self._current_count -= 1
                        waiter.set_result(None)
                        woken += 1

            # If we decreased capacity below current usage, we don't forcibly
            # revoke permits, but future acquires will be limited by the new capacity
