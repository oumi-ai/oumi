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
import collections
import time
from asyncio.locks import BoundedSemaphore
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncioPoliteSemaphore(BoundedSemaphore):
    def __init__(self, capacity: int, politeness_policy: float):
        """A semaphore that enforces a politeness policy.

        Args:
            capacity: The maximum number of concurrent tasks.
            politeness_policy: The politeness policy in seconds.
        """
        super().__init__(capacity)
        self._capacity = capacity
        self._politeness_policy = politeness_policy
        self._queue: collections.deque[float] = collections.deque(maxlen=capacity)
        for _ in range(capacity):
            self._queue.append(-1)

    def _get_wait_time(self) -> float:
        """Calculates the time to wait after acquiring the semaphore.

        Returns a negative number if no wait is needed.

        Returns:
            The time to wait to acquire the semaphore.
        """
        next_start_time = self._queue.popleft()
        return next_start_time - time.time()

    async def acquire(self):
        """Acquires the semaphore and waits for the politeness policy to be respected.

        If the queue is empty, no wait is needed.
        """
        await super().acquire()
        print(f"queue: {self._queue}")
        wait_time = self._get_wait_time()
        print(f"wait_time: {wait_time}")
        if wait_time > 0:
            await asyncio.sleep(wait_time)

    async def release(self):
        """Releases the semaphore.

        Adds the current time to the queue. So the next task will wait for the
        politeness policy to be respected.
        """
        if len(self._queue) == self._queue.maxlen:
            raise ValueError(
                f"Released too many times. Capacity {self._capacity} reached."
            )
        self._queue.append(time.time() + self._politeness_policy)
        super().release()


def safe_asyncio_run(main: Coroutine[Any, Any, T]) -> T:
    """Run an Awaitable in a new thread. Blocks until the thread is finished.

    This circumvents the issue of running async functions in the main thread when
    an event loop is already running (Jupyter notebooks, for example).

    Prefer using `safe_asyncio_run` over `asyncio.run` to allow upstream callers to
    ignore our dependency on asyncio.

    Args:
        main: The Coroutine to resolve.

    Returns:
        The result of the Coroutine.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        task = executor.submit(asyncio.run, main)
        return task.result()
