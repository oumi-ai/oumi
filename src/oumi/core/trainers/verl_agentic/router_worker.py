# Copyright 2026 - Oumi
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

"""Thread-affine execution of one rollout's synchronous tool environment."""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from functools import cache, partial
from itertools import cycle
from queue import Queue
from threading import Lock, Thread, current_thread
from typing import Any, TypeVar

from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.tool_call import ToolResult
from oumi.utils.logging import logger

_T = TypeVar("_T")


class _Lane:
    """FIFO worker that closes live routers on its own thread at process exit.

    ThreadPoolExecutor stops workers before ordinary atexit callbacks, so a GC
    finalizer cannot submit thread-affine teardown there during interpreter exit.
    """

    def __init__(self, name: str) -> None:
        self.routers: dict[int, ToolRouter] = {}
        self._queue: Queue[tuple[Future[Any], Callable[[], Any]] | None] = Queue()
        self._lock = Lock()
        self._stopped = False
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def submit(self, call: Callable[[], _T]) -> Future[_T]:
        """Enqueue work atomically with respect to shutdown."""
        future: Future[_T] = Future()
        with self._lock:
            if self._stopped:
                raise RuntimeError("Tool execution lane has shut down.")
            self._queue.put((future, call))
        return future

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            future, call = job
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(call())
                except BaseException as error:
                    future.set_exception(error)
        # This registry is accessed exclusively on this lane's thread.
        for router in self.routers.values():
            try:
                router.close()
            except Exception as error:
                logger.warning(f"Rollout tool environment teardown failed: {error}")
        self.routers.clear()

    def shutdown(self, wait: bool = True) -> None:
        """Drain queued calls, then close remaining environments on their owner."""
        with self._lock:
            if not self._stopped:
                self._stopped = True
                self._queue.put(None)
        if wait and current_thread() is not self._thread:
            self._thread.join()


@cache
def _lanes() -> Iterator[_Lane]:
    """Bound process-wide tool threads while retaining per-router affinity."""
    return cycle(_Lane(f"oumi-tool-{i}") for i in range(8))


def next_lane() -> _Lane:
    """Assign a rollout to one of eight FIFO lanes for its whole lifetime."""
    return next(_lanes())


class RouterWorker:
    """Build, drive and close a router on one owning thread.

    A default asyncio executor cannot preserve SQLite thread affinity. A worker
    serializes tools within a rollout while independent rollouts can progress.
    It must be closed when the rollout owner is released, including on failure.
    """

    def __init__(
        self,
        build: Callable[[], ToolRouter],
        lane: _Lane | None = None,
    ) -> None:
        """Schedule environment construction without blocking the event loop."""
        self._owns_lane = lane is None
        self._lane = lane or _Lane("oumi-tool")
        self._router = self._lane.submit(partial(self._build, build))
        self._closing: Future[None] | None = None

    def _build(self, build: Callable[[], ToolRouter]) -> ToolRouter:
        router = build()
        self._lane.routers[id(router)] = router
        return router

    def _route(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute after construction, on the same worker thread."""
        return self._router.result().route_batch([(tool_id, arguments)])[0]

    async def execute(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Await a serialized call without occupying the rollout event loop."""
        if self._closing is not None:
            raise RuntimeError("Rollout tool worker is closed.")
        return await asyncio.wrap_future(
            self._lane.submit(partial(self._route, tool_id, arguments))
        )

    def _close(self) -> None:
        """Close after queued work; a failed builder owns its own cleanup."""
        if self._router.exception() is None:
            router = self._router.result()
            self._lane.routers.pop(id(router), None)
            router.close()

    def close(self) -> Future[None]:
        """Queue teardown on the owner thread and stop accepting new work.

        Running calls cannot be preempted by asyncio cancellation. Closing is
        queued behind them, so a cancelled rollout cannot close an active DB.
        This method never waits on the event loop and is idempotent.
        """
        if self._closing is None:
            self._closing = self._lane.submit(self._close)
            self._closing.add_done_callback(self._report_close_error)
            if self._owns_lane:
                self._lane.shutdown(wait=False)
        return self._closing

    @staticmethod
    def _report_close_error(future: Future[None]) -> None:
        """Report teardown failures even when GC initiated the cleanup."""
        if error := future.exception():
            logger.warning(f"Rollout tool environment teardown failed: {error}")
