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
from collections.abc import Callable
from concurrent.futures import Future
from functools import partial
from itertools import cycle
from queue import SimpleQueue
from threading import Lock, Thread, current_thread
from typing import Any, TypeVar

from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.types.tool_call import ToolResult
from oumi.utils.logging import logger

_T = TypeVar("_T")

# Lanes are process-wide, so shutdown must bound it or one stuck tool blocks
# interpreter exit forever.
_SHUTDOWN_JOIN_TIMEOUT_SECONDS = 30.0


class _Lane:
    """FIFO worker that closes live routers on its own thread at process exit.

    ThreadPoolExecutor stops workers before ordinary atexit callbacks, so a GC
    finalizer cannot submit thread-affine teardown there during interpreter exit.

    `submit` may be reached reentrantly: `Queue.put` allocates, an allocation can
    trigger a GC pass, and that pass runs a `weakref.finalize` callback for
    another rollout assigned to this same lane. Locking there would self-deadlock,
    so the queue is a lock-free `SimpleQueue` (documented as reentrant and safe
    from `__del__`/weakref callbacks) and shutdown is a plain flag write.
    """

    def __init__(self, name: str) -> None:
        self.routers: dict[int, ToolRouter] = {}
        self._queue: SimpleQueue[_Job | None] = SimpleQueue()
        self._stopped = False
        self._thread = Thread(target=self._run, name=name, daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def submit(self, call: Callable[[], _T]) -> Future[_T]:
        """Enqueue work; safe to call from a GC callback on any thread."""
        future: Future[_T] = Future()
        if self._stopped:
            raise RuntimeError("Tool execution lane has shut down.")
        # `SimpleQueue` is documented as reentrant and safe to use from `__del__`
        # and weakref callbacks, and `put` never blocks, so a finalizer that
        # re-enters here cannot deadlock against the caller.
        self._queue.put(_Job(future, call))
        if self._stopped:
            # `shutdown` raced in between the check and the put, so the lane may
            # exit without running this job. Cancel it so the caller observes a
            # cancelled future instead of waiting on one that never completes.
            future.cancel()
        return future

    def _run(self) -> None:
        while (job := self._queue.get()) is not None:
            job.run()
        # This registry is accessed exclusively on this lane's thread.
        for router in self.routers.values():
            try:
                router.close()
            except Exception as error:
                logger.warning(f"Rollout tool environment teardown failed: {error}")
        self.routers.clear()

    def shutdown(self, wait: bool = True) -> None:
        """Drain queued calls, then close remaining environments on their owner."""
        if not self._stopped:
            self._stopped = True
            self._queue.put(None)
        if wait and current_thread() is not self._thread:
            # A tool that never returns must not block interpreter exit; the
            # thread is a daemon, so abandoning it is safe.
            self._thread.join(timeout=_SHUTDOWN_JOIN_TIMEOUT_SECONDS)
            if self._thread.is_alive():
                logger.warning(
                    f"Tool execution lane {self._thread.name} did not finish "
                    f"within {_SHUTDOWN_JOIN_TIMEOUT_SECONDS:.0f}s of shutdown."
                )


class _Job:
    """One queued call, kept out of the futures registry on purpose."""

    __slots__ = ("future", "call")

    def __init__(self, future: Future[Any], call: Callable[[], Any]) -> None:
        self.future = future
        self.call = call

    def run(self) -> None:
        if self.future.set_running_or_notify_cancel():
            try:
                self.future.set_result(self.call())
            except BaseException as error:
                self.future.set_exception(error)


class _LanePool:
    """The process-wide lanes, cycle-ordered.

    Built explicitly rather than through `functools.cache` on a generator:
    two threads calling `next_lane()` for the first time must not each build a
    private cycle (that would create 16 lanes instead of 8).
    """

    def __init__(self, count: int) -> None:
        self._lanes = [_Lane(f"oumi-tool-{i}") for i in range(count)]
        self._rotation = cycle(self._lanes)
        self._lock = Lock()

    def next(self) -> _Lane:
        with self._lock:
            return next(self._rotation)


_POOL_SIZE = 8
_pool: _LanePool | None = None
_pool_lock = Lock()


def _lanes() -> _LanePool:
    """Return the lazy process-wide pool without racing its construction."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _LanePool(_POOL_SIZE)
    return _pool


def next_lane() -> _Lane:
    """Assign a rollout to one of eight FIFO lanes for its whole lifetime."""
    return _lanes().next()


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
        # Set on the lane thread inside `_close`. A GC finalizer and an event
        # loop can both reach `_close`, so teardown must tolerate two runs.
        self._closed = False

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
        if self._closed:
            return
        self._closed = True
        if self._router.cancelled():
            # The lane shut down before the build ran, so there is nothing to close.
            return
        if self._router.exception() is None:
            router = self._router.result()
            self._lane.routers.pop(id(router), None)
            router.close()

    def _queue_teardown(self) -> bool:
        """Publish teardown exactly once, without any lock.

        Reached from the event loop and from `weakref.finalize` callbacks that
        may fire on this very lane. `_Lane.submit` is lock-free by construction
        and `_close` is idempotent, so a losing race is harmless.

        Returns False when the lane had already shut down and swept its routers.
        """
        if self._closing is not None:
            return True
        try:
            future = self._lane.submit(self._close)
        except RuntimeError:
            # Lane already stopped; its `_run` epilogue closed live routers.
            return False
        future.add_done_callback(self._report_close_error)
        self._closing = future
        return True

    def close(self) -> None:
        """Queue teardown on the owner thread and stop accepting new work.

        Running calls cannot be preempted by asyncio cancellation. Closing is
        queued behind them, so a cancelled rollout cannot close an active DB.
        This method never waits on the event loop and is idempotent.
        """
        self._queue_teardown()
        if self._owns_lane:
            self._lane.shutdown(wait=False)

    def close_from_finalizer(self) -> None:
        """Queue teardown from a GC callback without waiting or shutting down.

        Shutdown is deliberately skipped: this can run on the lane's own thread,
        where draining and joining would deadlock against the frame running the
        callback. Process-wide lanes are shut down by `atexit`; a private lane
        has already been shut down by its rollout's `close`.
        """
        self._queue_teardown()

    async def wait_closed(self) -> None:
        """Wait for teardown without letting a cancelled waiter cancel cleanup."""
        self.close()
        assert self._closing is not None
        await asyncio.shield(asyncio.wrap_future(self._closing))

    @staticmethod
    def _report_close_error(future: Future[None]) -> None:
        """Report teardown failures even when GC initiated the cleanup."""
        if error := future.exception():
            logger.warning(f"Rollout tool environment teardown failed: {error}")
