import asyncio
import gc
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.trainers.verl_agentic import env_provider
from oumi.core.trainers.verl_agentic.router_worker import RouterWorker, _Lane, next_lane
from oumi.core.types.tool_call import ToolResult


def test_build_execute_and_close_keep_sqlite_thread_affinity():
    threads = []

    def build():
        connection = sqlite3.connect(":memory:")
        threads.append(threading.get_ident())

        def route(calls):
            threads.append(threading.get_ident())
            return [ToolResult(output=connection.execute("SELECT 42").fetchone()[0])]

        def close():
            threads.append(threading.get_ident())
            connection.close()

        return Mock(route_batch=route, close=close)

    async def run():
        worker = RouterWorker(build)
        try:
            results = await asyncio.gather(
                *(worker.execute("sql", {}) for _ in range(4))
            )
            assert [r.output for r in results] == [42] * 4
        finally:
            await worker.wait_closed()

    asyncio.run(run())
    assert len(threads) == 6
    assert len(set(threads)) == 1
    assert threads[0] != threading.get_ident()


def test_slow_rollout_does_not_block_other_rollouts_or_event_loop():
    started, finish = threading.Event(), threading.Event()

    def slow(calls):
        started.set()
        assert finish.wait(timeout=5)
        return [ToolResult(output="slow")]

    async def run():
        a = RouterWorker(lambda: Mock(route_batch=slow))
        b = RouterWorker(
            lambda: Mock(route_batch=lambda calls: [ToolResult(output="fast")])
        )
        task = asyncio.create_task(a.execute("slow", {}))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            result = await asyncio.wait_for(b.execute("fast", {}), timeout=2)
            assert result.output == "fast"
            assert not task.done()
        finally:
            finish.set()
            await task
            await asyncio.gather(*(w.wait_closed() for w in (a, b)))

    asyncio.run(run())


def test_cancelled_call_finishes_before_close_and_next_call():
    started, finish = threading.Event(), threading.Event()
    actions = []

    def route(calls):
        actions.append("start")
        started.set()
        assert finish.wait(timeout=5)
        actions.append("end")
        return [ToolResult(output="done")]

    async def run():
        worker = RouterWorker(
            lambda: Mock(route_batch=route, close=lambda: actions.append("close"))
        )
        task = asyncio.create_task(worker.execute("slow", {}))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        worker.close()
        closing = worker._closing
        worker.close()
        assert closing is worker._closing
        assert closing is not None
        assert not closing.done()
        finish.set()
        await worker.wait_closed()
        with pytest.raises(RuntimeError, match="closed"):
            await worker.execute("slow", {})

    asyncio.run(run())
    assert actions == ["start", "end", "close"]


def test_builder_failure_is_returned_and_worker_can_close():
    def build():
        raise ValueError("bad database")

    async def run():
        worker = RouterWorker(build)
        try:
            with pytest.raises(ValueError, match="bad database"):
                await worker.execute("sql", {})
        finally:
            await worker.wait_closed()

    asyncio.run(run())


def test_cancelled_queued_call_does_not_execute_side_effects():
    started, finish = threading.Event(), threading.Event()
    executed = []

    def route(calls):
        name = calls[0][0]
        executed.append(name)
        if name == "running":
            started.set()
            assert finish.wait(timeout=5)
        return [ToolResult(output=name)]

    async def run():
        worker = RouterWorker(lambda: Mock(route_batch=route))
        running = asyncio.create_task(worker.execute("running", {}))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            queued = asyncio.create_task(worker.execute("cancelled", {}))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
        finally:
            finish.set()
            await running
            await worker.wait_closed()
        assert executed == ["running"]

    asyncio.run(run())


def test_shared_pool_keeps_other_lanes_responsive():
    started, finish = threading.Event(), threading.Event()
    lanes = [next_lane() for _ in range(9)]
    assert lanes[0] is lanes[8]
    assert lanes[0] is not lanes[1]

    def slow(calls):
        started.set()
        assert finish.wait(timeout=5)
        return [ToolResult(output="slow")]

    async def run():
        blocked = RouterWorker(lambda: Mock(route_batch=slow), lanes[0])
        independent = RouterWorker(
            lambda: Mock(route_batch=lambda calls: [ToolResult(output="fast")]),
            lanes[1],
        )
        running = asyncio.create_task(blocked.execute("slow", {}))
        queued = None
        same_lane = None
        try:
            assert await asyncio.to_thread(started.wait, 2)
            same_lane = RouterWorker(lambda: Mock(), lanes[8])
            queued = asyncio.create_task(same_lane.execute("queued", {}))
            result = await asyncio.wait_for(independent.execute("fast", {}), 2)
            assert result.output == "fast"
            assert not running.done()
            assert not queued.done()
        finally:
            if queued is not None:
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
            finish.set()
            await running
            workers = [blocked, independent]
            if same_lane is not None:
                workers.append(same_lane)
            await asyncio.gather(*(w.wait_closed() for w in workers))

    asyncio.run(run())


def test_cancelled_teardown_waiter_does_not_cancel_shared_lane_cleanup():
    started, finish, closed = threading.Event(), threading.Event(), threading.Event()
    lane = next_lane()

    def route(calls):
        started.set()
        assert finish.wait(timeout=5)
        return [ToolResult(output="done")]

    async def run():
        worker = RouterWorker(lambda: Mock(route_batch=route, close=closed.set), lane)
        call = asyncio.create_task(worker.execute("sql", {}))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            worker.close()
            waiting = asyncio.create_task(worker.wait_closed())
            await asyncio.sleep(0)
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
        finally:
            finish.set()
            await call
            await worker.wait_closed()
        assert closed.is_set()

    asyncio.run(run())


def test_worker_is_shared_and_gc_closes_without_retaining_owner():
    class Owner(SimpleNamespace):
        request_id: str
        tools_kwargs: dict[str, Any]

    closed = threading.Event()
    parent = Mock(tool_env_map={"sql": "db"})
    parent.for_sample.return_value = Mock(close=closed.set)
    owner = Owner(request_id="threaded-gc", tools_kwargs={})
    worker = env_provider.get_or_build_worker(owner, parent)
    assert env_provider.get_or_build_worker(owner, parent) is worker
    with pytest.raises(ValueError, match="single environment config"):
        env_provider.get_or_build_worker(owner, Mock())
    with pytest.raises(ValueError, match="execution_mode"):
        env_provider.get_or_build_router(owner, parent)
    del owner
    gc.collect()
    assert closed.wait(timeout=5)
    assert "threaded-gc" not in env_provider._WORKERS


def test_many_rollouts_share_bounded_lanes_without_losing_affinity():
    class Owner(SimpleNamespace):
        request_id: str
        tools_kwargs: dict[str, Any]

    thread_ids = set()
    closed = []

    def build(overrides):
        owner_thread = threading.get_ident()
        thread_ids.add(owner_thread)

        def route(calls):
            assert threading.get_ident() == owner_thread
            return [ToolResult(output=42)]

        def close():
            assert threading.get_ident() == owner_thread
            closed.append(True)

        return Mock(route_batch=route, close=close)

    parent = Mock(tool_env_map={}, for_sample=build)

    async def run():
        owners = [Owner(request_id=f"bounded-{i}", tools_kwargs={}) for i in range(40)]
        workers = [env_provider.get_or_build_worker(o, parent) for o in owners]
        try:
            results = await asyncio.gather(*(w.execute("sql", {}) for w in workers))
            assert [r.output for r in results] == [42] * 40
        finally:
            await asyncio.gather(*(w.wait_closed() for w in workers))

    asyncio.run(run())
    assert len(thread_ids) <= 8
    assert len(closed) == 40


def test_process_exit_closes_live_sqlite_router_on_owning_thread(tmp_path):
    marker = tmp_path / "closed"
    code = f"""
import sqlite3
import threading
from pathlib import Path
from unittest.mock import Mock
from oumi.core.trainers.verl_agentic.router_worker import RouterWorker

def build():
    connection = sqlite3.connect(':memory:')
    owner = threading.get_ident()
    def close():
        assert threading.get_ident() == owner
        connection.close()
        Path({str(marker)!r}).write_text('closed', encoding='utf-8')
    return Mock(close=close)

worker = RouterWorker(build)
worker._router.result(timeout=10)
# Intentionally leave the rollout alive until interpreter shutdown.
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert "cannot schedule new futures" not in result.stderr
    assert marker.read_text(encoding="utf-8") == "closed"


class _QueueFiringCallbackInsidePut:
    """Delegates to a real queue but runs a callback from inside `put`.

    Models the reviewer's deadlock precisely: `Queue.put` allocates, the
    allocation can trigger a GC pass, and that pass runs a `weakref.finalize`
    callback for another rollout on the same lane. Making the callback run
    inside `put` reproduces the reentrancy without depending on allocation
    accounting.
    """

    def __init__(self, inner, on_put):
        self._inner = inner
        self._on_put = on_put
        self.fired = False

    def put(self, item):
        self._inner.put(item)
        if not self.fired:
            self.fired = True
            self._on_put()

    def get(self):
        return self._inner.get()


def test_finalizer_during_put_does_not_deadlock_the_lane():
    """GC teardown of a same-lane rollout must not re-enter a held lane lock."""
    closed = threading.Event()

    def build():
        return Mock(route_batch=lambda calls: ["ok"], close=closed.set)

    lane = _Lane("oumi-tool-reentrancy")
    sibling = RouterWorker(build, lane=lane)
    victim = RouterWorker(build, lane=lane)
    env_provider._WORKERS["victim"] = (0, victim)

    original_queue = lane._queue
    lane._queue = _QueueFiringCallbackInsidePut(  # type: ignore[assignment]
        original_queue, lambda: env_provider._teardown_worker("victim")
    )
    try:
        # If the lane re-enters a lock it already holds this never returns.
        lane.submit(lambda: "outer").result(timeout=10)
    finally:
        lane._queue = original_queue
        env_provider._WORKERS.pop("victim", None)
        lane.shutdown()

    assert sibling is not None
    assert closed.wait(timeout=5)


def test_shutdown_does_not_block_on_a_stuck_tool(monkeypatch):
    """A tool that never returns must not hold interpreter exit forever."""
    started, release = threading.Event(), threading.Event()

    def stuck():
        started.set()
        assert release.wait(timeout=30)
        return "late"

    lane = _Lane("oumi-tool-stuck")
    lane.submit(stuck).exception()  # noqa: B018 - dispatched to the lane
    assert started.wait(timeout=5)

    monkeypatch.setattr(
        "oumi.core.trainers.verl_agentic.router_worker._SHUTDOWN_JOIN_TIMEOUT_SECONDS",
        0.05,
    )
    began = threading.Event()

    def timed_shutdown():
        began.set()
        lane.shutdown()

    t = threading.Thread(target=timed_shutdown, daemon=True)
    t.start()
    assert began.wait(timeout=5)
    t.join(timeout=10)
    # The bounded join returns even though the tool still holds the lane.
    assert not t.is_alive()
    release.set()


def test_lane_pool_is_built_once_under_concurrent_first_use():
    """Concurrent first use must not build competing pools (16 lanes, not 8)."""
    import oumi.core.trainers.verl_agentic.router_worker as rw

    saved = rw._pool
    rw._pool = None
    try:
        barrier = threading.Barrier(8)
        seen: list[int] = []
        lock = threading.Lock()

        def grab():
            barrier.wait(timeout=10)
            lane = rw.next_lane()
            with lock:
                seen.append(id(lane))

        threads = [threading.Thread(target=grab) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        # Every caller must draw from the same eight-lane rotation.
        assert len(set(seen)) == 8
        pool = rw._lanes()
        assert len(pool._lanes) == 8
    finally:
        built = rw._pool
        for lane in [] if built is None else built._lanes:
            lane.shutdown(wait=False)
        rw._pool = saved


def test_finalizer_after_lane_shutdown_is_a_no_op():
    """Teardown must not raise when the lane already stopped and swept routers."""
    lane = _Lane("oumi-tool-stopped")
    worker = RouterWorker(lambda: Mock(close=lambda: None), lane=lane)
    lane.shutdown()
    # Pending build never ran, so the lane cancelled the future; teardown must
    # be silent rather than surfacing a CancelledError to the interpreter.
    worker.close_from_finalizer()
    assert worker._closing is None


def test_thread_mode_rejects_simulated_environment():
    """Thread mode cannot drive an asyncio-bound simulator engine."""
    cfg = EnvironmentConfig(
        environments=[
            EnvironmentParams(
                id="sim",
                env_type="simulated",
                env_kwargs={"tool_persona": "you are a helpful tool"},
                tools=[
                    {
                        "id": "lookup",
                        "name": "lookup",
                        "description": "simulated",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            )
        ]
    )
    parent = ToolRouter.from_environment_config(cfg)
    ad = SimpleNamespace(request_id="sim-1", tools_kwargs={})
    with pytest.raises(ValueError, match="asyncio-bound inference engine"):
        env_provider.get_or_build_worker(ad, parent)  # type: ignore[arg-type]
