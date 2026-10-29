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

from oumi.core.trainers.verl_agentic import env_provider
from oumi.core.trainers.verl_agentic.router_worker import RouterWorker, next_lane
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
