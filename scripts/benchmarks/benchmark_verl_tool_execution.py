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
#

"""Measure real Oumi router scheduling with SQLite and controlled blocking I/O."""

import argparse
import asyncio
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.synthesis.tool_router import ToolRouter
from oumi.core.trainers.verl_agentic.env_provider import (
    get_or_build_router,
    get_or_build_worker,
)
from oumi.core.types.tool_call import ToolResult


def query(arguments, context):
    """Run a read-only SQLite query with an optional controlled I/O delay."""
    context.create_function("blocking_io", 1, time.sleep)
    return ToolResult(
        output=[list(r) for r in context.execute(arguments["query"]).fetchall()]
    )


class Owner(SimpleNamespace):
    """Keep rollout ownership weak-referenceable, like VERL's AgentData."""

    request_id: str
    tools_kwargs: dict[str, Any]


async def trial(mode, n, workload, parent):
    """Measure cold construction and three warm tool rounds separately."""
    owners = [Owner(request_id=uuid4().hex, tools_kwargs={}) for _ in range(n)]
    latencies, heartbeats = [], []
    sql = {
        "io": "SELECT blocking_io(0.05)",
        "sqlite": (
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 "
            "FROM c WHERE x<200000) SELECT sum(x) FROM c"
        ),
        "cheap": "SELECT 42",
    }[workload]
    done = False

    async def heartbeat():
        """Track scheduling delays while tool execution is in progress."""
        while not done:
            before = time.perf_counter()
            await asyncio.sleep(0.001)
            heartbeats.append(max(0, time.perf_counter() - before - 0.001))

    async def rollout(owner):
        """Execute one request through the selected Oumi provider."""
        start = time.perf_counter()
        if mode == "thread":
            router = get_or_build_worker(owner, parent)
            result = await router.execute("sql", {"query": sql})
        else:
            router = get_or_build_router(owner, parent)
            result = router.route_batch([("sql", {"query": sql})])[0]
        latencies.append(time.perf_counter() - start)
        return result.output

    beat = asyncio.create_task(heartbeat())
    before = time.perf_counter()
    outputs = await asyncio.gather(*(rollout(o) for o in owners))
    elapsed = time.perf_counter() - before
    # Keep environments alive and measure subsequent turns separately, so cold
    # database-file initialization cannot hide warm dispatch overhead.
    warm_start = time.perf_counter()
    for _ in range(3):
        warm_outputs = await asyncio.gather(*(rollout(o) for o in owners))
        assert warm_outputs == outputs
    warm_elapsed = time.perf_counter() - warm_start
    done = True
    await beat
    if mode == "thread":
        await asyncio.gather(
            *(get_or_build_worker(o, parent).wait_closed() for o in owners)
        )
    owners.clear()
    gc.collect()
    assert outputs == [outputs[0]] * n
    return {
        "mode": mode,
        "rollouts": n,
        "workload": workload,
        "wall_seconds": elapsed,
        "warm_three_turn_seconds": warm_elapsed,
        "rollouts_per_second": n / elapsed,
        "max_event_loop_lag_seconds": max(heartbeats),
        "median_execution_seconds": statistics.median(latencies),
        "outputs": outputs,
    }


async def main():
    """Compare both modes with identical queries and alternating run order."""
    # The source itself is an importable executor module for EnvironmentParams.
    sys.modules["benchmark"] = sys.modules[__name__]
    cfg = EnvironmentConfig(
        environments=[
            EnvironmentParams(
                id="db",
                name="db",
                description="benchmark",
                env_type="database",
                env_kwargs={"schema_sql": "CREATE TABLE t(x);"},
                tools=[
                    {
                        "id": "sql",
                        "name": "sql",
                        "description": "query",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                        "executor": "benchmark.query",
                        "read_only": True,
                    }
                ],
            )
        ]
    )
    parent = ToolRouter.from_environment_config(cfg)
    results = []
    for workload in ("io", "sqlite", "cheap"):
        for n in (1, 8, 32):
            for repeat in range(3):
                order = ("sync", "thread") if repeat % 2 == 0 else ("thread", "sync")
                pair = []
                for mode in order:
                    result = await trial(mode, n, workload, parent)
                    result["repeat"] = repeat
                    results.append(result)
                    pair.append(result)
                assert pair[0]["outputs"] == pair[1]["outputs"]
            subset = [
                r for r in results if r["workload"] == workload and r["rollouts"] == n
            ]
            print(
                workload,
                n,
                {
                    m: round(
                        statistics.median(
                            r["wall_seconds"] for r in subset if r["mode"] == m
                        ),
                        4,
                    )
                    for m in ("sync", "thread")
                },
                flush=True,
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(asyncio.run(main()), indent=2), encoding="utf-8")
